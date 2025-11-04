#!/usr/bin/env python3
"""
app.py - Single-file Solana Telegram bot with:
- RPC-only operation (No Helius/Jupiter required)
- Raydium and Orca/Whirlpool pool parsers for reserve extraction
- On-chain reserve-based price computation + CoinGecko fallback
- Heuristic / RandomForest predictor with SQLite-backed interactions
- Telegram webhook handlers + approval/publish flow
"""

import os, time, base64, json, sqlite3, math, random, threading
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import requests
from flask import Flask, request
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, CallbackQueryHandler
import numpy as np
import joblib

# Optional ML libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --------------------
# Config
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN required")

RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # optional
ADMIN_CHAT = os.getenv("ADMIN_CHAT")    # optional
PORT = int(os.getenv("PORT", "8080"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1.6"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "120"))
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/pocket_model.joblib")
DB_PATH = os.getenv("DB_PATH", "/tmp/pocket.db")
TRAIN_ON_START = os.getenv("TRAIN_ON_START", "true").lower() in ("1","true","yes")

COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
SIMPLE_PRICE = "https://api.coingecko.com/api/v3/simple/price"

# --------------------
# App & bot
# --------------------
app = Flask(__name__)
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

# --------------------
# Simple caches
# --------------------
_cache = {}
def cache_get(k):
    v = _cache.get(k)
    if not v: return None
    val, exp = v
    if datetime.utcnow() > exp:
        del _cache[k]; return None
    return val

def cache_set(k, v, ttl=CACHE_TTL):
    _cache[k] = (v, datetime.utcnow() + timedelta(seconds=ttl))

_price_cache = {}
def price_cache_get(k):
    v = _price_cache.get(k)
    if not v: return None
    val, exp = v
    if datetime.utcnow() > exp:
        del _price_cache[k]; return None
    return val

def price_cache_set(k, v, ttl=30):
    _price_cache[k] = (v, datetime.utcnow() + timedelta(seconds=ttl))

# --------------------
# DB (SQLite)
# --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        mint TEXT,
        features TEXT,
        label INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        chat TEXT,
        mint TEXT,
        action TEXT,
        probs TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS published (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        mint TEXT,
        probs TEXT
    )""")
    conn.commit(); conn.close()

init_db()

def db_insert_interaction(mint, feats, label=0):
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("INSERT INTO interactions (ts,mint,features,label) VALUES (?,?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(feats), int(label)))
    conn.commit(); conn.close()

def db_insert_approval(chat, mint, action, probs):
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("INSERT INTO approvals (ts,chat,mint,action,probs) VALUES (?,?,?,?,?)",
              (datetime.utcnow().isoformat(), str(chat), mint, action, json.dumps(probs)))
    conn.commit(); conn.close()

def db_insert_published(mint, probs):
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("INSERT INTO published (ts,mint,probs) VALUES (?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(probs)))
    conn.commit(); conn.close()

def db_fetch_training():
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("SELECT features,label FROM interactions")
    rows=c.fetchall(); conn.close()
    X=[]; y=[]
    for feats_json,label in rows:
        feats=json.loads(feats_json)
        X.append([feats.get("top10_pct_inv",0.0), feats.get("supply_inv",0.0), feats.get("activity",0.0)])
        y.append(int(label))
    if not X:
        return None,None
    return np.array(X), np.array(y)

# --------------------
# HTTP helpers with retries/backoff
# --------------------
def backoff_sleep(attempt: int):
    time.sleep((BACKOFF_FACTOR ** attempt) * 0.3)

def http_post(url, payload, headers=None):
    last=None
    for i in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}")
                backoff_sleep(i); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; backoff_sleep(i)
    raise last

def http_get(url, params=None, headers=None):
    last=None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}")
                backoff_sleep(i); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; backoff_sleep(i)
    raise last

# --------------------
# RPC helpers
# --------------------
def rpc_post(method, params):
    payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
    res = http_post(RPC_URL, payload)
    return res.get("result")

def get_token_largest_accounts(mint):
    return rpc_post("getTokenLargestAccounts", [mint])

def get_token_supply(mint):
    return rpc_post("getTokenSupply", [mint])

def get_token_account_balance(token_account):
    return rpc_post("getTokenAccountBalance", [token_account])

def get_account_info_base64(pubkey):
    return rpc_post("getAccountInfo", [pubkey, {"encoding":"base64"}])

# --------------------
# Featurize & model
# --------------------
MODEL = None

def featurize(mint, metrics):
    top10 = metrics.get("top10_pct") or 0.0
    supply = metrics.get("supply") or 0.0
    top10_inv = max(0.0, (100.0 - top10)/100.0)
    try:
        s_log = math.log10(supply+1) if supply>0 else 0.0
        supply_inv = 1.0/(s_log+1.0)
    except Exception:
        supply_inv = 0.0
    activity = (metrics.get("recent_activity",0) or 0)/100.0
    feats = {"top10_pct_inv":top10_inv, "supply_inv":supply_inv, "activity":activity}
    return feats

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
        except Exception:
            MODEL = None

def train_model():
    if not SKLEARN_AVAILABLE:
        return None
    X,y = db_fetch_training()
    if X is None or len(X)==0:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    joblib.dump(clf, MODEL_PATH)
    load_model()
    return score

def predict(mint, metrics):
    feats = featurize(mint, metrics)
    X = [feats["top10_pct_inv"], feats["supply_inv"], feats["activity"]]
    if MODEL is None:
        base = 40*feats["top10_pct_inv"] + 25*feats["supply_inv"] + 10*feats["activity"]
        p1 = max(0,min(95, base + random.random()*3))
        p2 = p1*0.75; p3 = p1*0.55
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1), "model":"heuristic"}
    try:
        probs = MODEL.predict_proba([X])[0]
        uplift = probs[1] if len(probs)>1 else probs[0]
        p1 = uplift*100; p2=p1*0.75; p3=p1*0.55
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1), "model":"rf"}
    except Exception:
        return {"1m":0.0,"2m":0.0,"3m":0.0,"model":"error"}

# --------------------
# Token metrics
# --------------------
def fetch_token_metrics(mint):
    cache_key = f"metrics:{mint}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    out = {"mint":mint}
    try:
        largest = get_token_largest_accounts(mint) or {}
        supply_res = get_token_supply(mint) or {}
        supply = None
        try:
            if isinstance(supply_res, dict) and "value" in supply_res:
                supply = float(supply_res["value"].get("uiAmount") or supply_res["value"].get("uiAmountString") or 0.0)
        except Exception:
            supply=None
        out["supply"]=supply
        holders=[]
        try:
            val = largest.get("value") if isinstance(largest, dict) else largest
            if isinstance(val, list):
                for ent in val:
                    if isinstance(ent, dict):
                        amt = ent.get("uiAmount") or ent.get("amount")
                        holders.append({"address":ent.get("address"), "amount":amt})
        except Exception:
            pass
        out["holders"]=holders
        if supply and holders:
            topn = sum((h.get("amount") or 0.0) for h in holders[:10])
            out["top10_pct"]= (topn / supply * 100.0) if supply>0 else None
        else:
            out["top10_pct"]=None
    except Exception:
        out["supply"]=None; out["holders"]=[]; out["top10_pct"]=None
    cache_set(cache_key, out, ttl=60)
    return out

# --------------------
# Price: reserve-based + CoinGecko
# --------------------
def get_token_account_ui_amount(token_account_pubkey):
    cache_key = f"tkbal:{token_account_pubkey}"
    cached = price_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        bal = get_token_account_balance(token_account_pubkey)
        if not bal or "value" not in bal:
            return None
        val = bal["value"]
        if val.get("uiAmountString") is not None:
            v = Decimal(val["uiAmountString"]); price_cache_set(cache_key, v); return v
        if val.get("uiAmount") is not None:
            v = Decimal(str(val["uiAmount"])); price_cache_set(cache_key, v); return v
        amt = Decimal(val.get("amount",0)); dec = int(val.get("decimals",0))
        v = amt / (Decimal(10) ** dec); price_cache_set(cache_key, v); return v
    except Exception:
        return None

def price_from_reserves(tokenA_acc, tokenB_acc, min_liquidity_ui=Decimal("0.0001")):
    cache_key = f"res:{tokenA_acc}:{tokenB_acc}"
    cached = price_cache_get(cache_key)
    if cached is not None:
        return cached
    a = get_token_account_ui_amount(tokenA_acc)
    b = get_token_account_ui_amount(tokenB_acc)
    if a is None or b is None or a == 0:
        return None
    if b < min_liquidity_ui:
        return None
    price = (b / a)
    price_cache_set(cache_key, price, ttl=30)
    return price

def coingecko_price_by_contract(contract_address, vs_currency="usd"):
    cache_key = f"cg:{contract_address}:{vs_currency}"
    cached = price_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        params = {"vs_currency": vs_currency, "contract_addresses": contract_address, "platform":"solana"}
        r = http_get(COINGECKO_MARKETS, params=params)
        if not r: return None
        if isinstance(r, list) and len(r)>0:
            p = r[0].get("current_price")
            if p is not None:
                d = Decimal(str(p)); price_cache_set(cache_key, d); return d
    except Exception:
        pass
    return None

# --------------------
# Raydium pool parser (common CPMM layouts)
# --------------------
import struct

def parse_raydium_pool_accounts(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    """
    Best-effort: fetch pool account binary and decode known Raydium V3/V4 layout fields:
    Many Raydium pools store tokenAccountA and tokenAccountB pubkeys at predictable offsets.
    This function tries a sequence of known offsets and returns the first plausible pair whose balances read successfully.
    """
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data", [None])[0]
        if not data_b64:
            return None
        raw = base64.b64decode(data_b64)
        # Known offsets for Raydium V4-like pool (community examples): token account A/B often near offsets 72..140 but vary
        candidate_offsets = [72, 88, 104, 120, 136, 160, 200, 232]
        candidates = []
        for off in candidate_offsets:
            if off+64 <= len(raw):
                a = raw[off:off+32]; b = raw[off+32:off+64]
                candidates.append((a,b))
        # Validate by attempting to read token account balances (base58 encode bytes)
        for a_bytes,b_bytes in candidates:
            a_pub = _b58encode(a_bytes); b_pub = _b58encode(b_bytes)
            try:
                ba = get_token_account_balance(a_pub)
                bb = get_token_account_balance(b_pub)
                if ba and bb and "value" in ba and "value" in bb:
                    return a_pub, b_pub
            except Exception:
                continue
    except Exception:
        return None
    return None

# --------------------
# Orca Whirlpool parser (best-effort)
# --------------------
def parse_orca_whirlpool(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    """
    Whirlpool accounts (Orca) have specific struct layouts. This function is a best-effort that
    scans for token vault pubkeys inside account data using heuristics and validates balances.
    """
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data", [None])[0]
        if not data_b64:
            return None
        raw = base64.b64decode(data_b64)
        # Heuristic: scan data for 32-byte sequences with low zero count and validate as token accounts.
        candidates = []
        for i in range(0, max(1, len(raw)-32)):
            part = raw[i:i+32]
            if part.count(b'\x00') > 20:
                continue
            candidates.append(part)
            if len(candidates) > 24:
                break
        tested = []
        for b in candidates:
            pub = _b58encode(b)
            try:
                bal = get_token_account_balance(pub)
                if bal and "value" in bal:
                    tested.append(pub)
            except Exception:
                continue
        # pick first plausible pair
        for i in range(len(tested)):
            for j in range(i+1, len(tested)):
                if tested[i]==tested[j]: continue
                a = get_token_account_balance(tested[i]); b = get_token_account_balance(tested[j])
                if a and b and "value" in a and "value" in b:
                    return tested[i], tested[j]
    except Exception:
        return None
    return None

# --------------------
# Base58 encode helper (small, avoids external lib)
# --------------------
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def _b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    if n == 0:
        return "1"
    out = []
    while n:
        n, r = divmod(n, 58)
        out.append(ALPHABET[r])
    # leading zeros
    for ch in b:
        if ch == 0:
            out.append(ALPHABET[0])
        else:
            break
    return "".join(reversed(out))

# --------------------
# Combined pool discovery: Raydium -> Orca -> fallback binary scan -> None
# --------------------
def discover_reserve_accounts_for_pool(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    # Try Raydium parser
    r = parse_raydium_pool_accounts(pool_account_pubkey)
    if r: return r
    # Try Orca whirlpool parser
    o = parse_orca_whirlpool(pool_account_pubkey)
    if o: return o
    # Generic fallback: scan for 32-byte sequences and test as token accounts
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data", [None])[0]; raw = base64.b64decode(data_b64)
        candidates = []
        for i in range(0, max(1, len(raw)-32)):
            chunk = raw[i:i+32]
            if chunk.count(b'\x00') > 20: continue
            candidates.append(chunk)
            if len(candidates) >= 40: break
        tested=[]
        for c in candidates:
            pub = _b58encode(c)
            try:
                bal = get_token_account_balance(pub)
                if bal and "value" in bal:
                    tested.append(pub)
            except Exception:
                continue
        for i in range(len(tested)):
            for j in range(i+1, len(tested)):
                a = get_token_account_balance(tested[i]); b = get_token_account_balance(tested[j])
                if a and b and "value" in a and "value" in b:
                    return tested[i], tested[j]
    except Exception:
        return None
    return None

# --------------------
# Candidate discovery & approval flow
# --------------------
PENDING = {}  # chat -> {"candidates":[], "i":int}

def discover_candidates(seed_list: List[str], top_k=8) -> List[dict]:
    out=[]
    for m in seed_list:
        try:
            metrics = fetch_token_metrics(m)
            probs = predict(m, metrics)
            out.append({"mint":m, "metrics":metrics, "probs":probs})
        except Exception:
            continue
    out_sorted = sorted(out, key=lambda x: x["probs"]["1m"], reverse=True)
    return out_sorted[:top_k]

def send_candidate_prompt(chat_id):
    s = PENDING.get(str(chat_id))
    if not s:
        bot.send_message(chat_id=chat_id, text="No pending session.")
        return
    idx = s["i"]; cand = s["candidates"][idx]
    total = len(s["candidates"])
    m = cand["mint"]; p=cand["probs"]; top10=cand["metrics"].get("top10_pct")
    text = f"[{idx+1}/{total}] Mint: {m}\nTop10%: {top10}\n1m:{p['1m']}% 2m:{p['2m']}% 3m:{p['3m']}%\nApprove?"
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Yes", callback_data=f"approve|{idx}"), InlineKeyboardButton("❌ No", callback_data=f"reject|{idx}")],
        [InlineKeyboardButton("Next", callback_data=f"next|{idx}"), InlineKeyboardButton("Stop", callback_data=f"stop|{idx}")]
    ])
    bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)

def start_approval(chat_id, seeds: List[str]):
    cands = discover_candidates(seeds)
    if not cands:
        bot.send_message(chat_id=chat_id, text="No candidates found.")
        return
    PENDING[str(chat_id)] = {"candidates":cands, "i":0}
    send_candidate_prompt(chat_id)

def handle_callback(update, context):
    q = update.callback_query
    if not q:
        return
    data = q.data or ""
    parts = data.split("|"); action = parts[0]; idx = int(parts[1]) if len(parts)>1 else 0
    chat = str(q.message.chat.id)
    state = PENDING.get(chat)
    if not state:
        q.answer("No active session"); return
    candidates = state["candidates"]
    if idx<0 or idx>=len(candidates):
        q.answer("Out of range"); return
    if action == "approve":
        cand = candidates.pop(idx)
        db_insert_approval(chat, cand["mint"], "approved", cand["probs"])
        db_insert_published(cand["mint"], cand["probs"])
        q.message.reply_text(f"Published: {cand['mint']} 1m:{cand['probs']['1m']}%")
        if ADMIN_CHAT:
            bot.send_message(chat_id=ADMIN_CHAT, text=f"PUBLISHED {cand['mint']} -> {cand['probs']}")
        if not candidates:
            del PENDING[chat]; q.answer("Approved"); return
        state["i"] = min(idx, len(candidates)-1); send_candidate_prompt(chat); q.answer("Approved"); return
    if action == "reject":
        cand = candidates.pop(idx)
        db_insert_approval(chat, cand["mint"], "rejected", cand["probs"])
        q.message.reply_text("Rejected.")
        if not candidates:
            del PENDING[chat]; q.answer("Rejected"); return
        state["i"] = min(idx, len(candidates)-1); send_candidate_prompt(chat); q.answer("Rejected"); return
    if action == "next":
        state["i"] = min(idx+1, len(candidates)-1); send_candidate_prompt(chat); q.answer("Next"); return
    if action == "stop":
        del PENDING[chat]; q.message.reply_text("Stopped."); q.answer("Stopped"); return

# --------------------
# Telegram command handlers
# --------------------
def cmd_start(update, context):
    update.message.reply_text("Pocket Option Bot\n/ analyze <mint> [--price]\n/ suggest <mints>\n/ train")

def cmd_status(update, context):
    up = str(datetime.utcnow() - START).split('.')[0]
    update.message.reply_text(f"Running\nUptime: {up}\nRPC: {RPC_URL}\nModel: {'loaded' if MODEL is not None else 'none'}")

def cmd_analyze(update, context):
    args = context.args or []
    if not args:
        update.message.reply_text("Usage: /analyze <TOKEN_MINT> [--price]"); return
    mint = args[0].strip(); want_price = "--price" in args or "-p" in args
    update.message.reply_text(f"Inspecting {mint} ...")
    metrics = fetch_token_metrics(mint)
    probs = predict(mint, metrics)
    lines = [f"Mint: {mint}", f"Top10%: {metrics.get('top10_pct')}", f"Supply: {metrics.get('supply')}", f"1m:{probs['1m']}% 2m:{probs['2m']}% 3m:{probs['3m']}%"]
    if want_price:
        # Try deterministic known pool accounts mapping (maintain this dict for best results)
        known_pools = {}  # e.g. {"<mint>":["<pool_acc1>","<pool_acc2>"]}
        pools = known_pools.get(mint, [])
        price = None
        for pool in pools:
            discovered = discover_reserve_accounts_for_pool(pool)
            if discovered:
                price = price_from_reserves(discovered[0], discovered[1])
                if price: break
        # if still none try CoinGecko
        if price is None:
            price = coingecko_price_by_contract(mint)
        lines.append(f"Price (USD or pool): {price if price is not None else 'not found'}")
    update.message.reply_text("\n".join(lines))
    # Save interaction for future training
    feats = featurize_and_store(mint, metrics)
    db_insert_interaction(mint, feats, label=0)

def cmd_suggest(update, context):
    args = context.args or []
    if args:
        seeds = [s.strip() for s in " ".join(args).replace(",", " ").split() if s.strip()]
    else:
        seeds = []
    chat = update.message.chat.id
    bot.send_message(chat_id=chat, text=f"Discovering among {len(seeds)} seeds. This runs in background.")
    threading.Thread(target=start_approval, args=(str(chat), seeds), daemon=True).start()

def cmd_train(update, context):
    uid = str(update.message.chat.id)
    if ADMIN_CHAT and uid != str(ADMIN_CHAT):
        update.message.reply_text("Admin only")
        return
    update.message.reply_text("Retraining model from DB...")
    score = train_model()
    if score is None:
        update.message.reply_text("Not enough data or sklearn missing")
    else:
        update.message.reply_text(f"Trained; test score {score:.3f}")

def featurize_and_store(mint, metrics):
    feats = featurize(mint, metrics)
    db_insert_interaction(mint, feats, label=0)
    return feats

dispatcher.add_handler(CommandHandler("start", cmd_start))
dispatcher.add_handler(CommandHandler("status", cmd_status))
dispatcher.add_handler(CommandHandler("analyze", cmd_analyze))
dispatcher.add_handler(CommandHandler("suggest", cmd_suggest))
dispatcher.add_handler(CommandHandler("train", cmd_train))
dispatcher.add_handler(CallbackQueryHandler(handle_callback))

# --------------------
# Webhook endpoint
# --------------------
@app.route("/", methods=["POST"])
def webhook():
    try:
        update = request.get_json(force=True)
        dispatcher.process_update(bot.update_class.de_json(update))
        return "", 200
    except Exception as e:
        print("webhook error", e)
        return "", 200

# --------------------
# Startup & helpers
# --------------------
START = datetime.utcnow()

def set_webhook():
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set; not setting webhook")
        return
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook", json={"url": WEBHOOK_URL}, timeout=REQUEST_TIMEOUT)
        print("setWebhook", resp.status_code, resp.text)
    except Exception as e:
        print("setWebhook failed", e)

def heartbeat_loop():
    while True:
        if ADMIN_CHAT:
            try:
                bot.send_message(chat_id=ADMIN_CHAT, text=f"Heartbeat {datetime.utcnow().isoformat()}")
            except Exception:
                pass
        time.sleep(1800)

# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    print("Starting bot with RPC:", RPC_URL)
    set_webhook()
    load_model()
    if TRAIN_ON_START:
        try:
            threading.Thread(target=train_model, daemon=True).start()
        except Exception:
            pass
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
