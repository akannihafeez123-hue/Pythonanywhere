#!/usr/bin/env python3
"""
app.py - Single-file Telegram bot (python-telegram-bot v20+)
Features:
 - Solana RPC helpers (rpc_post)
 - Raydium/Orca parsers + generic reserve discovery
 - On-chain reserve-based price + CoinGecko fallback
 - Heuristic + RandomForest model stored with joblib; SQLite interactions
 - Commands: /start, /status, /analyze, /suggest, /train
 - Interactive approval flow with inline buttons
 - Proper startup: background tasks scheduled via ApplicationBuilder.post_init
 - Polling mode by default; set WEBHOOK_URL for webhook mode
"""

import os
import time
import json
import base64
import sqlite3
import math
import random
import threading
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import requests
import joblib
import numpy as np

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# Optional sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# --------------------
# Config (env)
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN required")

RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # empty means polling
ADMIN_CHAT = os.getenv("ADMIN_CHAT")
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
# Caches
# --------------------
_cache: Dict[str, Any] = {}
def cache_get(k):
    v = _cache.get(k)
    if not v: return None
    val, expiry = v
    if datetime.utcnow() > expiry:
        del _cache[k]; return None
    return val
def cache_set(k, v, ttl=CACHE_TTL):
    _cache[k] = (v, datetime.utcnow() + timedelta(seconds=ttl))

_price_cache: Dict[str, Any] = {}
def price_cache_get(k):
    v = _price_cache.get(k)
    if not v: return None
    val, expiry = v
    if datetime.utcnow() > expiry:
        del _price_cache[k]; return None
    return val
def price_cache_set(k, v, ttl=30):
    _price_cache[k] = (v, datetime.utcnow() + timedelta(seconds=ttl))

# --------------------
# SQLite DB
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
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO interactions (ts,mint,features,label) VALUES (?,?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(feats), int(label)))
    conn.commit(); conn.close()

def db_insert_approval(chat, mint, action, probs):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO approvals (ts,chat,mint,action,probs) VALUES (?,?,?,?,?)",
              (datetime.utcnow().isoformat(), str(chat), mint, action, json.dumps(probs)))
    conn.commit(); conn.close()

def db_insert_published(mint, probs):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO published (ts,mint,probs) VALUES (?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(probs)))
    conn.commit(); conn.close()

def db_fetch_training():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT features,label FROM interactions")
    rows = c.fetchall(); conn.close()
    X=[]; y=[]
    for feats_json,label in rows:
        feats = json.loads(feats_json)
        X.append([feats.get("top10_pct_inv",0.0), feats.get("supply_inv",0.0), feats.get("activity",0.0), feats.get("noise",0.0)])
        y.append(int(label))
    if not X:
        return None, None
    return np.array(X), np.array(y)

# --------------------
# HTTP / RPC helpers
# --------------------
def backoff_sleep(attempt:int):
    time.sleep((BACKOFF_FACTOR ** attempt) * 0.3)

def http_get(url, params=None, headers=None):
    last=None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}"); backoff_sleep(attempt); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; backoff_sleep(attempt)
    raise last

def http_post(url, payload, headers=None):
    last=None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}"); backoff_sleep(attempt); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; backoff_sleep(attempt)
    raise last

def rpc_post(method, params):
    payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
    r = http_post(RPC_URL, payload)
    return r.get("result")

def get_token_largest_accounts(mint):
    return rpc_post("getTokenLargestAccounts", [mint])

def get_token_supply(mint):
    return rpc_post("getTokenSupply", [mint])

def get_token_account_balance(token_account):
    return rpc_post("getTokenAccountBalance", [token_account])

def get_account_info_base64(pubkey):
    return rpc_post("getAccountInfo", [pubkey, {"encoding":"base64"}])

# --------------------
# Featurize + model
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
    noise = random.random()*0.05
    return {"top10_pct_inv": top10_inv, "supply_inv": supply_inv, "activity": activity, "noise": noise}

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
        except Exception:
            MODEL = None

def train_model_from_db():
    if not SKLEARN_OK:
        return None
    X,y = db_fetch_training()
    if X is None:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    joblib.dump(clf, MODEL_PATH)
    load_model()
    return score

def predict_with_model(mint, metrics):
    feats = featurize(mint, metrics)
    X = np.array([[feats["top10_pct_inv"], feats["supply_inv"], feats["activity"], feats["noise"]]])
    if MODEL is None:
        base = 40*feats["top10_pct_inv"] + 25*feats["supply_inv"] + 10*feats["activity"] + 5*feats["noise"]
        p1 = max(0,min(95, base + random.random()*2))
        p2 = p1*0.75; p3 = p1*0.55
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1),"model":"heuristic"}
    try:
        probs = MODEL.predict_proba(X)[0]
        uplift = float(probs[1]) if len(probs)>1 else float(probs[0])
        p1 = uplift*100; p2 = p1*0.75; p3 = p1*0.55
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1),"model":"rf"}
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
        val = largest.get("value") if isinstance(largest, dict) else largest
        if isinstance(val, list):
            for ent in val:
                if isinstance(ent, dict):
                    amt = ent.get("uiAmount") or ent.get("amount")
                    holders.append({"address":ent.get("address"), "amount":amt})
        out["holders"]=holders
        if supply and holders:
            topn = sum((h.get("amount") or 0.0) for h in holders[:10])
            out["top10_pct"]=(topn/supply*100.0) if supply>0 else None
        else:
            out["top10_pct"]=None
    except Exception:
        out["supply"]=None; out["holders"]=[]; out["top10_pct"]=None
    cache_set(cache_key,out,ttl=60)
    return out

# --------------------
# Price helpers: reserves + CoinGecko
# --------------------
def get_token_account_ui_amount(token_account_pubkey):
    cache_key = f"tkbal:{token_account_pubkey}"
    cached = price_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        res = get_token_account_balance(token_account_pubkey)
        if not res or "value" not in res:
            return None
        val = res["value"]
        if val.get("uiAmountString") is not None:
            v = Decimal(val["uiAmountString"]); price_cache_set(cache_key, v); return v
        if val.get("uiAmount") is not None:
            v = Decimal(str(val["uiAmount"])); price_cache_set(cache_key, v); return v
        amt = Decimal(val.get("amount",0)); dec = int(val.get("decimals",0))
        v = amt / (Decimal(10) ** dec); price_cache_set(cache_key, v); return v
    except Exception:
        return None

def price_from_reserves(tokenA_acc, tokenB_acc, cache_ttl=15, min_liquidity_ui=Decimal("0.0001")):
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
    price_cache_set(cache_key, price, ttl=cache_ttl)
    return price

def coingecko_price_by_contract(contract_address, vs_currency="usd", cache_ttl=30):
    cache_key = f"cg:{contract_address}:{vs_currency}"
    cached = price_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        params = {"vs_currency": vs_currency, "contract_addresses": contract_address, "platform": "solana"}
        r = http_get(COINGECKO_MARKETS, params=params)
        if isinstance(r, list) and r:
            p = r[0].get("current_price")
            if p is not None:
                d = Decimal(str(p)); price_cache_set(cache_key,d,ttl=cache_ttl); return d
    except Exception:
        pass
    return None

# --------------------
# Pool parsers (Raydium & Orca) - best-effort heuristics
# --------------------
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def _b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    if n == 0:
        return "1"
    out=[]
    while n:
        n, r = divmod(n,58); out.append(ALPHABET[r])
    for ch in b:
        if ch == 0:
            out.append(ALPHABET[0])
        else:
            break
    return "".join(reversed(out))

def parse_raydium_pool_accounts(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data",[None])[0]
        if not data_b64:
            return None
        raw = base64.b64decode(data_b64)
        candidate_offsets = [72,88,104,120,136,160,200,232]
        for off in candidate_offsets:
            if off+64 <= len(raw):
                a_bytes = raw[off:off+32]; b_bytes = raw[off+32:off+64]
                a_pub = _b58encode(a_bytes); b_pub = _b58encode(b_bytes)
                try:
                    ba = get_token_account_balance(a_pub); bb = get_token_account_balance(b_pub)
                    if ba and bb and "value" in ba and "value" in bb:
                        return a_pub, b_pub
                except Exception:
                    continue
    except Exception:
        pass
    return None

def parse_orca_whirlpool(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data",[None])[0]
        if not data_b64:
            return None
        raw = base64.b64decode(data_b64)
        candidates=[]
        for i in range(0, max(1,len(raw)-32)):
            part = raw[i:i+32]
            if part.count(b'\x00') > 20: continue
            candidates.append(part)
            if len(candidates)>24: break
        tested=[]
        for b in candidates:
            pub = _b58encode(b)
            try:
                bal = get_token_account_balance(pub)
                if bal and "value" in bal:
                    tested.append(pub)
            except Exception:
                continue
        for i in range(len(tested)):
            for j in range(i+1,len(tested)):
                a = get_token_account_balance(tested[i]); b = get_token_account_balance(tested[j])
                if a and b and "value" in a and "value" in b:
                    return tested[i], tested[j]
    except Exception:
        pass
    return None

def discover_reserve_accounts_for_pool(pool_account_pubkey: str) -> Optional[Tuple[str,str]]:
    r = parse_raydium_pool_accounts(pool_account_pubkey)
    if r: return r
    o = parse_orca_whirlpool(pool_account_pubkey)
    if o: return o
    # generic fallback
    try:
        resp = get_account_info_base64(pool_account_pubkey)
        if not resp or "value" not in resp or resp["value"] is None:
            return None
        data_b64 = resp["value"].get("data",[None])[0]; raw = base64.b64decode(data_b64)
        candidates=[]
        for i in range(0, max(1,len(raw)-32)):
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
            for j in range(i+1,len(tested)):
                a = get_token_account_balance(tested[i]); b = get_token_account_balance(tested[j])
                if a and b and "value" in a and "value" in b:
                    return tested[i], tested[j]
    except Exception:
        pass
    return None

# --------------------
# Candidate discovery + approval flow
# --------------------
PENDING: Dict[str, Dict] = {}

def discover_candidates(seed_mints: List[str], top_k=8) -> List[Dict]:
    out=[]
    for m in seed_mints:
        try:
            metrics = fetch_token_metrics(m)
            probs = predict_with_model(m, metrics)
            out.append({"mint":m,"metrics":metrics,"probs":probs})
        except Exception:
            continue
    out_sorted = sorted(out, key=lambda x: x["probs"]["1m"], reverse=True)
    return out_sorted[:top_k]

def format_candidate_msg(cand, idx, total):
    m=cand["mint"]; p=cand["probs"]; top10=cand["metrics"].get("top10_pct")
    return f"[{idx+1}/{total}] Mint: {m}\nTop10%: {top10}\n1m:{p['1m']}% 2m:{p['2m']}% 3m:{p['3m']}%\nApprove?"

# --------------------
# Telegram handlers (async)
# --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pocket Option Bot\n/analyze <mint> [--price]\n/suggest <mints>\n/train")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    up = str(datetime.utcnow() - START).split('.')[0]
    await update.message.reply_text(f"Running\nUptime: {up}\nRPC: {RPC_URL}\nModel: {'loaded' if MODEL is not None else 'none'}")

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /analyze <TOKEN_MINT> [--price]"); return
    mint = args[0].strip(); want_price="--price" in args or "-p" in args
    await update.message.reply_text(f"Inspecting {mint} ...")
    metrics = fetch_token_metrics(mint)
    probs = predict_with_model(mint, metrics)
    lines=[f"Token: {mint}", f"Top10 pct: {metrics.get('top10_pct')}", f"Supply: {metrics.get('supply')}", f"1m:{probs['1m']}% 2m:{probs['2m']}% 3m:{probs['3m']}%"]
    if want_price:
        known_pools = {}  # add mint->pool mapping for deterministic results
        price=None
        for pool in known_pools.get(mint, []):
            discovered = discover_reserve_accounts_for_pool(pool)
            if discovered:
                price = price_from_reserves(discovered[0], discovered[1])
                if price: break
        if price is None:
            price = coingecko_price_by_contract(mint)
        lines.append(f"Price (USD or pool): {price if price is not None else 'not found'}")
    await update.message.reply_text("\n".join(lines))
    feats = featurize(mint, metrics); db_insert_interaction(mint, feats, label=0)

async def cmd_suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    seeds = [s.strip() for s in " ".join(args).replace(",", " ").split()] if args else []
    chat_id = str(update.effective_chat.id)
    await context.bot.send_message(chat_id=chat_id, text=f"Discovering among {len(seeds)} seeds. Running in background.")
    # perform discovery in a thread then send messages via context.application
    def bg_discover():
        cands = discover_candidates(seeds)
        if not cands:
            asyncio.run_coroutine_threadsafe(context.bot.send_message(chat_id=chat_id, text="No candidates found."), context.application.bot.loop)
            return
        PENDING[chat_id] = {"candidates": cands, "i": 0}
        text = format_candidate_msg(cands[0],0,len(cands))
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Yes", callback_data=f"approve|0"), InlineKeyboardButton("❌ No", callback_data=f"reject|0")],
            [InlineKeyboardButton("Next", callback_data=f"next|0"), InlineKeyboardButton("Stop", callback_data=f"stop|0")]
        ])
        asyncio.run_coroutine_threadsafe(context.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup), context.application.bot.loop)
    threading.Thread(target=bg_discover, daemon=True).start()

async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_chat.id)
    if ADMIN_CHAT and uid != str(ADMIN_CHAT):
        await update.message.reply_text("Admin only"); return
    await update.message.reply_text("Retraining model from DB...")
    # run blocking training in thread
    fut = asyncio.get_running_loop().run_in_executor(None, train_model_from_db)
    score = await fut
    if score is None:
        await update.message.reply_text("Not enough data or sklearn missing")
    else:
        await update.message.reply_text(f"Trained; test score {score:.3f}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()
    data = q.data or ""
    parts = data.split("|"); action = parts[0]; idx = int(parts[1]) if len(parts)>1 else 0
    chat = str(q.message.chat.id)
    state = PENDING.get(chat)
    if not state:
        await q.message.reply_text("No active session"); return
    candidates = state["candidates"]
    if idx<0 or idx>=len(candidates):
        await q.message.reply_text("Index out of range"); return
    if action=="approve":
        cand = candidates.pop(idx)
        db_insert_approval(chat, cand["mint"], "approved", cand["probs"])
        db_insert_published(cand["mint"], cand["probs"])
        await q.message.reply_text(f"Published: {cand['mint']} 1m:{cand['probs']['1m']}%")
        if ADMIN_CHAT:
            await context.bot.send_message(chat_id=ADMIN_CHAT, text=f"PUBLISHED {cand['mint']} -> {cand['probs']}")
        if not candidates:
            del PENDING[chat]; return
        state["i"] = min(idx, len(candidates)-1)
        text = format_candidate_msg(candidates[state["i"]], state["i"], len(candidates))
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Yes", callback_data=f"approve|{state['i']}"), InlineKeyboardButton("❌ No", callback_data=f"reject|{state['i']}")],
            [InlineKeyboardButton("Next", callback_data=f"next|{state['i']}"), InlineKeyboardButton("Stop", callback_data=f"stop|{state['i']}")]
        ])
        await context.bot.send_message(chat_id=chat, text=text, reply_markup=markup)
        return
    if action=="reject":
        cand = candidates.pop(idx)
        db_insert_approval(chat, cand["mint"], "rejected", cand["probs"])
        await q.message.reply_text("Rejected.")
        if not candidates:
            del PENDING[chat]; return
        state["i"] = min(idx, len(candidates)-1)
        text = format_candidate_msg(candidates[state["i"]], state["i"], len(candidates))
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Yes", callback_data=f"approve|{state['i']}"), InlineKeyboardButton("❌ No", callback_data=f"reject|{state['i']}")],
            [InlineKeyboardButton("Next", callback_data=f"next|{state['i']}"), InlineKeyboardButton("Stop", callback_data=f"stop|{state['i']}")]
        ])
        await context.bot.send_message(chat_id=chat, text=text, reply_markup=markup)
        return
    if action=="next":
        state["i"] = min(idx+1, len(candidates)-1)
        text = format_candidate_msg(candidates[state["i"]], state["i"], len(candidates))
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Yes", callback_data=f"approve|{state['i']}"), InlineKeyboardButton("❌ No", callback_data=f"reject|{state['i']}")],
            [InlineKeyboardButton("Next", callback_data=f"next|{state['i']}"), InlineKeyboardButton("Stop", callback_data=f"stop|{state['i']}")]
        ])
        await context.bot.send_message(chat_id=chat, text=text, reply_markup=markup)
        return
    if action=="stop":
        del PENDING[chat]; await q.message.reply_text("Stopped."); return

# --------------------
# Background tasks (async)
# --------------------
async def heartbeat_task(application):
    while True:
        if ADMIN_CHAT:
            try:
                await application.bot.send_message(chat_id=ADMIN_CHAT, text=f"Heartbeat {datetime.utcnow().isoformat()} RPC:{RPC_URL}")
            except Exception:
                pass
        await asyncio.sleep(1800)

async def on_startup(application):
    # schedule heartbeat
    application.create_task(heartbeat_task(application))
    # optionally run DB training in executor if requested
    if TRAIN_ON_START and SKLEARN_OK:
        application.create_task(asyncio.to_thread(train_model_from_db))

# --------------------
# Build and register handlers
# --------------------
START = datetime.utcnow()
application = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CommandHandler("status", cmd_status))
application.add_handler(CommandHandler("analyze", cmd_analyze))
application.add_handler(CommandHandler("suggest", cmd_suggest))
application.add_handler(CommandHandler("train", cmd_train))
application.add_handler(CallbackQueryHandler(handle_callback))

# Load model (non-blocking)
load_model()

# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    mode = "webhook" if WEBHOOK_URL else "polling"
    print(f"Starting Pocket Option bot. Mode: {mode}")
    if WEBHOOK_URL:
        application.run_webhook(listen="0.0.0.0", port=PORT, webhook_url=WEBHOOK_URL)
    else:
        # ensure no webhook is set (so polling works)
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook", timeout=5)
        except Exception:
            pass
        application.run_polling()
