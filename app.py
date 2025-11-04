#!/usr/bin/env python3
"""
app.py
Single-file Solana meme-coin bot (webhook-ready) with:
- Telegram webhook (Flask)
- Helius DAS + RPC fallback
- Jupiter quote integration
- Trainable ML predictor (RandomForest) with joblib persistence
- SQLite logging of interactions, approvals, published signals
- Daily re-evaluation and interactive Yes/No approval flow
"""

import os
import time
import threading
import requests
import json
import math
import random
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, CallbackQueryHandler
from typing import Dict, Any, List, Optional

# ML
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # model persistence

# --------------------
# Configuration (env)
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN required")

WEBHOOK_URL = os.getenv("WEBHOOK_URL")               # e.g. https://<your-choreo>.choreo.dev/webhook
ADMIN_CHAT = os.getenv("ADMIN_CHAT")                 # numeric id (optional, receives heartbeats)
PORT = int(os.getenv("PORT", "8080"))

RPC_URL = os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")         # optional (recommended)
HELIUS_BASE = "https://api.helius.xyz" if HELIUS_API_KEY else None
JUPITER_QUOTE_API = os.getenv("JUPITER_QUOTE_API", "https://quote-api.jup.ag/v6/quote")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1.5"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "120"))
RATE_LIMIT_SLEEP = float(os.getenv("RATE_LIMIT_SLEEP", "0.12"))

# Predictor & persistence
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/pocket_option_model.joblib")
TRAIN_ON_START = os.getenv("TRAIN_ON_START", "true").lower() in ("1","true","yes")
DAILY_REFRESH_HOUR_UTC = int(os.getenv("DAILY_REFRESH_HOUR_UTC", "6"))

# Candidate & thresholds
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "10"))
ALERT_CONF_THRESHOLD = float(os.getenv("ALERT_CONF_THRESHOLD", "70.0"))

# SQLite DB
DB_PATH = os.getenv("DB_PATH", "/tmp/pocket_option.db")

# --------------------
# Globals & cache
# --------------------
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)
app = Flask(__name__)

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

# --------------------
# SQLite for persistence
# --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        mint TEXT,
        features TEXT,
        label INTEGER,
        horizon TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        chat_id TEXT,
        mint TEXT,
        action TEXT,
        probs TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS published (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        mint TEXT,
        probs TEXT
    )""")
    conn.commit(); conn.close()
init_db()

def db_insert_interaction(mint, features: Dict, label:int, horizon:str):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO interactions (timestamp,mint,features,label,horizon) VALUES (?,?,?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(features), label, horizon))
    conn.commit(); conn.close()

def db_insert_approval(chat_id, mint, action, probs):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO approvals (timestamp,chat_id,mint,action,probs) VALUES (?,?,?,?,?)",
              (datetime.utcnow().isoformat(), str(chat_id), mint, action, json.dumps(probs)))
    conn.commit(); conn.close()

def db_insert_published(mint, probs):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO published (timestamp,mint,probs) VALUES (?,?,?)",
              (datetime.utcnow().isoformat(), mint, json.dumps(probs)))
    conn.commit(); conn.close()

def db_fetch_training_data():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT mint,features,label FROM interactions")
    rows = c.fetchall(); conn.close()
    X=[]; y=[]
    for mint, feats_json, label in rows:
        feats = json.loads(feats_json)
        X.append([feats.get("top10_pct_inv",0.0), feats.get("supply_inv",0.0), feats.get("activity",0.0), feats.get("noise",0.0)])
        y.append(int(label))
    if len(X)==0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

# --------------------
# HTTP helpers with retries
# --------------------
def backoff_sleep(attempt:int):
    time.sleep((BACKOFF_FACTOR ** attempt) * 0.4)

def http_get(url, params=None, headers=None):
    last=None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}")
                backoff_sleep(attempt); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e; backoff_sleep(attempt)
    raise last

def http_post(url, payload, headers=None):
    last=None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429,502,503,504):
                last = Exception(f"HTTP {r.status_code}")
                backoff_sleep(attempt); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; backoff_sleep(attempt)
    raise last

# --------------------
# Helius helpers & RPC fallback
# --------------------
def helius_get(path, params=None):
    if not HELIUS_API_KEY: raise RuntimeError("Helius not configured")
    url = f"{HELIUS_BASE}{path}"
    p = params or {}
    p["api-key"] = HELIUS_API_KEY
    return http_get(url, params=p)

def helius_token_holders(mint):
    try:
        res = helius_get("/v0/token/holders", {"mint": mint})
        return res.get("holders") if isinstance(res, dict) else None
    except Exception as e:
        print("Helius holders error:", e); return None

def rpc_post(method, params):
    payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
    try:
        return http_post(RPC_URL, payload)
    except Exception as e:
        print("RPC error:", e); return None

def get_token_largest_accounts_rpc(mint):
    time.sleep(RATE_LIMIT_SLEEP); return rpc_post("getTokenLargestAccounts", [mint])

def get_token_supply_rpc(mint):
    time.sleep(RATE_LIMIT_SLEEP); return rpc_post("getTokenSupply", [mint])

# --------------------
# Jupiter quote helper
# --------------------
def jupiter_quote(input_mint, output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1", amount=1_000_000):
    cache_key=f"jup:{input_mint}:{output_mint}:{amount}"
    cached = cache_get(cache_key)
    if cached: return cached
    try:
        params={"inputMint":input_mint,"outputMint":output_mint,"amount":amount}
        data = http_get(JUPITER_QUOTE_API, params=params)
        cache_set(cache_key,data,ttl=15); return data
    except Exception as e:
        print("jupiter error:", e); return None

# --------------------
# Metrics fetch & featurize
# --------------------
def fetch_token_metrics(mint):
    cache_key=f"metrics:{mint}"
    cached = cache_get(cache_key)
    if cached: return cached
    out={"mint":mint,"fetched_at":datetime.utcnow().isoformat()}
    holders=None
    if HELIUS_API_KEY:
        holders = helius_token_holders(mint)
        if holders is not None:
            out["holders"]=holders
            try:
                top10_share = sum(float(h.get("share",0.0)) for h in holders[:10])
                out["top10_pct"]= top10_share*100.0 if 0<=top10_share<=1 else None
            except Exception:
                out["top10_pct"]=None
    if holders is None:
        largest = get_token_largest_accounts_rpc(mint) or {}
        supply_res = get_token_supply_rpc(mint) or {}
        supply=None
        try:
            if isinstance(supply_res, dict) and "value" in supply_res:
                supply = float(supply_res["value"].get("uiAmount") or supply_res["value"].get("uiAmountString") or 0.0)
        except Exception:
            supply=None
        out["supply"]=supply
        try:
            val = largest.get("value") if isinstance(largest, dict) else largest
            parsed=[]
            if isinstance(val, list):
                for ent in val:
                    if isinstance(ent, dict):
                        parsed.append({"address":ent.get("address"), "amount": ent.get("uiAmount") or ent.get("amount")})
            out["holders"]=parsed
            if supply and parsed:
                total_top = sum((h.get("amount") or 0.0) for h in parsed[:10])
                out["top10_pct"]=(total_top/supply*100.0) if supply and supply!=0 else None
            else:
                out["top10_pct"]=None
        except Exception:
            out["holders"]=[]; out["top10_pct"]=None
    out["recent_activity_count"]=None
    cache_set(cache_key,out,ttl=60)
    return out

def featurize(mint, metrics):
    top10 = metrics.get("top10_pct")
    f={}
    f["top10_pct_inv"] = (100.0 - top10)/100.0 if isinstance(top10,(int,float)) else 0.0
    supply = metrics.get("supply") or 0.0
    try:
        s_log = math.log10(supply+1) if supply>0 else 0.0
        f["supply_inv"]= 1.0/(s_log+1.0)
    except Exception:
        f["supply_inv"]=0.0
    f["activity"] = (metrics.get("recent_activity_count") or 0)/100.0
    f["noise"]= random.random()*0.05
    return f

# --------------------
# Predictor: train/save/load/predict
# --------------------
MODEL: Optional[RandomForestClassifier]=None

def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print("Loaded model from", MODEL_PATH)
        except Exception as e:
            print("Model load failed:", e); MODEL=None

def train_model_from_db():
    global MODEL
    X,y = db_fetch_training_data()
    if X.size==0 or len(y)==0:
        print("Not enough training data in DB to train model"); return None
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    joblib.dump(clf, MODEL_PATH)
    MODEL=clf
    print(f"Trained model and saved to {MODEL_PATH}; test score {score:.3f}")
    return score

def predict_with_model(mint, metrics):
    feats = featurize(mint, metrics)
    X = np.array([[feats["top10_pct_inv"], feats["supply_inv"], feats["activity"], feats["noise"]]])
    if MODEL is None:
        # heuristic fallback
        base = 40*feats["top10_pct_inv"] + 25*feats["supply_inv"] + 10*feats["activity"] + 5*feats["noise"]
        p1 = max(0,min(95,base))
        p2 = max(0,min(95,p1*0.75)); p3 = max(0,min(95,p1*0.55))
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1),"model":"heuristic"}
    try:
        probs = MODEL.predict_proba(X)[0]
        prob_uplift = float(probs[1]) if len(probs)>1 else float(probs[0])
        p1 = prob_uplift*100; p2 = p1*0.75; p3 = p1*0.55
        return {"1m":round(p1,1),"2m":round(p2,1),"3m":round(p3,1),"model":"rf"}
    except Exception as e:
        print("Model predict failed:", e)
        return {"1m":0.0,"2m":0.0,"3m":0.0,"model":"error"}

# --------------------
# Candidate discovery & interactive approval
# --------------------
PENDING_APPROVALS: Dict[str, Dict] = {}

def discover_candidates(seed_mints: List[str]) -> List[Dict]:
    candidates=[]
    for mint in seed_mints:
        try:
            metrics = fetch_token_metrics(mint)
            probs = predict_with_model(mint, metrics)
            candidates.append({"mint":mint,"metrics":metrics,"probs":probs})
        except Exception as e:
            print("discover error", e)
    candidates_sorted = sorted(candidates, key=lambda x: x["probs"]["1m"], reverse=True)
    return candidates_sorted[:TOP_K_CANDIDATES]

def format_candidate_msg(cand, idx, total):
    m=cand["mint"]; p=cand["probs"]; top10=cand["metrics"].get("top10_pct")
    return (f"[{idx+1}/{total}] Mint: {m}\nTop10%: {top10}\nPredicted uplift → 1m:{p['1m']}% 2m:{p['2m']}% 3m:{p['3m']}%\nApprove?")

def start_approval_sequence(chat_id, seed_mints):
    candidates = discover_candidates(seed_mints)
    if not candidates:
        bot.send_message(chat_id=chat_id, text="No candidates discovered.")
        return
    PENDING_APPROVALS[str(chat_id)] = {"candidates":candidates,"index":0}
    send_candidate_prompt(chat_id)

def send_candidate_prompt(chat_id):
    key = str(chat_id); state = PENDING_APPROVALS.get(key)
    if not state:
        bot.send_message(chat_id=chat_id, text="No pending approvals."); return
    idx = state["index"]; candidates = state["candidates"]
    cand = candidates[idx]; total = len(candidates)
    text = format_candidate_msg(cand, idx, total)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Yes", callback_data=f"approve|{idx}"), InlineKeyboardButton("❌ No", callback_data=f"reject|{idx}")],
        [InlineKeyboardButton("Next", callback_data=f"next|{idx}"), InlineKeyboardButton("Stop", callback_data=f"stop|{idx}")]
    ])
    bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)

def handle_approval_callback(update, context):
    query = update.callback_query
    if not query: return
    data = (query.data or "")
    chat_id = str(query.message.chat.id)
    parts = data.split("|"); action = parts[0]; idx = int(parts[1]) if len(parts)>1 else 0
    state = PENDING_APPROVALS.get(chat_id)
    if not state:
        query.answer("No pending approvals."); return
    candidates = state["candidates"]
    if idx <0 or idx>=len(candidates):
        query.answer("Index out of range"); return
    if action=="approve":
        cand = candidates.pop(idx)
        db_insert_approval(chat_id, cand["mint"], "approved", cand["probs"])
        db_insert_published(cand["mint"], cand["probs"])
        msg = f"📡 PUBLISHED SIGNAL\nMint: {cand['mint']}\n1m {cand['probs']['1m']}% | 2m {cand['probs']['2m']}% | 3m {cand['probs']['3m']}%"
        if ADMIN_CHAT: bot.send_message(chat_id=ADMIN_CHAT, text=msg)
        query.message.reply_text("✅ Approved and published.")
        if not candidates:
            del PENDING_APPROVALS[chat_id]; return
        state["index"]=min(idx, len(candidates)-1); send_candidate_prompt(chat_id); query.answer("Approved"); return
    if action=="reject":
        cand = candidates.pop(idx); db_insert_approval(chat_id, cand["mint"], "rejected", cand["probs"])
        query.message.reply_text("❌ Rejected.")
        if not candidates:
            del PENDING_APPROVALS[chat_id]; return
        state["index"]=min(idx, len(candidates)-1); send_candidate_prompt(chat_id); query.answer("Rejected"); return
    if action=="next":
        state["index"]=min(idx+1, len(candidates)-1); send_candidate_prompt(chat_id); query.answer("Next"); return
    if action=="stop":
        del PENDING_APPROVALS[chat_id]; query.message.reply_text("Stopped approval session."); query.answer("Stopped"); return

# --------------------
# Telegram handlers
# --------------------
def cmd_start(update, context):
    update.message.reply_text("🤖 Pocket Option AI — /suggest <mints> ; /analyze <mint> [--price] ; /train to retrain model")

def cmd_status(update, context):
    uptime = str(datetime.utcnow() - START_TIME).split('.')[0]
    update.message.reply_text(f"✅ Running\nUptime: {uptime}\nHelius: {'on' if HELIUS_API_KEY else 'off'}\nModel: {'loaded' if MODEL is not None else 'none'}")

def cmd_analyze(update, context):
    args = context.args or []
    if not args:
        update.message.reply_text("Usage: /analyze <TOKEN_MINT> [--price]"); return
    mint = args[0].strip(); want_price="--price" in args or "-p" in args
    update.message.reply_text(f"🔎 Inspecting {mint} ...")
    metrics = fetch_token_metrics(mint); probs = predict_with_model(mint, metrics)
    lines=[f"Token: {mint}", f"Top10 pct: {metrics.get('top10_pct')}", f"Supply: {metrics.get('supply')}", f"Predicted uplift → 1m: {probs['1m']}% | 2m: {probs['2m']}% | 3m: {probs['3m']}%"]
    if want_price:
        j = jupiter_quote(mint)
        lines.append(f"Jupiter route: {'ok' if j and j.get('data') else 'no route'}")
    update.message.reply_text("\n".join(lines))

def cmd_suggest(update, context):
    args = context.args or []
    if args:
        raw=" ".join(args); seeds=[s.strip() for s in raw.replace(","," ").split() if s.strip()]
    else:
        seeds = [ # default seed list (replace with real token mints)
            "So11111111111111111111111111111111111111112"
        ]
    chat_id=str(update.message.chat.id)
    update.message.reply_text(f"🔎 Discovering among {len(seeds)} seeds. This may take a moment.")
    threading.Thread(target=start_approval_sequence, args=(chat_id, seeds), daemon=True).start()

def cmd_train(update, context):
    cid = str(update.message.chat.id)
    if ADMIN_CHAT and cid != str(ADMIN_CHAT):
        update.message.reply_text("Only admin can retrain."); return
    update.message.reply_text("🔁 Training model from DB...")
    try:
        score = train_model_from_db()
        if score is not None:
            update.message.reply_text(f"Trained model; test score {score:.3f}")
        else:
            update.message.reply_text("Not enough data to train.")
    except Exception as e:
        update.message.reply_text(f"Train failed: {e}")

dispatcher.add_handler(CommandHandler("start", cmd_start))
dispatcher.add_handler(CommandHandler("status", cmd_status))
dispatcher.add_handler(CommandHandler("analyze", cmd_analyze))
dispatcher.add_handler(CommandHandler("suggest", cmd_suggest))
dispatcher.add_handler(CommandHandler("train", cmd_train))
dispatcher.add_handler(CallbackQueryHandler(handle_approval_callback))

# --------------------
# Heartbeat & daily refresh
# --------------------
def heartbeat_loop():
    while True:
        if ADMIN_CHAT:
            try: bot.send_message(chat_id=ADMIN_CHAT, text=f"💓 Heartbeat {datetime.utcnow().isoformat()} Hel:{'on' if HELIUS_API_KEY else 'off'}")
            except Exception as e: print("heartbeat err", e)
        time.sleep(1800)

def daily_refresh_loop():
    while True:
        now = datetime.utcnow()
        target = now.replace(hour=DAILY_REFRESH_HOUR_UTC, minute=0, second=0, microsecond=0)
        if target <= now: target += timedelta(days=1)
        sleep_seconds = (target - now).total_seconds()
        time.sleep(sleep_seconds + 2)
        try:
            conn = sqlite3.connect(DB_PATH); c = conn.cursor()
            c.execute("SELECT mint FROM published ORDER BY id DESC LIMIT 20")
            rows = c.fetchall(); conn.close()
            mints = [r[0] for r in rows]
            for m in mints:
                metrics = fetch_token_metrics(m); probs = predict_with_model(m, metrics)
                if ADMIN_CHAT:
                    bot.send_message(chat_id=ADMIN_CHAT, text=f"Refresh {m}: 1m {probs['1m']}% 2m {probs['2m']}%")
        except Exception as e:
            print("daily refresh failed:", e)

# --------------------
# Webhook set & startup
# --------------------
START_TIME = datetime.utcnow()
MODEL=None

def load_model_on_start():
    global MODEL
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH); print("Loaded model", MODEL_PATH)
    except Exception as e:
        print("load model err", e)

def train_if_requested():
    if TRAIN_ON_START:
        try:
            train_model_from_db()
            load_model_on_start()
        except Exception as e:
            print("train on start failed:", e)

def set_telegram_webhook():
    if not WEBHOOK_URL:
        print("WEBHOOK_URL not set; skip setWebhook"); return
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook", json={"url": WEBHOOK_URL}, timeout=REQUEST_TIMEOUT)
        print("setWebhook:", resp.status_code, resp.text)
    except Exception as e:
        print("setWebhook failed:", e)

if __name__ == "__main__":
    print("Starting Pocket Option single-file bot (Helius + Jupiter + ML + SQLite)")
    set_telegram_webhook()
    load_model_on_start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=daily_refresh_loop, daemon=True).start()
    if TRAIN_ON_START:
        threading.Thread(target=train_if_requested, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
