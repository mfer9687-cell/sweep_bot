import os
import json
import pandas as pd
import requests

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
         "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
         "XAUUSD", "NAS100", "US30"]
SPECIAL = {"XAUUSD": "XAU_USD", "NAS100": "NAS100_USD", "US30": "US30_USD"}

MIN_DEPTH = 0.1   # حداقل عمق شکار (ATR)
MAX_DEPTH = 1.0   # حداکثر عمق شکار (ATR)
SL_BUF = 0.3      # بافر حد ضرر (ATR)
MIN_RR = 3.0      # حداقل ریسک به ریوارد
LOOKBACK = 3      # تعداد کندل اخیر برای بررسی (تاخیر کرون گیت‌هاب)
STATE = "state.json"

# ساعت‌ها به UTC (تهران = UTC + 3:30)
ASIA_END_MIN = 390                        # 06:30 UTC = 10:00 تهران
KILL_ZONES = [(390, 570), (750, 870)]     # 10-13 و 16-18 تهران

OANDA_TOKEN = os.environ.get("OANDA_TOKEN")
OANDA_HOST = ("https://api-fxtrade.oanda.com"
              if os.environ.get("OANDA_ENV") == "live"
              else "https://api-fxpractice.oanda.com")
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT = os.environ.get("TELEGRAM_CHAT_ID")


def load_oanda(pair):
    inst = SPECIAL.get(pair, pair[:3] + "_" + pair[3:])
    r = requests.get(
        f"{OANDA_HOST}/v3/instruments/{inst}/candles",
        headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
        params={"granularity": "M5", "count": 3000, "price": "M"},
        timeout=30,
    )
    r.raise_for_status()
    rows = [(pd.to_datetime(c["time"], utc=True),
             float(c["mid"]["o"]), float(c["mid"]["h"]),
             float(c["mid"]["l"]), float(c["mid"]["c"]))
            for c in r.json()["candles"] if c["complete"]]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close"]).set_index("t")
    return df


YAHOO_SPECIAL = {"XAUUSD": "GC=F", "NAS100": "NQ=F", "US30": "YM=F"}
AGE = {}          # سن آخرین کندل بسته‌شده (دقیقه)
MAX_AGE = 8.0     # داده مسن‌تر از این رد می‌شود


def load_yahoo(pair):
    import yfinance as yf
    df = yf.Ticker(YAHOO_SPECIAL.get(pair, pair + "=X")).history(period="10d", interval="5m")
    if df.empty:
        return None
    df.index = df.index.tz_convert("UTC")
    df = df[["Open", "High", "Low", "Close"]].dropna()
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=5) <= now]


def load(pair):
    df = load_oanda(pair) if OANDA_TOKEN else load_yahoo(pair)
    if df is not None and len(df):
        end = df.index[-1] + pd.Timedelta(minutes=5)
        AGE[pair] = (pd.Timestamp.now(tz="UTC") - end).total_seconds() / 60
    return df


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low,
                    (df.High - pc).abs(),
                    (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]


def cloud_trend(df):
    h = df.resample("1h").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(h) < 80:
        return 0
    ten = (h.High.rolling(9).max() + h.Low.rolling(9).min()) / 2
    kij = (h.High.rolling(26).max() + h.Low.rolling(26).min()) / 2
    a = ((ten + kij) / 2).shift(26).iloc[-1]
    b = ((h.High.rolling(52).max() + h.Low.rolling(52).min()) / 2).shift(26).iloc[-1]
    if pd.isna(a) or pd.isna(b):
        return 0
    p = h.Close.iloc[-1]
    if p > max(a, b):
        return 1
    if p < min(a, b):
        return -1
    return 0


def in_kill_zone(ts):
    m = ts.hour * 60 + ts.minute
    return any(s <= m < e for s, e in KILL_ZONES)


def find_signal(pair, df):
    a = atr(df)
    if pd.isna(a) or a == 0:
        return None
    trend = cloud_trend(df)
    if trend == 0:
        return None
    last_close = df.Close.iloc[-1]

    for i in range(1, LOOKBACK + 1):
        bar = df.iloc[-i]
        if not in_kill_zone(bar.name):
            continue
        day = bar.name.normalize()
        asia = df[(df.index >= day) & (df.index < day + pd.Timedelta(minutes=ASIA_END_MIN))]
        before = df[df.index < day]
        if len(asia) < 60 or before.empty:
            continue
        prev = before[before.index.normalize() == before.index[-1].normalize()]
        ah, al = asia.High.max(), asia.Low.min()
        pdh, pdl = prev.High.max(), prev.Low.min()

        # فروش: شکار سقف و بسته‌شدن زیر آن
        if trend == -1:
            for name, lvl in (("سقف آسیا", ah), ("سقف روز قبل", pdh)):
                depth = bar.High - lvl
                if MIN_DEPTH * a <= depth <= MAX_DEPTH * a and bar.Close < lvl:
                    entry = bar.Close
                    sl = bar.High + SL_BUF * a
                    risk = sl - entry
                    for tp in sorted([x for x in (al, pdl) if x < entry], reverse=True):
                        if (entry - tp) / risk >= MIN_RR and entry - 0.5 * risk < last_close < sl:
                            return ("فروش", name, bar.name, entry, sl, tp, (entry - tp) / risk)

        # خرید: شکار کف و بسته‌شدن بالای آن
        if trend == 1:
            for name, lvl in (("کف آسیا", al), ("کف روز قبل", pdl)):
                depth = lvl - bar.Low
                if MIN_DEPTH * a <= depth <= MAX_DEPTH * a and bar.Close > lvl:
                    entry = bar.Close
                    sl = bar.Low - SL_BUF * a
                    risk = entry - sl
                    for tp in sorted([x for x in (ah, pdh) if x > entry]):
                        if (tp - entry) / risk >= MIN_RR and sl < last_close < entry + 0.5 * risk:
                            return ("خرید", name, bar.name, entry, sl, tp, (tp - entry) / risk)
    return None


def send(text):
    if not TOKEN or not CHAT:
        print(text)
        return
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT, "text": text}, timeout=20)


def main():
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    state = {k: v for k, v in state.items() if v >= today}

    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send("ربات وصل است")

    for pair in PAIRS:
        key = f"{pair}-{today}"
        if key in state:
            continue
        try:
            df = load(pair)
            if df is None or len(df) < 200:
                continue
            if AGE.get(pair, 0) > MAX_AGE:
                print(pair, "stale data, age", round(AGE[pair], 1))
                continue
            sig = find_signal(pair, df)
        except Exception as e:
            print(pair, "error:", e)
            continue
        if not sig:
            continue
        side, lvl, t, entry, sl, tp, rr = sig
        d = 2 if pair == "XAUUSD" else 1 if pair in ("NAS100", "US30") else 3 if "JPY" in pair else 5
        send(
            f"سیگنال شکار نقدینگی\n"
            f"{side}\n"
            f"سطح شکارشده: {lvl}\n"
            f"ورود: {entry:.{d}f}\n"
            f"حد ضرر: {sl:.{d}f}\n"
            f"حد سود: {tp:.{d}f}\n"
            f"ریسک به ریوارد: {rr:.1f}\n"
            f"زمان کندل (UTC): {t.strftime('%H:%M')}\n"
            f"{pair}"
        )
        state[key] = today

    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch" and AGE:
        lines = ["سن آخرین کندل (دقیقه)"] + [f"{p}: {v:.1f}" for p, v in AGE.items()]
        send("\n".join(lines))
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
