import numpy as np
import pandas as pd
import requests
from sweep_bot import (PAIRS, SPECIAL, YAHOO_SPECIAL,
                       OANDA_TOKEN, OANDA_HOST)

RR = 1.5              # نسبت ریسک به ریوارد (بعد از بک‌تست تنظیم کن)
SL_MIN = 1.0          # حداقل فاصله حد ضرر (ATR)
SL_MAX = 2.5          # حداکثر فاصله حد ضرر (ATR)
COST_ATR = 0.05       # هزینه فرضی اسپرد در بک‌تست (ATR)
MAX_BARS = 40         # حداکثر ماندن در معامله (کندل)
MAX_PER_DAY = 2       # حداکثر سیگنال در روز برای هر نماد
SESSION_START = 390   # 06:30 UTC = 10:00 تهران
SESSION_END = 1260    # 21:00 UTC

STEP = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
GRAN = {"15m": "M15", "1h": "H1"}


def fetch(pair, interval, period):
    if OANDA_TOKEN:
        inst = SPECIAL.get(pair, pair[:3] + "_" + pair[3:])
        r = requests.get(
            f"{OANDA_HOST}/v3/instruments/{inst}/candles",
            headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
            params={"granularity": GRAN[interval], "count": 5000, "price": "M"},
            timeout=30)
        r.raise_for_status()
        rows = [(pd.to_datetime(c["time"], utc=True), float(c["mid"]["o"]),
                 float(c["mid"]["h"]), float(c["mid"]["l"]), float(c["mid"]["c"]))
                for c in r.json()["candles"] if c["complete"]]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close"]).set_index("t")
    else:
        import yfinance as yf
        df = yf.Ticker(YAHOO_SPECIAL.get(pair, pair + "=X")).history(period=period, interval=interval)
        if df.empty:
            return None
        df.index = df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close"]].dropna()
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + STEP[interval] <= now]


def ichimoku(df):
    h, l = df.High, df.Low
    t = (h.rolling(9).max() + l.rolling(9).min()) / 2
    k = (h.rolling(26).max() + l.rolling(26).min()) / 2
    a = ((t + k) / 2).shift(26)
    b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return t, k, a, b


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def htf_trend(df, rule):
    h = df.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    t, k, a, b = ichimoku(h)
    top = pd.concat([a, b], axis=1).max(axis=1)
    bot = pd.concat([a, b], axis=1).min(axis=1)
    tr = pd.Series(0, index=h.index)
    tr[h.Close > top] = 1
    tr[h.Close < bot] = -1
    tr[top.isna()] = 0
    # فقط کندل‌های بسته‌شده تایم بالاتر استفاده شود (بدون نگاه به آینده)
    return tr.shift(1).reindex(df.index, method="ffill").fillna(0)


def signals(df, tf):
    t, k, a, b = ichimoku(df)
    top = pd.concat([a, b], axis=1).max(axis=1)
    bot = pd.concat([a, b], axis=1).min(axis=1)
    at = atr(df)
    r = rsi(df.Close)
    c = df.Close
    cross_up = (t > k) & (t.shift(1) <= k.shift(1))
    cross_dn = (t < k) & (t.shift(1) >= k.shift(1))
    rec_up = (cross_up.rolling(3).max() == 1) & (t > k)
    rec_dn = (cross_dn.rolling(3).max() == 1) & (t < k)
    htf = htf_trend(df, "1h" if tf == "15m" else "4h")
    thick = (top - bot) > 0.3 * at
    m = df.index.hour * 60 + df.index.minute
    sess = (m >= SESSION_START) & (m < SESSION_END)
    buy = rec_up & (c > top) & (c > c.shift(26)) & (r > 50) & (r < 72) & (htf == 1) & thick & sess
    sell = rec_dn & (c < bot) & (c < c.shift(26)) & (r < 50) & (r > 28) & (htf == -1) & thick & sess
    side = pd.Series(0, index=df.index)
    side[buy] = 1
    side[sell] = -1
    return side, at, k


def levels(entry, side, atr_v, kijun_v, rr=None):
    rr = RR if rr is None else rr
    d = abs(entry - kijun_v)
    d = min(max(d, SL_MIN * atr_v), SL_MAX * atr_v)
    return entry - side * d, entry + side * d * rr, d


def backtest(df, tf, rr):
    side, at, k = signals(df, tf)
    H, L, C = df.High.values, df.Low.values, df.Close.values
    S, A, K = side.values, at.values, k.values
    idx = df.index
    mid = idx[len(idx) // 2]
    trades, per_day = [], {}
    n, i = len(df), 100
    while i < n - 1:
        s = S[i]
        if s == 0 or np.isnan(A[i]) or np.isnan(K[i]):
            i += 1
            continue
        day = idx[i].date()
        if per_day.get(day, 0) >= MAX_PER_DAY:
            i += 1
            continue
        entry = C[i]
        sl, tp, d = levels(entry, s, A[i], K[i], rr)
        R, j = None, i
        for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
            hit_sl = L[j] <= sl if s == 1 else H[j] >= sl
            hit_tp = H[j] >= tp if s == 1 else L[j] <= tp
            if hit_sl:               # اگر هر دو در یک کندل بود، حد ضرر فرض می‌شود
                R = -1.0
                break
            if hit_tp:
                R = rr
                break
        if R is None:
            R = (C[j] - entry) * s / d
        R -= COST_ATR * A[i] / d
        trades.append((idx[i], R, 0 if idx[i] < mid else 1))
        per_day[day] = per_day.get(day, 0) + 1
        i = j + 1
    return trades


def stats(rs):
    rs = np.array(rs)
    if len(rs) == 0:
        return None
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    return len(rs), 100 * len(wins) / len(rs), rs.mean(), pf
