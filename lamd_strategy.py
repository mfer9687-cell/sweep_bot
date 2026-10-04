import numpy as np
import pandas as pd
from vwap_strategy import fetch, liquidity_levels, stats, PAIRS
from ichi_strategy import atr

SL_BUF = 0.3          # بافر حد ضرر (ATR)
MAX_RISK_ATR = 3.0    # ستاپ با حد ضرر دور رد می‌شود
COST_ATR = 0.05       # هزینه فرضی اسپرد (ATR)
FVG_WIN = 12          # تا چند کندل بعد از شکار، دنبال MSS و FVG بگرد
FILL_WIN = 12         # تا چند کندل بعد از FVG، منتظر برگشت به ۵۰٪ بمان
MAX_PER_DAY = 2
RR_FIX = {"2R": 2.0, "1.5R": 1.5}
MAX_BARS = {"5m": 96, "15m": 48}


def london_levels(df):
    day = df.index.normalize()
    m = df.index.hour * 60 + df.index.minute
    lon = df[(m >= 390) & (m < 750)]
    ld = lon.index.normalize()
    lh = lon.High.groupby(ld).max().reindex(day).values
    ll = lon.Low.groupby(ld).min().reindex(day).values
    after = m >= 750
    return np.where(after, lh, np.nan), np.where(after, ll, np.nan)


def bias_4h(df):
    h = df.resample("4h", label="left", closed="left").agg({"Close": "last"}).dropna()
    e = h.Close.ewm(span=20, adjust=False).mean()
    b = np.sign(h.Close - e)
    return b.shift(1).reindex(df.index, method="ffill").fillna(0).values


def run_side(O, H, L, C, A, sweep_levels, tp_levels, ok, tp_mode, tf):
    n = len(C)
    sw = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        for lv in sweep_levels:
            sw |= (L < lv - 0.05 * A) & (C > lv)
    out = []
    maxb = MAX_BARS[tf]
    for j in range(60, n - 20):
        if not sw[j] or not ok[j] or np.isnan(A[j]):
            continue
        mss = H[j - 6:j + 1].max()
        found = None
        for m in range(j + 2, min(j + 1 + FVG_WIN, n)):
            gap = L[m] - H[m - 2]
            if gap < 0.1 * A[m] or gap > 1.5 * A[m]:
                continue
            if max(C[m - 1], C[m]) <= mss:
                continue
            if max(C[m - 1] - O[m - 1], C[m] - O[m]) < 0.8 * A[m]:
                continue
            found = m
            break
        if found is None:
            continue
        m = found
        ce = (H[m - 2] + L[m]) / 2
        sl = L[j:m + 1].min() - SL_BUF * A[m]
        risk = ce - sl
        if risk <= 0 or risk > MAX_RISK_ATR * A[m]:
            continue
        if tp_mode == "ext":
            cands = sorted(x[m] for x in tp_levels if not np.isnan(x[m]) and x[m] > ce)
            tp = next((x for x in cands if (x - ce) / risk >= 3.0), None)
            if tp is None:
                continue
        else:
            tp = ce + RR_FIX[tp_mode] * risk
        t = None
        for u in range(m + 1, min(m + 1 + FILL_WIN, n)):
            if L[u] <= sl:
                break
            if L[u] <= ce:
                t = u
                break
        if t is None:
            continue
        R, e = None, t
        for e in range(t + 1, min(t + 1 + maxb, n)):
            if L[e] <= sl:          # اگر هر دو در یک کندل بود، حد ضرر فرض می‌شود
                R = -1.0
                break
            if H[e] >= tp:
                R = (tp - ce) / risk
                break
        if R is None:
            e = min(t + maxb, n - 1)
            R = (C[e] - ce) / risk
        R -= COST_ATR * A[m] / risk
        out.append((t, e, R, (tp - ce) / risk))
    return out


def backtest(df, tf, use_bias, tp_mode):
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    A = atr(df).values
    idx = df.index
    mins = idx.hour * 60 + idx.minute
    sess = (mins >= 390) & (mins < 1200)
    ah, al, pdh, pdl = (x.values for x in liquidity_levels(df))
    lh, ll = london_levels(df)
    b = bias_4h(df)
    ok_long = sess & ((b == 1) if use_bias else True)
    ok_short = sess & ((b == -1) if use_bias else True)
    longs = run_side(O, H, L, C, A, [al, pdl, ll], [ah, pdh, lh], ok_long, tp_mode, tf)
    shorts = run_side(-O, -L, -H, -C, A, [-ah, -pdh, -lh], [-al, -pdl, -ll], ok_short, tp_mode, tf)
    mid = idx[len(idx) // 2]
    trades, last_exit, per_day = [], -1, {}
    for t, e, R, rr in sorted(longs + shorts):
        day = idx[t].date()
        if t <= last_exit or per_day.get(day, 0) >= MAX_PER_DAY:
            continue
        trades.append((idx[t], R, 0 if idx[t] < mid else 1, rr))
        per_day[day] = per_day.get(day, 0) + 1
        last_exit = e
    return trades
