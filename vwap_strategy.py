import numpy as np
import pandas as pd
import requests
from sweep_bot import (PAIRS, SPECIAL, YAHOO_SPECIAL,
                       OANDA_TOKEN, OANDA_HOST)
from ichi_strategy import atr, rsi

MIN_RR = 1.5          # حداقل ریسک به ریوارد برای گرفتن ستاپ
MAX_RISK_ATR = 3.0    # ستاپ با حد ضرر خیلی دور رد می‌شود
SL_BUF = 0.3          # بافر حد ضرر (ATR)
COST_ATR = 0.05       # هزینه فرضی اسپرد (ATR)
MAX_PER_DAY = 2
LOOK = 6              # تعداد کندل برای شکار/کشیدگی/RSI
SESSION_START, SESSION_END = 390, 1200   # 06:30 تا 20:00 UTC

STEP = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15)}
GRAN = {"5m": "M5", "15m": "M15"}
MAX_BARS = {"5m": 60, "15m": 40}


def fetch(pair, interval, period):
    if OANDA_TOKEN:
        inst = SPECIAL.get(pair, pair[:3] + "_" + pair[3:])
        r = requests.get(
            f"{OANDA_HOST}/v3/instruments/{inst}/candles",
            headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
            params={"granularity": GRAN[interval], "count": 5000, "price": "M"},
            timeout=30)
        r.raise_for_status()
        rows = [(pd.to_datetime(c["time"], utc=True), float(c["mid"]["o"]), float(c["mid"]["h"]),
                 float(c["mid"]["l"]), float(c["mid"]["c"]), float(c["volume"]))
                for c in r.json()["candles"] if c["complete"]]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume"]).set_index("t")
    else:
        import yfinance as yf
        df = yf.Ticker(YAHOO_SPECIAL.get(pair, pair + "=X")).history(period=period, interval=interval)
        if df.empty:
            return None
        df.index = df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + STEP[interval] <= now]


def vwap_bands(df):
    tp = (df.High + df.Low + df.Close) / 3
    vol = df.Volume.clip(lower=1) if df.Volume.sum() > 0 else pd.Series(1.0, index=df.index)
    day = df.index.normalize()
    pv = (tp * vol).groupby(day).cumsum()
    v = vol.groupby(day).cumsum()
    p2 = (tp * tp * vol).groupby(day).cumsum()
    vw = pv / v
    sd = np.sqrt((p2 / v - vw ** 2).clip(lower=0))
    return vw, sd


def liquidity_levels(df):
    day = df.index.normalize()
    m = df.index.hour * 60 + df.index.minute
    asia = df[m < 390]
    ad = asia.index.normalize()
    ah = asia.High.groupby(ad).max().reindex(day).values
    al = asia.Low.groupby(ad).min().reindex(day).values
    after = m >= 390                         # سطح آسیا فقط بعد از پایان آسیا معتبر است
    ah = np.where(after, ah, np.nan)
    al = np.where(after, al, np.nan)
    counts = df.groupby(day).size()
    good = counts[counts >= 0.5 * counts.median()].index
    dh = df.High.groupby(day).max()[good]
    dl = df.Low.groupby(day).min()[good]
    pdh = dh.shift(1).reindex(day).values
    pdl = dl.shift(1).reindex(day).values
    return (pd.Series(ah, index=df.index), pd.Series(al, index=df.index),
            pd.Series(pdh, index=df.index), pd.Series(pdl, index=df.index))


def signals(df, k, use_liq):
    vw, sd = vwap_bands(df)
    at = atr(df)
    r = rsi(df.Close)
    o, c, lo, hi = df.Open, df.Close, df.Low, df.High
    m = df.index.hour * 60 + df.index.minute
    sess = (m >= SESSION_START) & (m < SESSION_END)
    roll = lambda s: s.astype(float).rolling(LOOK).max() == 1
    ext_dn = roll(lo < vw - k * sd)
    ext_up = roll(hi > vw + k * sd)
    rsi_lo = r.rolling(LOOK).min() < 30
    rsi_hi = r.rolling(LOOK).max() > 70
    turn_up = (c > o) & (r > r.shift(1))
    turn_dn = (c < o) & (r < r.shift(1))
    if use_liq:
        ah, al, pdh, pdl = liquidity_levels(df)
        b = 0.05 * at
        sw_dn = ((lo < al - b) & (c > al)) | ((lo < pdl - b) & (c > pdl))
        sw_up = ((hi > ah + b) & (c < ah)) | ((hi > pdh + b) & (c < pdh))
        liq_b, liq_s = roll(sw_dn), roll(sw_up)
    else:
        liq_b = liq_s = pd.Series(True, index=df.index)
    buy = ext_dn & rsi_lo & turn_up & liq_b & sess & (c < vw)
    sell = ext_up & rsi_hi & turn_dn & liq_s & sess & (c > vw)
    side = pd.Series(0, index=df.index)
    side[buy] = 1
    side[sell] = -1
    sl_buy = lo.rolling(LOOK).min() - SL_BUF * at
    sl_sell = hi.rolling(LOOK).max() + SL_BUF * at
    sl = pd.Series(np.where(side == 1, sl_buy, sl_sell), index=df.index)
    return side, at, vw, sl


def setup(entry, s, sl, tp, at_v):
    """ستاپ فقط وقتی معتبر است که ریسک و ریوارد مناسب باشد."""
    risk = (entry - sl) * s
    reward = (tp - entry) * s
    if not (risk > 0 and reward > 0) or risk > MAX_RISK_ATR * at_v:
        return None
    rr = reward / risk
    return rr if rr >= MIN_RR else None


def backtest(df, tf, k, use_liq):
    side, at, vw, slv = signals(df, k, use_liq)
    H, L, C = df.High.values, df.Low.values, df.Close.values
    S, A, V, SL = side.values, at.values, vw.values, slv.values
    idx = df.index
    mid = idx[len(idx) // 2]
    trades, per_day = [], {}
    n, i = len(df), 100
    while i < n - 1:
        s = S[i]
        if s == 0 or np.isnan(A[i]) or np.isnan(SL[i]) or np.isnan(V[i]):
            i += 1
            continue
        day = idx[i].date()
        if per_day.get(day, 0) >= MAX_PER_DAY:
            i += 1
            continue
        entry, sl, tp = C[i], SL[i], V[i]
        rr = setup(entry, s, sl, tp, A[i])
        if rr is None:
            i += 1
            continue
        risk = abs(entry - sl)
        R, j = None, i
        for j in range(i + 1, min(i + 1 + MAX_BARS[tf], n)):
            hit_sl = L[j] <= sl if s == 1 else H[j] >= sl
            hit_tp = H[j] >= tp if s == 1 else L[j] <= tp
            if hit_sl:
                R = -1.0
                break
            if hit_tp:
                R = rr
                break
        if R is None:
            R = (C[j] - entry) * s / risk
        R -= COST_ATR * A[i] / risk
        trades.append((idx[i], R, 0 if idx[i] < mid else 1, rr))
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
