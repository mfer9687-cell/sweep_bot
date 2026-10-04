import os
import json
import pandas as pd
import requests
from sweep_bot import (PAIRS, SPECIAL, YAHOO_SPECIAL,
                       OANDA_TOKEN, OANDA_HOST, send)

STATE = "sessions_state.json"
WINDOW_MIN = 90        # اگر اجرا دیر شد، تا ۹۰ دقیقه بعد از مرز هنوز می‌فرستد
MIN_DAY_BARS = 100     # روزی که کمتر از این کندل دارد (آخر هفته) روز کاری حساب نمی‌شود

# ساعت‌ها به UTC (تهران = UTC + 3:30)
# آسیا 00:00-06:30 (تهران 03:30-10:00)
# لندن 06:30-12:30 (تهران 10:00-16:00)
# نیویورک 12:30-24:00 (تهران 16:00-03:30)
# پایان روز = ساعت 00:00 UTC (تهران 03:30)
SESSIONS = [("آسیا", 0, 390), ("لندن", 390, 750), ("نیویورک", 750, 1440)]
BOUNDS = [0, 390, 750]
INFO = {0: (2, 0), 390: (0, 1), 750: (1, 2)}   # مرز: (سشن تمام‌شده، سشن شروع‌شده)


def dec(pair):
    return 2 if pair == "XAUUSD" else 1 if pair in ("NAS100", "US30") else 3 if "JPY" in pair else 5


def load(pair):
    if OANDA_TOKEN:
        inst = SPECIAL.get(pair, pair[:3] + "_" + pair[3:])
        r = requests.get(
            f"{OANDA_HOST}/v3/instruments/{inst}/candles",
            headers={"Authorization": f"Bearer {OANDA_TOKEN}"},
            params={"granularity": "M5", "count": 5000, "price": "M"},
            timeout=30)
        r.raise_for_status()
        rows = [(pd.to_datetime(c["time"], utc=True),
                 float(c["mid"]["h"]), float(c["mid"]["l"]))
                for c in r.json()["candles"] if c["complete"]]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["t", "High", "Low"]).set_index("t")
    else:
        import yfinance as yf
        df = yf.Ticker(YAHOO_SPECIAL.get(pair, pair + "=X")).history(period="30d", interval="5m")
        if df.empty:
            return None
        df.index = df.index.tz_convert("UTC")
        df = df[["High", "Low"]].dropna()
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=5) <= now]


def hl(df, start, end, min_bars=1):
    x = df[(df.index >= start) & (df.index < end)]
    if len(x) < min_bars:
        return None
    return x.High.max(), x.Low.min()


def fmt(r, d):
    return f"سقف {r[0]:.{d}f} | کف {r[1]:.{d}f}"


def tehran(ts):
    return (ts + pd.Timedelta(hours=3, minutes=30)).strftime("%H:%M")


def session_msg(data, ts, b):
    end_i, start_i = INFO[b]
    name, s_min, e_min = SESSIONS[end_i]
    start = ts - pd.Timedelta(minutes=e_min - s_min)
    lines = [f"پایان سشن {name}",
             f"ساعت پایان (تهران): {tehran(ts)}",
             f"شروع سشن: {SESSIONS[start_i][0]}", ""]
    got = False
    for pair, df in data.items():
        r = hl(df, start, ts, 12)
        if r is None:
            continue
        got = True
        d = dec(pair)
        lines += [pair, f"سقف: {r[0]:.{d}f}", f"کف: {r[1]:.{d}f}", ""]
    return "\n".join(lines) if got else None


def day_msg(data, ts):
    D = ts.normalize() - pd.Timedelta(days=1)
    monday = D - pd.Timedelta(days=D.weekday())
    wk_start, wk_end = monday - pd.Timedelta(days=7), monday
    lines = [f"پایان روز {D.strftime('%Y-%m-%d')}",
             f"ساعت (تهران): {tehran(ts)}", ""]
    got = False
    for pair, df in data.items():
        counts = df.groupby(df.index.normalize()).size()
        good = [d for d, c in counts.items() if c >= MIN_DAY_BARS]
        if D not in good:
            continue
        d = dec(pair)
        prev = [x for x in good if x < D]
        block = [pair]
        r = hl(df, D, D + pd.Timedelta(days=1))
        block.append(f"روز تمام‌شده: {fmt(r, d)}")
        if len(prev) >= 1:
            x = prev[-1]
            block.append(f"روز قبل ({x.strftime('%m-%d')}): {fmt(hl(df, x, x + pd.Timedelta(days=1)), d)}")
        if len(prev) >= 2:
            x = prev[-2]
            block.append(f"دو روز قبل ({x.strftime('%m-%d')}): {fmt(hl(df, x, x + pd.Timedelta(days=1)), d)}")
        w = hl(df, wk_start, wk_end)
        if w:
            block.append(f"هفته قبل: {fmt(w, d)}")
        lines += block + [""]
        got = True
    return "\n".join(lines) if got else None


def send_long(text):
    chunk = ""
    for block in text.split("\n\n"):
        if len(chunk) + len(block) + 2 > 3500:
            send(chunk)
            chunk = ""
        chunk += block + "\n\n"
    if chunk.strip():
        send(chunk)


def boundaries(now, days_back):
    out = []
    for k in range(days_back + 1):
        day = now.normalize() - pd.Timedelta(days=k)
        for b in BOUNDS:
            ts = day + pd.Timedelta(minutes=b)
            if ts <= now:
                out.append((b, ts))
    return sorted(out, key=lambda x: x[1], reverse=True)


def main():
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    now = pd.Timestamp.now(tz="UTC")
    forced = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    if forced:
        todo = boundaries(now, 4)          # برای تست: آخرین مرزی که داده دارد
    else:
        todo = [(b, ts) for b, ts in boundaries(now, 0)
                if (now - ts).total_seconds() <= WINDOW_MIN * 60
                and f"{ts:%Y-%m-%d}-{b}" not in state]
        if not todo:
            return

    data = {}
    for pair in PAIRS:
        try:
            df = load(pair)
            if df is not None and len(df):
                data[pair] = df
        except Exception as e:
            print(pair, "error:", e)
    if not data:
        return

    for b, ts in todo:
        s_msg = session_msg(data, ts, b)
        d_msg = day_msg(data, ts) if b == 0 else None
        if forced and not (s_msg or d_msg):
            continue                        # در تست، مرز بدون داده را رد کن
        if s_msg:
            send_long(s_msg)
        if d_msg:
            send_long(d_msg)
        state[f"{ts:%Y-%m-%d}-{b}"] = "done"
        if forced:
            break

    if not forced:
        keep = (now - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        state = {k: v for k, v in state.items() if k[:10] >= keep}
        json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
