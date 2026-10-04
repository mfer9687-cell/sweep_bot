import os
import json
import pandas as pd
import ichi_strategy as st
from sweep_bot import send

STATE = "ichimoku_state.json"
MAX_AGE = 25.0   # دقیقه؛ داده مسن‌تر رد می‌شود


def dec(p):
    return 2 if p == "XAUUSD" else 1 if p in ("NAS100", "US30") else 3 if "JPY" in p else 5


def main():
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    now = pd.Timestamp.now(tz="UTC")
    keep = (now - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    state = {k: v for k, v in state.items() if k[:10] >= keep}

    for pair in st.PAIRS:
        try:
            df = st.fetch(pair, "15m", "20d")
            if df is None or len(df) < 200:
                continue
            age = (now - (df.index[-1] + st.STEP["15m"])).total_seconds() / 60
            if age > MAX_AGE:
                print(pair, "stale", round(age, 1))
                continue
            side, at, k = st.signals(df, "15m")
        except Exception as e:
            print(pair, "error:", e)
            continue
        for i in (-1, -2):
            s = int(side.iloc[i])
            if s == 0:
                continue
            ts = df.index[i]
            key = f"{ts:%Y-%m-%d}|{pair}|{ts:%H%M}"
            dkey = f"{ts:%Y-%m-%d}|{pair}|count"
            if key in state or state.get(dkey, 0) >= st.MAX_PER_DAY:
                continue
            entry = float(df.Close.iloc[i])
            sl, tp, d = st.levels(entry, s, float(at.iloc[i]), float(k.iloc[i]))
            last = float(df.Close.iloc[-1])
            if not (min(sl, tp) < last < max(sl, tp)) or abs(last - entry) > 0.5 * d:
                continue            # سیگنال کهنه شده
            m = ts.hour * 60 + ts.minute
            sess = "لندن" if m < 750 else "نیویورک"
            tehran = (ts + pd.Timedelta(hours=3, minutes=30) + st.STEP["15m"]).strftime("%H:%M")
            p = dec(pair)
            send("\n".join([
                "سیگنال ایچیموکو",
                "خرید" if s == 1 else "فروش",
                f"سشن: {sess}",
                f"ورود: {entry:.{p}f}",
                f"حد ضرر: {sl:.{p}f}",
                f"حد سود: {tp:.{p}f}",
                f"ریسک به ریوارد: {st.RR}",
                f"ساعت کندل (تهران): {tehran}",
                pair]))
            state[key] = "sent"
            state[dkey] = state.get(dkey, 0) + 1
            break
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
