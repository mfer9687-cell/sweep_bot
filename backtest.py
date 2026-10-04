import ichi_strategy as st
from sweep_bot import send

RUNS = [("15m", "60d"), ("1h", "730d")]
RRS = [1.0, 1.5, 2.0]


def line(label, s):
    if s is None:
        return f"{label}: معامله‌ای نبود"
    n, win, avg, pf = s
    return f"{label}: {n} معامله | برد {win:.0f}% | میانگین {avg:+.2f} | ضریب سود {pf:.2f}"


def main():
    for tf, period in RUNS:
        data = {}
        for p in st.PAIRS:
            try:
                df = st.fetch(p, tf, period)
                if df is not None and len(df) > 300:
                    data[p] = df
            except Exception as e:
                print(p, "error:", e)
        if not data:
            send(f"بک‌تست {tf}: داده‌ای نیامد")
            continue
        out = [f"بک‌تست تایم {tf}", f"تعداد نماد: {len(data)}", ""]
        per_pair = {}
        for rr in RRS:
            allr, h1, h2 = [], [], []
            for p, df in data.items():
                tr = st.backtest(df, tf, rr)
                if rr == st.RR:
                    per_pair[p] = [x[1] for x in tr]
                allr += [x[1] for x in tr]
                h1 += [x[1] for x in tr if x[2] == 0]
                h2 += [x[1] for x in tr if x[2] == 1]
            out += [f"ریسک به ریوارد {rr}",
                    line("همه", st.stats(allr)),
                    line("نیمه اول", st.stats(h1)),
                    line("نیمه دوم", st.stats(h2)), ""]
        out.append(f"هر نماد (ریسک به ریوارد {st.RR})")
        for p, rs in per_pair.items():
            s = st.stats(rs)
            out += [p, line("", s) if s else "معامله‌ای نبود"]
        text = "\n".join(out)
        print(text)
        send(text[:3900])


if __name__ == "__main__":
    main()
