import numpy as np
import vwap_strategy as vs
from sweep_bot import send

RUNS = [("5m", "60d"), ("15m", "60d")]
CONFIGS = [(False, 1.5), (False, 2.0), (True, 1.5), (True, 2.0)]
FIXED = (True, 2.0)    # تنظیم ثابتِ از پیش تعیین‌شده برای جدول هر نماد


def line(label, s):
    if s is None:
        return f"{label}: معامله‌ای نبود"
    n, win, avg, pf = s
    return f"{label}: {n} معامله | برد {win:.0f}% | میانگین {avg:+.2f} | ضریب سود {pf:.2f}"


def main():
    for tf, period in RUNS:
        data = {}
        for p in vs.PAIRS:
            try:
                df = vs.fetch(p, tf, period)
                if df is not None and len(df) > 500:
                    data[p] = df
            except Exception as e:
                print(p, "error:", e)
        if not data:
            send(f"بک‌تست وی‌وپ {tf}: داده‌ای نیامد")
            continue
        out = [f"بک‌تست وی‌وپ تایم {tf}", f"تعداد نماد: {len(data)}", ""]
        per_pair = {}
        for use_liq, k in CONFIGS:
            allt = []
            for p, df in data.items():
                tr = vs.backtest(df, tf, k, use_liq)
                allt += tr
                if (use_liq, k) == FIXED:
                    per_pair[p] = [x[1] for x in tr]
            name = "با نقدینگی" if use_liq else "بدون نقدینگی"
            rr = np.mean([x[3] for x in allt]) if allt else 0
            out += [f"{name} | فاصله باند {k}",
                    line("همه", vs.stats([x[1] for x in allt])),
                    line("نیمه اول", vs.stats([x[1] for x in allt if x[2] == 0])),
                    line("نیمه دوم", vs.stats([x[1] for x in allt if x[2] == 1])),
                    f"میانگین ریسک به ریوارد برنامه‌ریزی‌شده: {rr:.1f}", ""]
        out.append("هر نماد (با نقدینگی، فاصله باند 2.0)")
        for p, rs in per_pair.items():
            out += [p, line("", vs.stats(rs))]
        text = "\n".join(out)
        print(text)
        send(text[:3900])


if __name__ == "__main__":
    main()
