import numpy as np
import lamd_strategy as ls
from sweep_bot import send

RUNS = [("5m", "60d"), ("15m", "60d")]
CONFIGS = [(True, "ext"), (True, "2R"), (True, "1.5R"),
           (False, "ext"), (False, "2R"), (False, "1.5R")]
TP_NAME = {"ext": "هدف نقدینگی خارجی (حداقل ۱:۳)",
           "2R": "هدف ۲ برابر ریسک", "1.5R": "هدف ۱٫۵ برابر ریسک"}


def line(label, s):
    if s is None:
        return f"{label}: معامله‌ای نبود"
    n, win, avg, pf = s
    return f"{label}: {n} معامله | برد {win:.0f}% | میانگین {avg:+.2f} | ضریب سود {pf:.2f}"


def main():
    for tf, period in RUNS:
        data = {}
        for p in ls.PAIRS:
            try:
                df = ls.fetch(p, tf, period)
                if df is not None and len(df) > 500:
                    data[p] = df
            except Exception as e:
                print(p, "error:", e)
        if not data:
            send(f"بک‌تست لمد {tf}: داده‌ای نیامد")
            continue
        out = [f"بک‌تست LAMD-FVG تایم {tf}", f"تعداد نماد: {len(data)}", ""]
        for use_bias, tp in CONFIGS:
            allt, gold = [], []
            for p, df in data.items():
                tr = ls.backtest(df, tf, use_bias, tp)
                allt += tr
                if p == "XAUUSD":
                    gold = tr
            out += [("با بایاس ۴ ساعته" if use_bias else "بدون بایاس") + " | " + TP_NAME[tp],
                    line("همه", ls.stats([x[1] for x in allt])),
                    line("نیمه اول", ls.stats([x[1] for x in allt if x[2] == 0])),
                    line("نیمه دوم", ls.stats([x[1] for x in allt if x[2] == 1])),
                    line("فقط طلا", ls.stats([x[1] for x in gold])), ""]
        text = "\n".join(out)
        print(text)
        send(text[:3900])


if __name__ == "__main__":
    main()
