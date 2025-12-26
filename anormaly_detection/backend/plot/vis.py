import pandas as pd
import io
import base64
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def plot_timeseries(
    df: pd.DataFrame,
    time_col: str = "time_s",
    channels: Optional[List[str]] = None,
    mark_index: Optional[int] = None,
    title: str = "Time Series",
) -> str:

    if channels is None:
        channels = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]

    t = df[time_col].to_numpy(dtype=float)

    fig = plt.figure()
    for c in channels:
        plt.plot(t, df[c].to_numpy(dtype=float), label=c)

    if mark_index is not None and 0 <= mark_index < len(df):
        plt.axvline(float(t[mark_index]), linestyle="--")
    plt.xlabel(time_col)
    plt.ylabel("value")
    plt.title(title)
    plt.legend(loc="best")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")