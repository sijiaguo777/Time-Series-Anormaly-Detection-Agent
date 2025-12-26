import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

# ----------------------------
# EWMA detector
# ----------------------------
def _smooth_series(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()


def ewma_detect_one_series(
    t: np.ndarray,
    x: np.ndarray,
    *,
    smoothing_window: int = 7,
    warmup_points: int = 50,
    ewma_alpha: float = 0.02,
    k_sigma: float = 6.0,
    min_consecutive: int = 30,
    hold_window_s: float = 50.0,
    baseline_temp_window: int = 50,
    min_jump: float = 0.05,
    min_hold_ratio: float = 0.4,
    early_drop_ratio: float = 0.5,
) -> Dict[str, Any]:
    n = len(x)
    min_length = max(3, warmup_points + 2)
    if n < min_length:
        return {"anomaly_index": None, "reason": "too_short", "required_len": min_length}

    y = _smooth_series(np.asarray(x, dtype=float), smoothing_window)
    t = np.asarray(t, dtype=float)

    state = "BASELINE"
    consec = 0
    mu = 0.0
    var = 0.0
    initialized = False

    cand_start_idx: Optional[int] = None
    cand_start_time: Optional[float] = None
    cand_T0: Optional[float] = None
    cand_Tmax: float = -np.inf

    def compute_T0(idx: int) -> float:
        b0 = max(0, idx - baseline_temp_window)
        seg = x[b0:idx]
        seg = seg[np.isfinite(seg)]
        return float(np.median(seg)) if seg.size else float(x[idx])

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue

        dTdt = (y[i] - y[i - 1]) / dt

        if state in ("BASELINE", "ARMED"):
            if i <= warmup_points:
                if not initialized:
                    mu = float(dTdt)
                    var = 0.0
                    initialized = True
                else:
                    a = 1.0 / max(2, i)
                    delta = dTdt - mu
                    mu += a * delta
                    var = (1 - a) * var + a * (delta ** 2)
                continue
            else:
                if not initialized:
                    mu = float(dTdt)
                    var = 0.0
                    initialized = True
                else:
                    alpha = ewma_alpha
                    delta = dTdt - mu
                    mu = mu + alpha * delta
                    var = (1 - alpha) * var + alpha * (delta ** 2)

        std = float(np.sqrt(max(var, 1e-12)))
        threshold = mu + k_sigma * std

        if state in ("BASELINE", "ARMED"):
            if np.isfinite(dTdt) and dTdt >= threshold:
                consec += 1
                if consec >= min_consecutive:
                    start_idx = i - min_consecutive + 1
                    state = "CANDIDATE"
                    cand_start_idx = start_idx
                    cand_start_time = float(t[start_idx])
                    cand_T0 = compute_T0(start_idx)
                    cand_Tmax = float(np.max(x[start_idx : i + 1]))
            else:
                consec = 0
                state = "ARMED"

        elif state == "CANDIDATE":
            assert cand_start_idx is not None and cand_start_time is not None and cand_T0 is not None
            cand_Tmax = max(float(cand_Tmax), float(x[i]))
            jump = cand_Tmax - cand_T0

            if jump >= min_jump:
                hold_now = (float(x[i]) - cand_T0) / (jump + 1e-12)
                if hold_now < early_drop_ratio:
                    state = "ARMED"
                    consec = 0
                    cand_start_idx = None
                    cand_start_time = None
                    cand_T0 = None
                    cand_Tmax = -np.inf
                    continue

            elapsed = float(t[i]) - float(cand_start_time)
            if elapsed >= hold_window_s:
                Tend = float(x[i])
                jump = cand_Tmax - cand_T0
                if jump >= min_jump:
                    hold_ratio = (Tend - cand_T0) / (jump + 1e-12)
                    if hold_ratio >= min_hold_ratio:
                        return {
                            "anomaly_index": int(cand_start_idx),
                            "confirm_index": int(i),
                            "start_time": float(cand_start_time),
                            "confirm_time": float(t[i]),
                            "jump": float(jump),
                            "hold_ratio": float(hold_ratio),
                            "baseline_dTdt_mean": float(mu),
                            "baseline_dTdt_std": float(std),
                            "threshold_dTdt": float(threshold),
                        }
                state = "ARMED"
                consec = 0
                cand_start_idx = None
                cand_start_time = None
                cand_T0 = None
                cand_Tmax = -np.inf

    return {"anomaly_index": None, "reason": "not_found", "k_sigma": float(k_sigma)}


def detect_first_across_channels(
    df: pd.DataFrame,
    time_col: str = "time_s",
    channels: Optional[List[str]] = None,
    *,
    smoothing_window: int = 10,
    warmup_points: int = 100,
    ewma_alpha: float = 0.02,
    k_sigma: float = 6.0,
    min_consecutive: int = 10,
    hold_window_s: float = 20.0,
    baseline_temp_window: int = 100,
    min_jump: float = 0.1,
    min_hold_ratio: float = 0.6,
    early_drop_ratio: float = -1,
) -> Dict[str, Any]:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    t = df[time_col].to_numpy(dtype=float)
    if channels is None:
        channels = [c for c in df.columns if c.startswith("Tw") and pd.api.types.is_numeric_dtype(df[c])]

    per: Dict[str, Any] = {}
    earliest = None
    earliest_channel = None
    for c in channels:
        x = df[c].to_numpy(dtype=float)
        r = ewma_detect_one_series(
            t,
            x,
            smoothing_window=smoothing_window,
            warmup_points=warmup_points,
            ewma_alpha=ewma_alpha,
            k_sigma=k_sigma,
            min_consecutive=min_consecutive,
            hold_window_s=hold_window_s,
            baseline_temp_window=baseline_temp_window,
            min_jump=min_jump,
            min_hold_ratio=min_hold_ratio,
            early_drop_ratio=early_drop_ratio,
        )
        per[c] = r
        idx = r.get("anomaly_index", None)
        if idx is not None:
            if (earliest is None) or (idx < earliest):
                earliest = idx
                earliest_channel = c

    return {
        "time_col": time_col,
        "channels": channels,
        "earliest_channel": earliest_channel,
        "earliest_index": earliest,
        "earliest_time": None if earliest is None else float(t[earliest]),
        "per_channel": per,
        "params": {
            "smoothing_window": smoothing_window,
            "warmup_points": warmup_points,
            "ewma_alpha": ewma_alpha,
            "k_sigma": k_sigma,
            "min_consecutive": min_consecutive,
            "hold_window_s": hold_window_s,
            "baseline_temp_window": baseline_temp_window,
            "min_jump": min_jump,
            "min_hold_ratio": min_hold_ratio,
            "early_drop_ratio": early_drop_ratio,
        },
    }