import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def index_to_time(df: pd.DataFrame, idx: int, time_column: str = "time_s") -> float:
    """Convert row index (i) to time in seconds."""
    if time_column not in df.columns:
        raise ValueError(f"time column '{time_column}' not found")
    if idx < 0 or idx >= len(df):
        raise IndexError(f"index {idx} out of range (len={len(df)})")
    value = df.at[idx, time_column]
    if pd.isna(value):
        return 0.0
    return float(value.item() if hasattr(value, "item") else value)


def time_to_nearest_index(df: pd.DataFrame, t_sec: float, time_column: str = "time_s") -> int:
    """Convert time in seconds to the nearest row index (i)."""
    if time_column not in df.columns:
        raise ValueError(f"time column '{time_column}' not found")
    if df.empty:
        raise ValueError("dataframe is empty")
    deltas = (df[time_column].to_numpy(dtype=float) - float(t_sec))
    return int(np.abs(deltas).argmin())


def load_tsdata_excel(path: Path | str, *, time_column: str = "存储相对时间时间(s)", sheet_name: str | int | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    logger.debug(f"读取Excel: {path}，原始形状={df.shape}，列={df.columns.tolist()}")
    df = df.drop(columns=["飞升点", "Unnamed: 18"], errors="ignore")
    logger.debug(f"删除辅助列后形状={df.shape}")

    if time_column in df.columns:
        logger.debug(f"重命名时间列 '{time_column}' -> 'time_s'")
        df = df.rename(columns={time_column: "time_s"})
    else:
        logger.debug(f"未找到时间列 '{time_column}'，后续可能无法检测")

    tw_columns = [c for c in df.columns if isinstance(c, str) and c.startswith("Tw")]
    logger.debug(f"识别到传感器列: {tw_columns}")
    df = df[["time_s"] + tw_columns].copy()

    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    for c in tw_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_s"]).sort_values("time_s")
    df = df[~df["time_s"].duplicated(keep="first")].reset_index(drop=True)
    logger.debug(f"清洗完成，形状={df.shape}")
    return df


@dataclass
class SensorEvent:
    sensor: str
    flyup_start_time_s: float
    start_row_index: int
    temperature_at_start: float
    dTdt_at_start: float

    confirm_time_s: float
    confirm_row_index: int

    baseline_dTdt_mean: float
    baseline_dTdt_std: float
    threshold_dTdt: float

    jump: float
    hold_ratio: float


class OnlineFlyUpAgent:
    """
    Online / causal fly-up detection:
    For each sensor independently:
      - maintain baseline stats for dT/dt from history (EWMA)
      - if dT/dt >= threshold for N consecutive samples -> enter CANDIDATE at start index
      - during candidate, monitor window of length hold_window_s:
            * jump = max(T) - T0 >= min_jump
            * hold_ratio = (T_end - T0)/(max(T)-T0) >= min_hold_ratio
        If satisfied at window end -> CONFIRMED event (start time returned)
        If falls back too much early -> reject candidate
    Global event = earliest confirmed start among sensors.
    """

    def __init__(
        self,
        *,
        time_column: str = "time_s",
        smoothing_window: int = 7,      # 对温度做轻微平滑，降低噪声尖刺
        # baseline for dT/dt (online EWMA)
        warmup_points: int = 50,        # 先用多少个点建立 baseline（只用过去）
        ewma_alpha: float = 0.02,       # baseline 更新速率（越小越稳）
        k_sigma: float = 6.0,           # 阈值：mean + k_sigma*std
        min_consecutive: int = 30,       # dT/dt 连续超阈值的点数才进入候选

        # persistence check window (causal)
        hold_window_s: float = 50.0,    # 候选开始后向前看多少秒决定“没降回去”
        baseline_temp_window: int = 50, # start 前用多少个点算温度基线 T0（中位数）
        min_jump: float = 0.05,         # 候选窗口内最大升温幅度至少多少℃
        min_hold_ratio: float = 0.4,    # 末端保持比例

        # optional early rejection: if temperature drops back near baseline, reject early
        early_drop_ratio: float = 0.5,  # 若 (T - T0)/(Tmax - T0) < early_drop_ratio 且 Tmax-T0已显著 -> 可提前判为“回落型”
    ):
        self.time_column = time_column
        self.smoothing_window = smoothing_window

        self.warmup_points = warmup_points
        self.ewma_alpha = ewma_alpha
        self.k_sigma = k_sigma
        self.min_consecutive = min_consecutive

        self.hold_window_s = hold_window_s
        self.baseline_temp_window = baseline_temp_window
        self.min_jump = min_jump
        self.min_hold_ratio = min_hold_ratio
        self.early_drop_ratio = early_drop_ratio

    @staticmethod
    def _smooth_series(y: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return y
        return pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        logger.debug(
            f"参数: smooth={self.smoothing_window}, warmup={self.warmup_points}, "
            f"alpha={self.ewma_alpha}, k={self.k_sigma}, consec={self.min_consecutive}, "
            f"hold_window_s={self.hold_window_s}, min_jump={self.min_jump}, min_hold_ratio={self.min_hold_ratio}"
        )
        if self.time_column not in df.columns:
            logger.debug(f"run aborted: time column '{self.time_column}' not found")
            return None

        sensors = [c for c in df.columns if isinstance(c, str) and c.startswith("Tw")]
        if not sensors:
            logger.debug(f"run aborted: no Tw sensor columns {df.columns.tolist()}")
            return None

        df = df.sort_values(self.time_column).reset_index(drop=True)
        t = df[self.time_column].to_numpy(dtype=float)
        n = len(df)
        min_length = max(3, self.warmup_points + 2)
        if n < min_length:
            logger.debug(f"run aborted: {n} rows < required {min_length}")
            return None

        # 采样间隔诊断
        if n >= 3:
            dt_med = float(np.median(np.diff(t)))
            approx_pts = (self.hold_window_s / dt_med) if dt_med > 0 else float('inf')
            logger.debug(
                f"采样间隔中位数≈{dt_med:.6f}s，hold_window_s={self.hold_window_s} ≈ {approx_pts:.1f} 个样本"
            )

        y_smooth = {s: self._smooth_series(df[s].to_numpy(dtype=float), self.smoothing_window) for s in sensors}

        # 每个传感器一个 FSM 状态
        state = {s: "BASELINE" for s in sensors}          # BASELINE / ARMED / CANDIDATE / CONFIRMED
        consec = {s: 0 for s in sensors}

        # online baseline stats for dT/dt: mean, var (EWMA)
        mu = {s: 0.0 for s in sensors}
        var = {s: 0.0 for s in sensors}
        initialized = {s: False for s in sensors}

        # candidate buffers
        cand_start_idx = {s: None for s in sensors}
        cand_start_time = {s: None for s in sensors}
        cand_T0 = {s: None for s in sensors}
        cand_Tmax = {s: -np.inf for s in sensors}

        confirmed_events: List[SensorEvent] = []

        # helper: compute baseline T0 just before start using raw temperature
        def compute_T0(sensor: str, idx: int) -> float:
            b0 = max(0, idx - self.baseline_temp_window)
            seg = df[sensor].to_numpy(dtype=float)[b0:idx]
            seg = seg[np.isfinite(seg)]
            logger.debug(f"计算T0: sensor={sensor}, idx={idx}, baseline_window=[{b0},{idx}), valid_points={seg.size}")
            return float(np.median(seg)) if seg.size else float(df.at[idx, sensor])

        # loop causally from i=1 (needs previous point)
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            #print("[DEBUG] 时间点 i={}, t={:.6f}s, dt={:.6f}s".format(i, t[i], dt))
            if dt <= 0:
                continue

            for s in sensors:
                # if state[s] == "BASELINE":
                #     print("[DEBUG] 传感器 {} 状态机: i={}, state={}, consec={}, mu={:.6f}, var={:.6f}".format(s, i, state[s], consec[s], mu[s], var[s]))
                # online dT/dt from smoothed temps
                dy = y_smooth[s][i] - y_smooth[s][i - 1]
                dTdt = dy / dt

                # --- update baseline stats only when not in candidate (to avoid contaminating baseline) ---
                if state[s] in ("BASELINE", "ARMED"):
                    if i <= self.warmup_points:
                        # warmup: accumulate simple mean/var with Welford-like update
                        if not initialized[s]:
                            mu[s] = float(dTdt)
                            var[s] = 0.0
                            initialized[s] = True
                        else:
                            a = 1.0 / max(2, i)
                            delta = dTdt - mu[s]
                            mu[s] += a * delta
                            var[s] = (1 - a) * var[s] + a * (delta ** 2)
                        continue  # warmup阶段不触发
                    else:
                        if not initialized[s]:
                            mu[s] = float(dTdt)
                            var[s] = 0.0
                            initialized[s] = True
                        else:
                            alpha = self.ewma_alpha
                            delta = dTdt - mu[s]
                            mu[s] = mu[s] + alpha * delta
                            var[s] = (1 - alpha) * var[s] + alpha * (delta ** 2)

                std = float(np.sqrt(max(var[s], 1e-12)))
                threshold = mu[s] + self.k_sigma * std

                # --- FSM transitions ---
                if state[s] in ("BASELINE", "ARMED"):
                    # 只看正向斜率触发
                    if dTdt >= threshold and np.isfinite(dTdt):
                        consec[s] += 1
                        time = index_to_time(df, i, self.time_column)
                        # print("[DEBUG] 传感器 {} 状态机: i={}, t={}, state={}, consec={}, dTdt={:.6f} >= threshold={:.6f}".format(s, i, time, state[s], consec[s], dTdt, threshold))
                        if consec[s] == 1:
                            # 记第一下超阈值的位置，作为候选 start
                            pass
                        if consec[s] >= self.min_consecutive:
                            # 进入候选态：start 设为连续段的起点
                            start_idx = i - self.min_consecutive + 1
                            state[s] = "CANDIDATE"
                            cand_start_idx[s] = start_idx
                            cand_start_time[s] = float(t[start_idx])
                            cand_T0[s] = compute_T0(s, start_idx)
                            cand_Tmax[s] = float(np.max(df[s].to_numpy(dtype=float)[start_idx:i + 1]))
                            # print(
                            #     f"[DEBUG] 进入候选: sensor={s}, start_idx={start_idx}, "
                            #     f"t0={cand_start_time[s]:.6f}s, T0={cand_T0[s]:.6f}, "
                            #     f"baseline dT/dt mean={mu[s]:.6f}, std={std:.6f}, threshold={threshold:.6f}"
                            # )
                    else:
                        consec[s] = 0
                        state[s] = "ARMED"

                elif state[s] == "CANDIDATE":
                    start_idx = cand_start_idx[s]
                    assert start_idx is not None
                    T0 = float(cand_T0[s])
                    # 更新 Tmax
                    cand_Tmax[s] = max(float(cand_Tmax[s]), float(df.at[i, s]))
                    Tmax = float(cand_Tmax[s])
                    jump = Tmax - T0

                    # 可选：提前拒绝“回落型”——如果已经有明显jump，但当前又回到很低
                    if jump >= self.min_jump:
                        hold_now = (float(df.at[i, s]) - T0) / (jump + 1e-12)
                        if hold_now < self.early_drop_ratio:
                            # 说明像图1那样冲上去又掉回来了
                            state[s] = "ARMED"
                            consec[s] = 0
                            cand_start_idx[s] = None
                            cand_start_time[s] = None
                            cand_T0[s] = None
                            cand_Tmax[s] = -np.inf
                            logger.debug(
                                f"候选提前拒绝(回落型): sensor={s}, hold_now={hold_now:.3f}, "
                                f"jump={jump:.6f}, early_drop_ratio<{self.early_drop_ratio}"
                            )
                            continue

                    # 到窗口末端再正式判定
                    elapsed = float(t[i]) - float(cand_start_time[s])
                    elapsed_pts = (i - start_idx + 1)
                    if elapsed >= self.hold_window_s:
                        Tend = float(df.at[i, s])
                        jump = Tmax - T0
                        if jump >= self.min_jump:
                            hold_ratio = (Tend - T0) / (jump + 1e-12)
                            if hold_ratio >= self.min_hold_ratio:
                                # CONFIRMED
                                state[s] = "CONFIRMED"
                                evt = SensorEvent(
                                    sensor=s,
                                    flyup_start_time_s=float(cand_start_time[s]),
                                    start_row_index=int(start_idx),
                                    temperature_at_start=float(df.at[start_idx, s]),
                                    dTdt_at_start=float((y_smooth[s][start_idx] - y_smooth[s][start_idx - 1]) / (t[start_idx] - t[start_idx - 1]))
                                    if start_idx >= 1 else float("nan"),
                                    confirm_time_s=float(t[i]),
                                    confirm_row_index=int(i),
                                    baseline_dTdt_mean=float(mu[s]),
                                    baseline_dTdt_std=float(std),
                                    threshold_dTdt=float(threshold),
                                    jump=float(jump),
                                    hold_ratio=float(hold_ratio),
                                )
                                confirmed_events.append(evt)
                                logger.debug(
                                    f"确认飞升: sensor={s}, start={cand_start_time[s]:.6f}s -> confirm={t[i]:.6f}s, "
                                    f"jump={jump:.6f}, hold_ratio={hold_ratio:.3f}, 窗口持续={elapsed:.3f}s(~{elapsed_pts}点)"
                                )
                            else:
                                # not persistent -> reject
                                state[s] = "ARMED"
                                consec[s] = 0
                                logger.debug(
                                    f"候选未通过(保持不足): sensor={s}, jump={jump:.6f}, hold_ratio={hold_ratio:.3f} < {self.min_hold_ratio}, "
                                    f"窗口持续={elapsed:.3f}s(~{elapsed_pts}点)"
                                )
                        else:
                            # jump too small -> reject
                            state[s] = "ARMED"
                            logger.debug(
                                f"候选未通过(跳幅过小): sensor={s}, jump={jump:.6f} < {self.min_jump}, 窗口持续={elapsed:.3f}s(~{elapsed_pts}点)"
                            )

                        cand_start_idx[s] = None
                        cand_start_time[s] = None
                        cand_T0[s] = None
                        cand_Tmax[s] = -np.inf

        if not confirmed_events:
            logger.debug(f"run aborted: 0 confirmed events after scanning {len(sensors)} sensors")
            return None

        # 全局规则：取“飞升开始时间最早”的那个事件
        confirmed_events.sort(key=lambda e: e.flyup_start_time_s)
        first = confirmed_events[0]

        return {
            "all confirmed_events": confirmed_events,
            "flyup_time_s": first.flyup_start_time_s,
            "sensor": first.sensor,
            "temperature_at_start": first.temperature_at_start,
            "dTdt_at_start": first.dTdt_at_start,
            "confirm_time_s": first.confirm_time_s, 
            "jump": first.jump,
            "hold_ratio": first.hold_ratio,
            "threshold_dTdt": first.threshold_dTdt,
            "baseline_dTdt_mean": first.baseline_dTdt_mean,
            "baseline_dTdt_std": first.baseline_dTdt_std,
            "per_sensor_confirmed": [
                {
                    "sensor": e.sensor,
                    "flyup_start_time_s": e.flyup_start_time_s,
                    "confirm_time_s": e.confirm_time_s,
                    "jump": e.jump,
                    "hold_ratio": e.hold_ratio,
                    "threshold_dTdt": e.threshold_dTdt,
                }
                for e in confirmed_events
            ],
            "definition": (
                "Online causal FSM: if dT/dt exceeds adaptive threshold for >=N consecutive samples -> CANDIDATE; "
                "within hold_window_s require jump>=min_jump and hold_ratio>=min_hold_ratio to CONFIRM; "
                "global flyup is earliest confirmed start among sensors."
            ),
        }
# 3 5 
def main() -> None:
    tsdata_path = Path(__file__).with_name("tsdata.xlsx")
    df = load_tsdata_excel(tsdata_path, sheet_name="Sheet3")
    agent = OnlineFlyUpAgent(
        smoothing_window=10,
        warmup_points=100,
        ewma_alpha=0.04,
        k_sigma=6.0,
        min_consecutive=10,
        hold_window_s=20.0,
        baseline_temp_window=100,
        min_jump=0.1,
        min_hold_ratio=0.6,
        early_drop_ratio=-1,
    )

    out = agent.run(df)
    if out is None:
        logger.info("未检测到传热恶化（在线判定未确认）。")
        return

    logger.info(
        f"sensor={out['sensor']}, "
        f"start_time={out['flyup_time_s']:.6f}s, "
        f"T_start={out['temperature_at_start']:.6f}, "
        f"dT/dt_start={out['dTdt_at_start']:.6f} ℃/s, "
        f"confirmed_at={out['confirm_time_s']:.6f}s, "
        f"jump={out['jump']:.6f} ℃, hold_ratio={out['hold_ratio']:.3f}"
    )

if __name__ == "__main__":
    main()