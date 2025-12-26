import os
import logging
import time
import datetime
import pandas as pd
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# ---- modules ----
from detection.ewma import detect_first_across_channels
from plot.vis import plot_timeseries

# ---- PydanticAI agent framework ----
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# ----------------------------
# In-memory session state
# ----------------------------

SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "df": None,
            "filename": None,
            "file_id": 0,
            "last_plot_b64": None,
            "last_detection": None,
            "runs": [],
            "events": [],
        }
    return SESSIONS[session_id]

# ----------------------------
class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    tool_calls: List[Dict[str, Any]]
    artifacts: Dict[str, Any]

class Deps(BaseModel):
    session_id: str
    tool_calls: List[Dict[str, Any]] = []

# ----------------------------
# Proxy and OpenAI config for testing
# ----------------------------

USE_LOCAL_MODEL = True

PROXY_CONFIG = {
    "SOCKS_PROXY": "socks5://127.0.0.1:1081",
    "HTTP_PROXY":"http://127.0.0.1:8001",
    "HTTPS_PROXY":"http://127.0.0.1:8001",
}

os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

for key, value in PROXY_CONFIG.items():
    os.environ[key] = value

# ----------------------------
# Logging setup
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def emit_progress(tool_name: str, phase: str):
    """简单的进度提示：记录当前工具和阶段"""
    logger.info(f"[progress] {tool_name} - {phase} ░▒▓")

def format_ctx(ctx: RunContext[Deps]) -> str:
    """压缩输出 ctx 关键信息，避免过长日志"""
    try:
        deps = ctx.deps
        return f"session_id={deps.session_id}, tool_calls={len(deps.tool_calls)}"
    except Exception as e:
        return f"ctx_format_error={e}"

# ----------------------------
# PydanticAI Agent + tools
# ----------------------------

def log_tool(deps: Deps, tool: str, args: Dict[str, Any], result_preview: Any, result: Any = None):

    rec = {"tool": tool, "args": args, "result_preview": result_preview}
    if result is not None:
        rec["result"] = result
    deps.tool_calls.append(rec)

    try:
        sess = get_session(deps.session_id)
        q = sess.get("event_q")
        if q is not None:
            payload = {
                "tool": tool,
                "args": args,
                "result_preview": result_preview,
                "result": result,
                "ts": datetime.datetime.now().isoformat()
            }
            q.put_nowait({"type": "tool_call", "payload": payload})
    except Exception:
        pass


def _filter_sensors(df: pd.DataFrame, sensors: Optional[List[str]]) -> Optional[List[str]]:
    """传感器过滤：未指定返回 None（全保留）；指定则只保留存在的列，全部无效则报错。"""
    if sensors is None:
        return None
    valid = [s for s in sensors if s in df.columns]
    if not valid:
        raise ValueError(f"传感器列未找到：{sensors}")
    return valid


if USE_LOCAL_MODEL:
    LOCAL_MODEL_BASE_URL = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct") # "qwen3:latest"

    logger.info(f"使用本地 Ollama 模型: {LOCAL_MODEL_NAME}")
    logger.info(f"Base URL: {LOCAL_MODEL_BASE_URL}")
    os.environ["NO_PROXY"] = os.getenv("NO_PROXY", "") + ",localhost,127.0.0.1"

    provider = OpenAIProvider(base_url=LOCAL_MODEL_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "ollama"))
    model = OpenAIChatModel(LOCAL_MODEL_NAME, provider=provider)


agent = Agent(
    model,
    deps_type=Deps,
    system_prompt=(
        "你是一名工程时序异常检测助手，回答要简短、中文、工程友好。\n"
        "数据已预加载到会话，无需再 load。\n"
        "工具调用策略：\n"
        "0) 首先默认调用list_sensors列出所有可用传感器。\n"
        "1) 用户提到“画图/可视化”→ 调用 visualize，将list_sensors的结果作为参数。\n"
        "2) 用户提到“寻找飞升点/检测”→ 调用 detect，将list_sensors的结果作为参数。。\n"
        "3) 若既要检测又要画图，先 detect 再 visualize；若需要标记飞升点，把 detect 的 mark_index 传给 visualize。\n"
        "4) 最后调用 summarize 输出中文报告。\n"
        "异常处理：未加载数据或指令无关时，用简短中文提示并引导上传/重述需求。\n"
        "禁止中英夹杂。\n"
    ),
)

@agent.tool
def list_sensors(
    ctx: RunContext[Deps],
    time_col: str = "time_s",
    sensors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """列出可用传感器；未指定则返回全部数值列，指定则仅返回存在的列。"""
    sess = get_session(ctx.deps.session_id)
    df = sess.get("df")
    if df is None:
        raise ValueError("No data loaded.")
    # 默认全部数值列（排除时间列）
    if sensors is None:
        available = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    else:
        available = [s for s in sensors if s in df.columns]
        if not available:
            raise ValueError(f"传感器列未找到：{sensors}")
    result = {"time_col": time_col, "sensors": available}
    log_tool(ctx.deps, "list_sensors", {"time_col": time_col, "requested": sensors}, result, result=result)
    return result


@agent.tool
async def detect(
    ctx: RunContext[Deps],
    smoothing_window: int = 7,
    warmup_points: int = 10,
    ewma_alpha: float = 0.02,
    k_sigma: float = 2.0,
    min_consecutive: int = 5,
    hold_window_s: float = 100.0,
    baseline_temp_window: int = 10,
    min_jump: float = 0.1,
    min_hold_ratio: float = 0.6,
    early_drop_ratio: float = 0.5,
    sensors: Optional[List[str]] = None,
) -> str:
    start = time.perf_counter_ns()
    emit_progress("detect", "start")
    logger.info(f"[detect] ctx: {format_ctx(ctx)}")
    logger.info(
        "[detect] start params: "
        f"smooth={smoothing_window}, warmup={warmup_points}, alpha={ewma_alpha}, k={k_sigma}, "
        f"consec={min_consecutive}, hold_s={hold_window_s}, base_T_win={baseline_temp_window}, "
        f"jump>={min_jump}, hold_ratio>={min_hold_ratio}, early_drop<{early_drop_ratio}, sensors={sensors}"
    )
    sess = get_session(ctx.deps.session_id)
    df = sess.get("df")
    if df is None:
        raise ValueError("No data loaded.")
    if "time_s" not in df.columns:
        raise ValueError("未找到存储相对时间列，输入数据不完整，请重新检查上传。")

    selected = _filter_sensors(df, sensors) if sensors else None
    detect_output = detect_first_across_channels(
        df,
        "time_s",
        selected,
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
    logger.debug(f"[detect] output={detect_output}")

    try:
        idx = detect_output.get("earliest_index")
        channel = detect_output.get("earliest_channel")
        time_val = detect_output.get("earliest_time")

        detection_data = {
            "earliest_index": idx,
            "earliest_time": time_val,
            "earliest_channel": channel,
            "mark_index": idx,  # 用于后续 visualize 标记
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
        sess["last_detection"] = detection_data

        if idx is not None and time_val is not None:
            summary = (
                f"检测到飞升点：{channel or '未知通道'} 索引 {idx}, 时刻 {time_val:.2f}s "
                f"(k={k_sigma}, consec≥{min_consecutive}, jump≥{min_jump}, hold_ratio≥{min_hold_ratio})"
            )
        else:
            summary = (
                "未检测到飞升点 "
                f"(k={k_sigma}, consec≥{min_consecutive}, jump≥{min_jump}, hold_ratio≥{min_hold_ratio})"
            )

        logger.info(f"[detect] summary: {summary}")
        log_tool(ctx.deps, "detect", detection_data["params"], summary, result=detection_data)
        return summary
    except Exception as e:
        logger.error(f"[detect] failed: {e}")
        raise
    finally:
        logger.info(f"[detect] elapsed={(time.perf_counter_ns()-start)/1e6:.2f} ms")
        emit_progress("detect", "done")



@agent.tool
def visualize(ctx: RunContext[Deps], time_col: str = "time_s", sensors: Optional[List[str]] = None) -> Dict[str, Any]:
    start = time.perf_counter_ns()
    emit_progress("visualize", "start")
    logger.info(f"[visualize] ctx: {format_ctx(ctx)}")
    logger.info(f"[visualize] start, sensors={sensors}, time_col={time_col}")
    sess = get_session(ctx.deps.session_id)
    df = sess["df"]
    if df is None:
        raise ValueError("No data loaded.")
    # 若未指定 sensors，则用全部数值列（除时间列）；指定则过滤
    selected = _filter_sensors(df, sensors) if sensors else None
    b64 = plot_timeseries(df, time_col=time_col, channels=selected, mark_index=None, title="Uploaded time series")
    logger.debug(f"[visualize] plot length={len(b64)}")
    sess["last_plot_b64"] = b64
    preview = {"png_base64_len": len(b64), "time_col": time_col, "channels": selected}
    log_tool(ctx.deps, "visualize", {"time_col": time_col, "channels": selected}, preview)
    logger.info(f"[visualize] elapsed={(time.perf_counter_ns()-start)/1e6:.2f} ms")
    emit_progress("visualize", "done")
    return {"status": "ok", "note": "plot saved to session"}


@agent.tool
def summarize(ctx: RunContext[Deps]) -> str:
    start = time.perf_counter_ns()
    emit_progress("summarize", "start")
    logger.info(f"[summarize] ctx: {format_ctx(ctx)}")
    logger.info("[summarize] start")
    sess = get_session(ctx.deps.session_id)
    det = sess.get("last_detection", None)
    if det is None:
        msg = "No detection result yet. Run detect first."
        log_tool(ctx.deps, "summarize", {}, msg)
        return msg

    if det["earliest_index"] is None:
        msg = (
            "结论：未检出满足阈值与持续性条件的异常升温点。\n"
            f"参数：ewma_alpha={det['params']['ewma_alpha']}, k_sigma={det['params']['k_sigma']}, "
            f"min_consecutive={det['params']['min_consecutive']}。\n"
            "提示：若认为存在事件，可降低 k_sigma 或 min_consecutive 后重跑。"
        )
        log_tool(ctx.deps, "summarize", {}, "no_anomaly")
        logger.info(f"[summarize] elapsed={(time.perf_counter_ns()-start)/1e6:.2f} ms")
        return msg

    msg = (
        "结论：检测到最早的异常升温点（以最先出现异常的测点为准）。\n"
        f"- 测点：{det.get('earliest_channel')}\n"
        f"- 时间：t={det.get('earliest_time')}\n"
        f"- 索引：{det.get('earliest_index')}\n"
        "解释：该点之后导数连续超过阈值，满足持续性条件。\n"
        f"参数：ewma_alpha={det['params']['ewma_alpha']}, k_sigma={det['params']['k_sigma']}, "
        f"min_consecutive={det['params']['min_consecutive']}。"
    )
    log_tool(ctx.deps, "summarize", {}, {"earliest_time": det.get("earliest_time"), "channel": det.get("earliest_channel")}, result=msg)
    logger.info(f"[summarize] elapsed={(time.perf_counter_ns()-start)/1e6:.2f} ms")
    emit_progress("summarize", "done")
    return msg
