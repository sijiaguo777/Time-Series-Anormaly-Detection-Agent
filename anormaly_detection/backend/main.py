import io
import time
import json
import uuid
import asyncio
import datetime
import pandas as pd
from fastapi import FastAPI, Request
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

# ---- modules ----
from agent import ChatResponse, Deps, log_tool, get_session, agent, logger

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Time-Series Anormaly Detection Agent")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_KEYWORDS = {"飞升", "检测", "detect", "画图", "可视化", "plot", "visualize", "报告", "总结", "summary"}

STREAM_DELAY = 0.0
RETRY_TIMEOUT = 15000  # millisecond
PING_INTERVAL = 10.0   # seconds，keep-alive 防断线

def is_valid_user_message(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    return any(k.lower() in m for k in ALLOWED_KEYWORDS)

def create_sse_event(event_type: str, data: Dict[str, Any], event_id: Optional[str] = None) -> Dict[str, Any]:
    """创建SSE事件格式"""
    return {
        "event": event_type,
        "retry": RETRY_TIMEOUT,
        "data": json.dumps(data, ensure_ascii=False),
        "id": event_id or str(uuid.uuid4()),
    }

def load_uploaded_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    fname = (filename or "").lower()
    try:
        if fname.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        if fname.endswith(".xls"):
            try:
                return pd.read_excel(io.BytesIO(content), engine="xlrd")
            except ImportError as ie:
                raise ValueError("读取 .xls 需要安装 xlrd，或请将文件另存为 .xlsx/.csv 再上传。") from ie
        if fname.endswith(".csv"):
            
            return pd.read_csv(io.BytesIO(content))
            
        raise ValueError("不支持的文件类型，请上传 .xlsx/.xls/.csv。")
    
    except ValueError as ve:
        msg = str(ve)
        if "Excel file format cannot be determined" in msg:
            raise ValueError("Excel 文件格式无法识别，可能文件损坏或扩展名与内容不一致，请确认可正常打开，或另存为 .xlsx / 导出为 CSV 后再上传。") from ve
        raise
    except Exception as e:
        raise ValueError(f"读取文件失败：{e}") from e

async def sse_event_generator(request: Request, session_id: str, message: str, file_data: Optional[Dict] = None):
    """SSE事件生成器，用于实时发送agent执行过程中的事件"""
    session = get_session(session_id)
    session["events"] = []
    
    # 发送开始事件
    yield create_sse_event("agent_start", {
        "message": "开始处理您的请求",
        "session_id": session_id,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        # 检查请求是否断开
        if await request.is_disconnected():
            return
            
        # 验证用户消息
        if not is_valid_user_message(message):
            yield create_sse_event("agent_error", {
                "error": "当前仅支持：画图/可视化、飞升点检测/报告等相关指令，请重新描述需求（示例：'请帮我检测飞升点并画图'）。",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return
            
        # 发送验证通过事件
        yield create_sse_event("validation_passed", {
            "message": "请求验证通过，开始处理",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        if await request.is_disconnected():
            return
            
        # 处理文件上传
        if file_data:
            if file_data.get("error"):
                yield create_sse_event("agent_error", {
                    "error": file_data["error"],
                    "filename": file_data.get("filename"),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                return
            yield create_sse_event("file_processing", {
                "message": f"正在处理文件: {file_data.get('filename', 'unknown')}",
                "filename": file_data.get('filename'),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            if await request.is_disconnected():

                return
                
            # 发送文件处理完成事件
            yield create_sse_event("file_processed", {
                "message": f"文件处理完成: {file_data.get('filename')}",
                "shape": file_data.get('shape'),
                "columns": file_data.get('columns'),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
        if await request.is_disconnected():
            return
            
        # 开始agent执行
        yield create_sse_event("agent_executing", {
            "message": "AI智能体开始分析数据",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # 创建依赖项
        deps = Deps(session_id=session_id, tool_calls=[])
        
        # 准备prompt
        df = session.get("df")
        if df is None:
            yield create_sse_event("agent_error", {
                "error": "我还没有收到数据文件。请上传 Excel/CSV (包含 time_s 列与若干温度列），然后告诉我任务，例如：'寻找飞升点'。",
                "timestamp": datetime.datetime.now().isoformat()
            })
            return
            
        cols = list(df.columns)
        filename = session.get("filename", "uploaded")
        
        prompt = (
            f"数据已加载：{filename}, shape={df.shape}。\n"
            f"列名示例：{cols[:12]}。\n"
            f"用户请求：{message}\n"
            "数据已预加载到当前会话，无需再次调用 load_data，请直接分析/检测/可视化。\n"
        )
        
        if await request.is_disconnected():
            return
            
        # 执行agent
        agent_start = time.perf_counter_ns()
        result = await agent.run(prompt, deps=deps, model_settings={"max_tokens": 600, "temperature": 0.2})
        agent_end = time.perf_counter_ns()
        
        if await request.is_disconnected():
            return
        for i, tool_call in enumerate(deps.tool_calls):
            yield create_sse_event("tool_call", {
                "tool_name": tool_call["tool"],
                "args": tool_call["args"],
                "result_preview": tool_call.get("result_preview"),
                "result": tool_call.get("result"),
                "progress": f"{i+1}/{len(deps.tool_calls)}",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            if await request.is_disconnected():
                return
                
            await asyncio.sleep(STREAM_DELAY)
        
        # 准备结果
        artifacts: Dict[str, Any] = {}
        if session.get("last_detection"):
            artifacts["detection"] = session["last_detection"]
        if session.get("last_plot_b64"):
            artifacts["plot_b64"] = session["last_plot_b64"]

        # 发送完成事件
        yield create_sse_event("agent_completed", {
            "message": result.output,
            "tool_calls_count": len(deps.tool_calls),
            "execution_time_ms": (agent_end - agent_start) / 1e6,
            "artifacts": artifacts,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"SSE agent execution error: {e}")
        yield create_sse_event("agent_error", {
            "error": f"处理过程中发生错误: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    finally:
        yield create_sse_event("agent_end", {
            "message": "处理完成",
            "timestamp": datetime.datetime.now().isoformat()
        })

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    request_start = time.perf_counter_ns()
    sess = get_session(session_id)
    deps = Deps(session_id=session_id, tool_calls=[])

    # 非法指令直接返回，直到用户输入符合预期
    if not is_valid_user_message(message):
        return ChatResponse(
            session_id=session_id,
            assistant_message="当前仅支持：画图/可视化、飞升点检测/报告等相关指令，请重新描述需求（示例：'请帮我检测飞升点并画图'）。",
            tool_calls=deps.tool_calls,
            artifacts={}
        )

    try:
        if file is not None:
            start_read = time.perf_counter_ns()
            content = await file.read()
            filename = file.filename or ""
            df = load_uploaded_dataframe(content, filename)
            sess["df"] = df
            sess["filename"] = filename or "uploaded"
            sess["file_id"] = int(sess.get("file_id", 0)) + 1
            sess["last_plot_b64"] = None
            sess["last_detection"] = None

            preview = f"Loaded {filename}, shape={df.shape}, columns={list(df.columns)[:12]}"
            log_tool(deps, "load_data", {"filename": filename}, preview)
            end_read = time.perf_counter_ns()
            logger.info(f"t_read = {(end_read - start_read) / 1e6} ms")
    
    except Exception as e:
        return ChatResponse(
            session_id=session_id,
            assistant_message=str(e),
            tool_calls=deps.tool_calls,
            artifacts={}
        )
    
    if sess["df"] is None:
        return ChatResponse(
            session_id=session_id,
            assistant_message="我还没有收到数据文件。请上传 Excel/CSV (包含 time_s 列与若干温度列），然后告诉我任务，例如：'寻找飞升点'。",
            tool_calls=deps.tool_calls,
            artifacts={}
        )
    
    df = sess["df"]
    cols = list(df.columns)
    filename = sess.get("filename", "uploaded")
    
    prompt = (
        f"数据已加载：{filename}, shape={df.shape}。\n"
        f"列名示例：{cols[:12]}。\n"
        f"用户请求：{message}\n"
        "数据已预加载到当前会话，无需再次调用 load_data，请直接分析/检测/可视化。\n"
    )

    agent_start = time.perf_counter_ns()
    result = await agent.run(prompt, deps=deps, model_settings={"max_tokens": 600, "temperature": 0.2})
    agent_end = time.perf_counter_ns()
    logger.info(f"t_agent = {(agent_end - agent_start) / 1e6} ms")

    artifacts: Dict[str, Any] = {}
    if sess.get("last_detection"):
        artifacts["detection"] = sess["last_detection"]
    if sess.get("last_plot_b64"):
        artifacts["plot_b64"] = sess["last_plot_b64"]

    logger.info(f"t_total = {(time.perf_counter_ns() - request_start) / 1e6} ms, tool_calls={len(deps.tool_calls)}")

    return ChatResponse(
        session_id=session_id,
        assistant_message=result.output,
        tool_calls=deps.tool_calls,
        artifacts=artifacts,
    )

@app.post("/api/chat-sse")
async def chat_sse(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """SSE版本的聊天接口，实时发送agent执行过程"""
    
    file_data = None
    if file is not None:
        try:
            start_read = time.perf_counter_ns()
            content = await file.read()
            filename = file.filename or ""
            df = load_uploaded_dataframe(content, filename)
            sess = get_session(session_id)
            sess["df"] = df
            sess["filename"] = filename or "uploaded"
            sess["file_id"] = int(sess.get("file_id", 0)) + 1
            sess["last_plot_b64"] = None
            sess["last_detection"] = None

            file_data = {
                "filename": filename,
                "shape": df.shape,
                "columns": list(df.columns)[:12]
            }
            
            end_read = time.perf_counter_ns()
            logger.info(f"t_read = {(end_read - start_read) / 1e6} ms")
        except Exception as e:
            logger.error(f"File processing error: {e}")
            file_data = {"error": str(e), "filename": file.filename or ""}
    
    # 返回SSE响应
    return EventSourceResponse(sse_event_generator(request, session_id, message, file_data))

@app.get("/api/health")
def health():
    return {"ok": True}

