# Minimal Time-Series Anomaly Detection Agent - Copilot Instructions

## Architecture Overview
This is a **PydanticAI + FastAPI** chat application for time-series anomaly detection. The system uses an LLM agent to intelligently route user requests into specialized tool calls.

**Key components:**
- **Backend** (`backend/main.py`): FastAPI server with PydanticAI agent and EWMA anomaly detection
- **Frontend** (`frontend/index.html`): Single-file Chat UI showing messages, tool calls (with args/results), and plots
- **Session model**: In-memory dict-based sessions keyed by `session_id` (UUID), storing: dataframe, filename, latest plots, detection results

## Data Flow
1. User uploads Excel/CSV via form → frontend sends base64-encoded file + message to `/api/chat`
2. Backend `load_data()` tool decodes and reads file (pandas), stores in session state
3. Agent processes user intent (via LLM) and chains tools: `visualize` → `detect` → `summarize`
4. Each tool logs to `deps.tool_calls` for frontend debugging
5. Response includes: assistant message, tool call logs, and plot/detection artifacts (base64 PNG)
