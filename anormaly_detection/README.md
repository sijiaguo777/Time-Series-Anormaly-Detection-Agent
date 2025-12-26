# Time-Series Anormaly Detection Agent

## 1. 启动前端
```bash
cd ./frontend
python -m http.server 5173
# 浏览器访问 http://localhost:5173
```

## 2. 启动后端
```bash
cd ./backend
uvicorn main:app --reload 4 --port 8000
```

## 3. 准备本地模型
```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```
