import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 跨域配置（开发环境，生产不要用*）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储每个会话的异步队列
task_queues = {}

async def long_task(session_id: str):
    queue = asyncio.Queue()
    task_queues[session_id] = queue
    try:
        # 模拟5步任务，每秒推送一条消息
        for i in range(5):
            msg = f"会话{session_id}处理结果{i + 1}"
            await queue.put(msg)
            await asyncio.sleep(1)
        # 放入结束标记
        await queue.put(None)
    finally:
        # 任务结束清理队列，防止内存泄露
        if session_id in task_queues:
            del task_queues[session_id]


# 提交任务接口
@app.get("/submit/{session_id}")
async def submit_task(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(long_task, session_id)
    return {"message": "任务已启动", "session_id": session_id}


# SSE流式接口
@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    async def event_generator():
        # 等待会话队列创建
        while session_id not in task_queues:
            await asyncio.sleep(0.1)
        queue = task_queues[session_id]
        while True:
            msg = await queue.get()
            if msg is None:
                # 推送自定义done事件，通知前端任务正常完成
                yield "event: done\n"
                yield "data: task finished\n\n"
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)