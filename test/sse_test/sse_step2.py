import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/stream")
async def stream():
    async def event_generator():
        for i in range(3):
            yield "event: progress\n"
            yield f"data: 第{i + 1}步处理中\n\n"
            await asyncio.sleep(1)

        yield "event: complete\n"
        yield "data: 全部处理完成\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)