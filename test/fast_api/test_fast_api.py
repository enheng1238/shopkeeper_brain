"""
该案例演示了 fastapi 的使用
fastapi:
    概念： web开发框架，用于快速的开发web服务（接口）
    如何接收客户端传递的参数
    如何向客户端返回数据

"""
import asyncio
import time

from envs.nlp_python311.Lib.mimetypes import guess_type
from fastapi import FastAPI, UploadFile, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse, PlainTextResponse, RedirectResponse, StreamingResponse

from utils.path_util import PROJECT_ROOT

app = FastAPI()

@app.get("/cors")
def welcome():
    return {"message":"跨域处理成功"}

# CORS跨域问题：解决方案一（允许跨域）：添加 CORS 中间件
# 允许前端页面（如运行在 5500 端口）访问后端 API（8000 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源（生产环境建议指定具体域名）
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法
    allow_headers=["*"], # 允许所有请求头
)

# CORS跨域问题：解决方案2：（同源） http://127.0.0.1:8000/index.html - 直接请求该地址
@app.get("/index.html")
def index():
    return FileResponse(PROJECT_ROOT /"test"/ "fast_api" / "index.html", media_type="text/html")

"""最基本的"""
# @app.get("/hello")
# def say_hello():
#     return {"hello":"world"}

"""
请求参数:
http://127.0.0.1:8000/login?username=%E5%BC%A0%E4%B8%89&password=12345
"""
@app.get("/login")
def login(username: str, password: str):
    return {"username":username,"password":password}


"""
路径参数:
http://127.0.0.1:8000/search/909090
    适合 restful 风格的开发
    @app.get("/search/{std}")
    @app.delete("/search/{std}")
    @app.put("/search/{std}")
    @app.post("/search/{std}")
"""

@app.get("/search/{std}")
def search(std: int):
    return {"学生std":std}

"""
请求体参数:
"""
class User(BaseModel):
    username: str
    password: str

@app.post("login2")
def login2(User: User):
    return {"username":User.username,"password":User.password}

# JSONResponse
@app.post("login3")
def login3(User: User):
    return JSONResponse({"username": User.username, "password": User.password})

# FILEResponse
@app.get("/download")
def download():
    pdf_path = r"D:\pycharm_workplace\shopkeeper_brain\doc\hak180产品安全手册.pdf"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=pdf_path,
        filename="hak180产品安全手册.pdf",
        media_type=guess_type("hak180产品安全手册.pdf")[0]
    )

# 纯文本
@app.get("/getText")
def getText():
    return PlainTextResponse(content="这是纯文本响应~~~")

# 重定向
@app.get("/old-path")
def oldPath():
    # 重定向到 /new-path，状态码 307 表示临时重定向
    return RedirectResponse(url="/new-path", status_code=307)

@app.get("/new-path")
def newPath():
    return {"message": "这是新接口"}

async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流

@app.get("/stream")
async def stream_response():
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/upload",summary="单个文件上传接口")
async def upload(file: UploadFile):
    try:
        # 1. 校验文件类型（示例：只允许上传图片）
        ALLOWED_TYPES = ["image/jpeg", "image/png", "image/gif"]
        if file.content_type not in ALLOWED_TYPES:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=f"仅支持上传{ALLOWED_TYPES}类型的文件，当前文件类型：{file.content_type}"
            )
        file_path = PROJECT_ROOT / "output" / file.filename
        print(f"file_path: {file_path}")

        CHUNK_SIZE = 1024 * 1024  # 1MB分块
        with open(file_path, "wb") as f:
            # 分块读取大文件（避免内存溢出）
            while True:
                chunk = await file.read(CHUNK_SIZE)  # 每次读1MB
                if not chunk:
                    break
                f.write(chunk)
            """
            await的必要性：file.read()是异步函数，必须用await才能拿到文件内容，否则程序报错；
            分块读取的核心：给read()传一个字节数（如 1MB），循环读取 + 写入，避免大文件占满内存；
            用法选择：小文件可以用await file.read()一次性读取，大文件必须分块读取；
            """
        return JSONResponse(
            status_code=200,
            content={
                "code": 0,
                "msg": "文件上传成功",
                "data": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_size": f"{file.size} 字节",  # 文件大小
                }
            }
        )
    except Exception as e:
        # 异常捕获：返回友好的错误信息
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败：{str(e)}"
        )


# 后台任务
# 定义一个模拟的耗时任务
def write_log(message: str):
    time.sleep(2) # 模拟耗时 2 秒
    with open("log.txt", "a") as log:
        log.write(message + "\n")

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    # 1. 添加任务到后台队列
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    # 2. 立即返回响应给用户，不需要等待 write_log 执行完毕
    return {"message": "Notification sent in the background"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)









