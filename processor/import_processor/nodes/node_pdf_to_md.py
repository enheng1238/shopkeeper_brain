import shutil
import sys
import time

from envs.nlp.Lib.urllib import request
import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT, Path
from utils.task_utils import add_running_task, add_done_task
import os

@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState):
    # 从状态中获取 pdf_path 和 local_dir
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 判断是否为空
    # 1.如果 pdf_path为空，抛出异常
    if not pdf_path:
        logger.error("状态中没有 pdf_path, 流程终止")
        raise ValueError("状态中没有 pdf_path, 流程终止")

    # 2.如果local_dir为空,指定默认输出目录
    if not local_dir:
        logger.warning("状态中没有 local_dir,指定默认输出目录")
        local_dir = PROJECT_ROOT / "output"

    # 转换为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_path = Path(local_dir)

    # 判断文件是否存在
    # 1.如果 pdf_path_obj 不存在或者不是文件 抛出异常
    if not pdf_path_obj.is_file():
        logger.error(f"{pdf_path_obj}不存在或者不是文件")
        raise ValueError(f"{pdf_path_obj}不存在或者不是文件")

    # 2.如果local_dir_path 不存在或者不是目录，创建新的目录
    if not local_dir_path.is_dir():
        logger.warning(f"{local_dir_path}不存在或者不是文件，创建新目录")
        local_dir_path.mkdir(parents=True, exist_ok=True)


    return pdf_path_obj, local_dir_path

@step_log("step_2_validate_paths")
def step_2_validate_paths(pdf_path_obj):
    # 1. ===========校验minerU核心参数===========
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"

    # 2. ===========申请上传地址和批量ID===========
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        "model_version": "vlm"
    }

    # 发送请求-申请地址
    response = requests.post(url, headers=header, json=data)
    logger.info(f'response:{response}')

    if response.status_code == 200:
        result = response.json()
    # 获取请求响应的状态码
    http_status_code = response.status_code
    if http_status_code != 200:
        logger.error("向MinerU服务器发送请求获取上传文件地址失败")
        raise RuntimeError("向MinerU服务器发送请求获取上传文件地址失败")

    """
        获取响应内容: 
            response.json() 以json形式获取数据（字典）
            response.text 以字符串形式获取数据
            response.content 以二进制形式获取数据

    """

    response_dict = response.json()
    response_dict_code = response_dict.get("code")
    logger.info(f"response_dict_code:{response_dict_code}")
    response_dict_msg = response_dict.get("msg")
    response_dict_data = response_dict.get("data")
    batch_id = response_dict_data.get("batch_id")
    file_urls = response_dict_data.get("file_urls")

    if response_dict_code != 0:
        logger.error(f"申请地址请求成功,但是业务处理失败，错误码:{response_dict_code},错误信息:{response_dict_msg}")
        raise RuntimeError(
            f"申请地址请求成功,但是业务处理失败，错误码:{response_dict_code},错误信息:{response_dict_msg}")

    if not batch_id:
        logger.error("没有返回batch_id")
        raise RuntimeError("没有返回batch_id")

    if not file_urls:
        logger.error("没有返回file_urls")
        raise RuntimeError("没有file_urls")

    upload_file_url = file_urls[0]

    logger.info(f"===========申请上传地址成功:{upload_file_url}")

    # if response.status_code == 200:
    #     result = response.json()
    #     print('response success. result:{}'.format(result))
    #     if result["code"] == 0:
    #         batch_id = result["data"]["batch_id"]
    #         urls = result["data"]["file_urls"]
    #         logger.info(f"==============urls:{urls}")
    #         print('batch_id:{},urls:{}'.format(batch_id, urls))
    #         for i in range(0, len(urls)):
    #             with open(upload_file_url, 'rb') as f:
    #                 res_upload = requests.put(urls[i], data=f)
    #                 if res_upload.status_code == 200:
    #                     print(f"{urls[i]} upload success")
    #                 else:
    #                     print(f"{urls[i]} upload failed")
    #     else:
    #         print('apply upload url failed,reason:{}'.format(result["msg"]))
    # else:
    #     print('response not success. status:{} ,result:{}'.format(response.status_code, response))

    # 3. ===========向指定地址进行文件上传===========
    """
        向指定的位置上传文件
        pdf_path_obj.read_text() 从文件中读取字符数据
        pdf_path_obj.read_bytes() 从文件中读取字节数据
        pdf_path_obj.write_text() 向文件中写入字符数据
        pdf_path_obj.write_bytes() 向文件读取字节数据
    """
    file_update_data = pdf_path_obj.read_bytes()
    # 注意：如果直接使用 requests.put 进行文件上传，有很大的概率会失败
    # 如果有代理，会在请求头中携带额外信息，导致存储文件的服务器不能访问 而失败
    # res_upload = requests.put(upload_file_url, data=file_update_data)
    with requests.session() as session:
        # 纯净版的请求头，不随意携带代理的参数
        session.trust_env = False
        upload_res = session.put(upload_file_url, data=file_update_data)
        logger.info(f"=======session======,{session}")
        logger.info(f"upload_res: {upload_res}")

        if upload_res.status_code != 200:
            logger.error(f"文件上传失败，状态码: {upload_res.status_code}")
            raise RuntimeError(f"文件上传失败，状态码: {upload_res.status_code}")

        logger.info("文件上传成功")

    # logger.info(f"file_update_data:{file_update_data}")

    # 4. =========== 轮询获取服务器的解析响应结果 ===========
    poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    # 超市时间
    timeout = 600
    # 每隔多久轮询一次
    interval_time = 3
    # 轮询开始时间
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            logger.error("轮询获取解析结果超时")
            raise TimeoutError("轮询获取解析结果超时")

        try:
            poll_response = requests.get(poll_url, headers=header)
            logger.info(f"res.status_code:{poll_response.status_code}")
        except Exception as e:
            logger.warning(f"请求出现异常: {e}，稍后重试")
            time.sleep(interval_time)
            continue

        # 获取请求状态码
        poll_response_status_code = poll_response.status_code
        logger.info(f"poll_response_status_code:{poll_response_status_code}")

        # 判断是不是服务器内部问题
        if poll_response_status_code != 200:
            if 500 <= poll_response_status_code < 600:
                # 如果是服务器报错  给机会继续重试
                logger.warning(f"可有修复的网络异常,状态码为:{poll_response_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")


        # 判断业务接口返回的内容
        poll_response_dict = poll_response.json()
        logger.info(f"poll_response:{poll_response}")
        poll_response_code = poll_response_dict["code"]
        poll_response_msg = poll_response_dict["msg"]


        if poll_response_code != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_code},失败信息:{poll_response_msg}")

        poll_response_data = poll_response_dict["data"]
        poll_data_batch_id = poll_response_data["batch_id"]
        logger.info(f"poll_response_data:{poll_response_data}")
        poll_extract_result = poll_response_data["extract_result"][0]

        if not poll_extract_result:
            logger.warning("extract_result 为空，任务可能还在排队中...")
            time.sleep(interval_time)
            continue

        poll_extract_state= poll_extract_result["state"]
        logger.info(f"poll_extract_result:{poll_extract_result}")

        """
        任务处理状态，完成:done，waiting-file: 等待文件上传排队提交解析任务中，pending: 排队中，running: 正在解析，failed：解析失败，converting：格式转换中
        """
        if poll_extract_state == 'done':
            extract_result_url = poll_extract_result['full_zip_url']
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
            # 5. 返回压缩地址即可
            logger.info(f">>>获取mineru服务器解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif poll_extract_state == 'failed':
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{poll_extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{poll_extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{poll_extract_result}!")
            time.sleep(interval_time)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_path, stem):
    logger.info(f"zip_url:{zip_url}")
    logger.info(f"local_dir_path:{local_dir_path}")

    # ------ 从服务器下载文件 -------
    zip_response = requests.get(zip_url, timeout=60)
    if zip_response.status_code != 200:
        logger.error(f"zip_response:{zip_response.status_code}")
        raise RuntimeError(f"zip_response:{zip_response.status_code}")

    # ------ 解压到指定目录 ------
    extracted_file_Path:Path = Path(local_dir_path) / stem
    if extracted_file_Path.exists():
        shutil.rmtree(extracted_file_Path) # 删除已存在目录
    extracted_file_Path.mkdir(parents=True, exist_ok=True)
    md_zip_path_obj: Path = extracted_file_Path / f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(zip_response.content)

    # 解压
    shutil.unpack_archive(md_zip_path_obj, extracted_file_Path)

    # ------ 查找解压后的所有md文件 -> list ------
    md_file_list = list(extracted_file_Path.rglob("*md"))
    logger.info(f"md_file_list:{md_file_list}")

    if not md_file_list:
        logger.error(f"文件解压失败,在{extracted_file_Path}里没有找到任何md文件")
        raise RuntimeError(f"文件解压失败,在{extracted_file_Path}里没有找到任何md文件")

    # ------ 按优先级选择 md 文件 ------
    # 优先级1: stem.md
    for md_file in md_file_list:
        if md_file.stem == stem:
            logger.info(f"md_file:{md_file}解压成功")
            return md_file
    # 优先级2: full.md
    target_md_obj = None
    for md_file in md_file_list:
        if md_file.name.lower() == 'full.md':
            target_md_obj = md_file
            logger.info(f"target_md_obj:{target_md_obj}解压成功")
            break
    # 优先级3: list[0]
    if not target_md_obj:
        target_md_obj = md_file_list[0]
    # 改名: stem.md
    # target_md_obj.rename(stem + ".md") # 当前路径

    # ---------------------- 5. 重命名为统一规范名称 ----------------------
    # 将选中的 MD 重命名为 {stem}.md（和 PDF 同名）
    # with_name() :只负责生成新路径对象，不会修改磁盘文件
    # rename() 才是真正磁盘重命名操作。  rename会删除源文件，创建新的文件，所以需要嵌套一层
    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    logger.info(f"文件解压成功: {final_md_path_obj}")
    return final_md_path_obj

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    # TODO 1.添加运行时列表
    add_running_task(state.get("task_id"), "node_pdf_to_md")
    # TODO 2.从状态中获取数据并进行校验
    pdf_path_obj, local_dir_path = step_1_validate_paths(state)
    # TODO 3.上传文件并解析
    zip_url = step_2_validate_paths(pdf_path_obj)
    logger.info(f"zip_url:{zip_url}")
    # TODO 4.下载并解压
    md_path = step_3_download_and_extract(zip_url, local_dir_path, pdf_path_obj.stem)
    # TODO 5. 更新状态
    state['md_path'] = md_path
    # TODO 6.添加完成列表
    add_done_task(state.get("task_id"), "node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    logger.info(f"test_pdf_name:{test_pdf_name}")
    logger.info(f"test_pdf_path:{test_pdf_path}")

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")