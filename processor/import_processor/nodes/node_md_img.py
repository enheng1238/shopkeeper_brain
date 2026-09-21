import re
import sys
import os
import base64
from mimetypes import guess_type

from inscriptis.model.canvas import prefix
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from requests_toolbelt.auth import guess

from common.config.minio_config import minio_config
from common.config.lm_config import lm_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task
from pathlib import Path
from rich import print as rprint

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    md_path = state.get("md_path")
    if not md_path:
        logger.error("在状态中不存在 md_path 属性")
        raise RuntimeError("在状态中不存在 md_path 属性")

    # 封装Path对象
    md_path_obj = Path(md_path)

    # 判断md文件是否存在
    if not md_path_obj.exists():
        logger.error(f"{md_path_obj}不存在")
        raise RuntimeError(f"{md_path_obj}不存在")

    # 获取文件内容
    md_content = md_path_obj.read_text(encoding="utf-8")

    # 将 md_content 存放在 状态中
    state["md_content"] = md_content

    # 获取存放图片的路径
    md_images_dir_obj = Path(md_path_obj.parent) / "images"


    return md_content, md_path_obj, md_images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj)->list[tuple[str, str, tuple[str, str]]]:
    """
    获取图片信息及上下文
    :param md_content: md文件类型
    :param md_images_dir_obj: 图片所在路径
    :return: list[tuple[str, str, tuple[str, str]]] -> list[(图片名,图片路径,(上文,下文))]
    """
    images_info_list = []
    # 遍历 md_images_dir_obj 目录 获取目录中的每一个图片
    for md_image_file in md_images_dir_obj.iterdir():
        image_name = md_image_file.name
        image_path = str(md_image_file)
        # logger.warning(md_image_file)
        # logger.warning(md_image_file.name)

        # 过滤掉不支持的图片类型
        if not is_supported_image(image_name):
            logger.warning(f"不支持的图片类型{image_name}")
            continue

        # 定义正则表达式

        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # logger.info(f"pattern:{pattern}")
        # 根据正则匹配md所有图片
        search_match = pattern.search(md_content)
        # 判断是否匹配上
        if not search_match:
            logger.warning(f"{image_name}没有被md_content引用，跳过，直接下一次！")
            continue

        # 获取匹配内容的其实下标
        start = search_match.start()
        end = search_match.end()

        # 如果匹配上 获取上下文
        pre_context = md_content[max(0, start-100):start]
        post_context = md_content[end:min(len(md_content), end+100)]
        logger.info(f"image_name:{image_name}, start:{start}, end:{end}, pre_context:{pre_context}, post_context:{post_context}")
        images_info_list.append((image_name,image_path,(pre_context, post_context)))
        logger.info(f'images_info_list:{images_info_list}')


    # 过滤掉不支持的图片类型

    # 定义正则表达式

    # 根据正则匹配md所有的图片

    # 判断是否匹配上

    # 如果匹配上 获取上下文
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(images_info_list)}")

    return images_info_list

# @step_log("step_3_image_summary")
def step_3_image_summary(images_info_list,root_folder):
    """
    调用vm模型，给图片生成摘要
    :param images_info_list: 图片信息列表
    :param stem: md文件名 --- 给提示词文件中的变量用
    :return: 字典 dict[图片名:摘要]
    """
    # 获取模型对象
    image_summary_dict = {}
    vl_model = get_llm_client(model=lm_config.vl_model)
    # 循环images_info_list数据，获取每一张图片的信息
    for image_name, image_path, image_content in images_info_list:
        logger.info(f"image_name:{image_name}, image_path:{image_path}")
        # 封装提示词
        final_prompt = load_prompt(
            name='image_summary',
            root_folder=root_folder,  # 对应{root_folder}
            image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
        )
        # logger.info(f"final_prompt:{final_prompt}")
        image_path = Path(image_path)
        # logger.info(f"image_path:{image_path}")

        """
        guess_type 是 Python 标准库 mimetypes 模块里的函数，用来根据文件名（或 URL）猜出 MIME 类型，比如 image/png、image/jpeg。
        """

        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": final_prompt },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        # logger.info(f"message:{message}")

        # 调用视觉模型获取结果
        chains = vl_model | StrOutputParser()
        image_summary = chains.invoke([message])


        # logger.info(f"image_summary:{image_summary}")

        image_summary_dict[image_name] = image_summary
        # 添加范围限制
        apply_api_rate_limit()
    # 拼接提示返回结果到字典中
    return image_summary_dict

@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(images_info_list, stem) -> dict[str,str]:
    """
    dict[str,str] -> dict[图片名，图片地址]
    :param images_info_list:
    :param stem:
    :return:
    """
    print("step-1")
    # 获取操作minIO的客户端对象
    minio_client = get_minio_client()
    rprint(f"minio_client:{minio_client}")

    minio_bucket = list(minio_client.list_objects(bucket_name=minio_config.bucket_name,recursive=True))
    print(f"minio_bucket:{minio_bucket}, {len(minio_bucket)}")
    for image in minio_bucket:
        logger.warning(f"image:{image}")

    DeleteImageList = [DeleteObject(obj.object_name) for obj in minio_bucket]
    print(f"DeleteImageList:{DeleteImageList}")

    # 先删除掉所有图片 remove_objects
    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=DeleteImageList
    )
    for error in errors:
        print("error occurred when deleting object", error)

    # 重新上传图片 fput_object
    upload_image_dict = {}
    for image_name, image_path, _ in images_info_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=minio_config.minio_img_dir + "/" + stem + "/"+image_name,
                file_path=image_path,
                content_type=guess_type(image_name)[0],
            )
            # http://192.168.10.8:9000/knowledge-base-files/
            url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
            logger.warning(f"image:{image_name}, url:{url}")
            upload_image_dict[image_name] = url
        except:
            logger.warning(f"{image_name}上传失败，跳过，继续上传下一张")

    return upload_image_dict

@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(md_content, image_summary_dict, image_url_dict):
    """

    :param md_content:  文件内容
    :param image_summary_dict: image_name: 摘要
    :param image_url_dict: image_name: 图片url
    :return:
    """
    for image_name, summary in image_summary_dict.items():
        # print(f"===========image_name:{image_name}, summary:{summary}")
        image_url = image_url_dict.get(image_name)
        # print(f"******image_url:{image_url}")
        """
        re.compile() 是 Python 标准库 re 模块中的函数，用于将正则表达式字符串编译成一个可复用的正则表达式对象（Pattern 对象）。
        """
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
              # re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = reg.sub(lambda _: f"![{summary}]({image_url})", md_content)


        logger.error(f"已完成 image_name:{image_name}, image_url:{image_url}, summary:{summary}")
    return md_content

@step_log("step_6_backup_new_md_content")
def step_6_backup_new_md_content(md_content_new, md_path_obj):
    # 创建一个新的路径对象
    """
    with_name() 是 pathlib.Path 的一个方法，用于替换路径中的文件名部分，返回一个新的 Path 对象（不修改原对象）。
    """
    print(f"md_path_obj:{md_path_obj}")
    md_path_obj_new = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")

    # 将新的内容写到新的md对象中
    md_path_obj_new.write_text(md_content_new,encoding="utf-8")
    logger.info(f"md_path_obj_new:内容已完成写入")
    return md_path_obj_new


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    # TODO 1.添加运行时状态
    add_running_task(state.get("task_id"), "node_md_img")
    # TODO 2.从状态中获取数据并进行校验 返回 md_content md_path_obj md_images_dir_obj
    md_content, md_path_obj, md_images_dir_obj = step_1_validate_and_get_data(state)

    # 判断是否有图片，如果md没有图片，后边所有的操作都不需要做了
    # is_dir(): 是否是目录
    # iterdir(): 是 pathlib.Path 中用于遍历目录内容的方法。它会返回一个迭代器，逐个产出目录下的所有文件和子目录（不包括 . 和 ..）
    if not md_images_dir_obj.is_dir() or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"{md_images_dir_obj}里没有图片内容，无需处理，直接跳出。")
        return state

    # TODO 3.获取图片信息以及上下文
    # list[tuple[str, str, tuple[str, str]]] -> list[tuple[图片名, 图片地址, tuple[上文, 下文]]]
    images_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)
    # TODO 4.调用视觉模型 生成图片摘要
    # logger.info(f"md_path_obj:{md_path_obj}---{md_path_obj.stem}")
    image_summary_dict = step_3_image_summary(images_info_list, md_path_obj.stem)
    logger.warning(f"image_summary_dict: {image_summary_dict}")
    # TODO 5.将图片上传到 minio 服务器
    # 5. 将图片信息传递到minio的服务器,并获取图片和对应的网络地址
    # {image_name:http...}
    image_url_dict: dict[str, str] = step_4_upload_images_get_url(images_info_list, md_path_obj.stem)
    # step_4_upload_images_get_url(images_info_list, md_path_obj.stem)
    # TODO 6.替换md_content中图片内容
    # 6. 原md_content内图片 ![](./xx) -> ![图片的描述](图片的网络地址)
    # image_summary_dict -> image_name : 图片的描述  image_url_dict  image_name : url
    md_content_new: str = step_5_md_content_image_replace(md_content, image_summary_dict, image_url_dict)

    # TODO 7.将新的md文件进行磁盘存储
    # 7. 将md_content_new进行磁盘化备份(1.新和旧进行对比测试 2.后续节点md_content内容丢失->再次读取)
    md_path_obj_new: Path = step_6_backup_new_md_content(md_content_new, md_path_obj)

    # 更新状态
    state["md_path"] = str(md_path_obj_new)
    state["md_content"] = md_content_new

    # TODO 8.添加完成状态
    add_done_task(state.get("task_id"), "node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")