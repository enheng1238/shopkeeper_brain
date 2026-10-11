import sys
import os
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    """

    # TODO 从状态中获取 local_file_path
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.warning("没有指定上传文件路径")
        return state


    # TODO 判断文件类型
    if local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False

        state["md_path"] = local_file_path
        state["pdf_path"] = None
    elif local_file_path.endswith(".pdf"):
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True

        state["md_path"] = None
        state["pdf_path"] = local_file_path
    else:
        logger.warning(f"不支持的文件类型{local_file_path}，退出流程！")
        return state

    logger.warning(f"pdf_path: {state["pdf_path"]}")

    # TODO 补充file_title
    """D:\pycharm_workplace\shopkeeper_brain\doc\hak180产品安全手册.pdf"""
    # state["file_title"] = local_file_path.split("/")[-1].split(".")[0]
    state["file_title"] = os.path.basename(local_file_path).split(".")[0]
    # .name 获取文件名(带后缀) .stem 获取文件名(不带后缀) .parent 获取当前文件所在的文件夹 .suffix获取后缀
    state["file_title"] = Path(local_file_path).stem



    return state