import sys

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    """

    # TODO 1. 添加运行时状态
    # TODO 2. 从状态中获取数据并进行校验
    # TODO 3. 调用模型提取 item_name
    # TODO 4. 回填数据 --- chunk 中添加 item_name
    # TODO 5. 创建 milvus 集合（包含schema结构以及索引）
    # TODO 6. 向量化 并向 Milvus 中插入数据
    # TODO 7. 添加结束状态
    return state