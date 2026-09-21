import json
import re
import sys
import os
from pathlib import Path
from typing import Any

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ====================== 全局配置（可根据模型调整）======================
# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400

@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state):
    # 从状态中获取数据
    md_path = state.get("md_path")
    file_title = state.get("file_title")
    md_content = state.get("md_content")

    # state.md_path: D:\pycharm_workplace\shopkeeper_brain\output\hak180产品安全手册\hak180产品安全手册_new.md
    # logger.warning(f"md_path: {md_path}")
    # logger.warning(f"file_title: {file_title}") # hak180产品安全手册
    # logger.warning(f"md_content: {md_content}") # md文档内容 - 新

    # 检验是否存在
    if not md_content:
        if (not md_path) or not Path(md_path).is_file():
            # md_path为空,或者是不存在(不是文件)
            logger.error(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
            raise ValueError(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content
        # logger.warning(f"====md_content-2: {md_content}")

    if not file_title:
        file_title = Path(md_path).stem
        state["file_title"] = file_title
        # logger.warning(f"状态中没有====file_title-2: {file_title}")

        # 注意：不同的操作系统，对于md文档中换行表现是不一样的，这里做一个统一 换行就用 \n
        # 最终结果：所有换行符都变成 \n。
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content,file_title,md_path

@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content, file_title):
    # 存储最终返回的切块列表
    chunks: list[dict[str, Any]] = []

    # 当前激活的完整嵌套标题 (例如: # 标题1_## 标题2)
    current_title: str|None = None

    # chunk正文缓存区 存放当前标题下的正文与代码行,不包含标题行本身
    current_content_lines: list[str] = []

    # 孤儿数据缓存区 第一个标题出现前的所有文本行
    orphan_lines: list[str] = []

    # 标记位:记录开头累积的孤儿数据是否已经被成功合并到了首个chunk中
    has_flushed_first_chunk:bool = False

    # 状态标记 标记当前行是否处于代码块内部(```或者~~~)
    is_cool:bool = False

    # 多级标题栈:用于标记 H1~H6 例如: ["H1","H2",None, "H4"]
    heading_stack:list[str] = []

    # 对md_cosntent进行切割
    document_lines = md_content.splitlines()

    # 匹配标题的正则表达式 \s: 空格 .：任意内容 +：1~多个
    title_reg = re.compile(r"^\s*#{1,6}\s.+")

    """
    ~~~ 是 Markdown 中代码块的另一种标记符号，和三个反引号 ``` 的作用完全一样。
    """

    # 对整个文档所有行进行遍历
    for line in document_lines:
        # 取出前后空格
        line_strip = line.strip()

        if not line_strip:
            # 如果是空行，直接跳过处理逻辑
            continue

        # 判断是不是代码标记
        if line.startswith("```") or line.startswith("~~~"):
            is_cool = not is_cool # 状态取反 True 进入代码块 False 出来代码块
            if not current_title: # 孤儿数据
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        if is_cool: # 进入代码块
            if not current_title: # 孤儿数据
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        # 判断是不是标题
        if title_reg.match(line_strip): # 标题
            # 如果是标题 -- 标题的处理逻辑
            # 获取当前标题级别
            heading_level = len(line_strip) - len(line_strip.lstrip("#")) # (拿到有几个#) -> 标题层级
            if current_title and len(current_content_lines) >0: # 有标题并且标题下有正文

                """形成chunk"""
                # has_flushed_first_chunk 标记位: 记录开头累积的孤儿数据是否已经被成功合并到了首个chunk中
                if not has_flushed_first_chunk and orphan_lines:
                    current_content_lines = orphan_lines + current_content_lines
                    has_flushed_first_chunk = True
                    orphan_lines = []
                full_content = f"{current_title}\n "+" \n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title,
                })

                # 清空正文缓冲区
                current_content_lines = []

            # 维护多级标题栈
            while len(heading_stack) < heading_level:
                heading_stack.append(None)
            heading_stack = heading_stack[:heading_level]
            heading_stack[heading_level-1] = line_strip
            current_title = "_".join([h for h in heading_stack if h]) # 拼接 空值不要 例如: # 标题1_## 标题2
            continue
        else: # 正文
            if not current_title: # 孤儿数据
                orphan_lines.append(line)
            else: # 正文
                current_content_lines.append(line)

    # 循环结束之后，对最后一个标题下的内容进行处理
    if current_title and len(current_content_lines) >0 or (not has_flushed_first_chunk and len(orphan_lines) >0):

        """下一个标题出现，对上一个标题做结算"""
        if not has_flushed_first_chunk and len(orphan_lines): # 最后一部分是孤儿数据 且孤儿数据未被合并
            current_content_lines = orphan_lines + current_content_lines
            has_flushed_first_chunk = True
            orphan_lines = []
        full_content = f"{current_title}\n "+" \n".join(current_content_lines) # 包括三部分内容：1.标题 + 2.孤儿数据 + 3.正文数据
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title,
        })
    elif not current_title and len(orphan_lines):
        chunks.append({
            "title": current_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title,
        })
    logger.warning(f"经过首次切割后的chunk数量 len{len(chunks)}")

    return chunks


def _split_chunk_content(chunk) ->list[dict[str, Any]]:
    sub_chunks:list[dict[str, Any]] = []
    # content: ## 警告\n 不遵守说明和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟、发热、爆炸、
    content = chunk.get("content")
    # 去掉 ## 标题\n
    prefix = chunk.get("title") + "\n"
    deal_content = content[len(prefix):]

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    for index, text in enumerate(spliter.split_text(deal_content), start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "title": f"{chunk.get('title')}_{index}",
            "part": index,
            "content": prefix + text
        })


    return sub_chunks


def _merge_chunk_content(refine_chunks):
    merge_refine_chunk:list[str, Any] = []

    # 核心思路：指针
    # 每次循环用基准chunk和next_chunk进行合并（满足条件） base_chunk = next_chunk
    # 定义一个基准chunk，用于和下一个chunk进行比较
    base_chunk: dict[str, Any] = None

    # 对精细化切分后的所有chunk进行遍历
    for next_chunk in refine_chunks:
        if base_chunk is None:
            base_chunk = next_chunk
            continue

        # 判断 base_chunk 是否小于 CHUNK_SIZE 400
        is_too_long = len(base_chunk.get("content")) > CHUNK_SIZE # CHUNK_SIZE：400
        if not is_too_long:
            # 2.判断是不是在同一个标题下 parent_title以
            is_same_parent_title = (base_chunk.get("parent_title") == next_chunk.get("parent_title"))

            if is_same_parent_title:
                # 3.判断 base_chunk + next_chunk内容 是否超过 CHUNK_MAX_SIZE (1000)

                base_content: str = base_chunk.get("content")
                """
                + 1	多跳过 1 个字符（通常是标题后的换行符 \n）
                这行代码的作用是：从 next_chunk 的 content 中，截掉开头的标题部分，只保留正文内容。
                """
                next_content: str = next_chunk.get("content")[len(next_chunk.get("title")) + 1:]
                is_merged_too_long = len(base_content) + len(next_content)
                if is_merged_too_long <= CHUNK_MAX_SIZE:
                    base_chunk['content'] = base_content + "\n" + next_content
                else:
                    # base_chunk + next_chunk内容 是否超过 CHUNK_MAX_SIZE (1000),不需要合并
                    merge_refine_chunk.append(base_chunk)
                    base_chunk = next_chunk
            else:
                # 不在同一个标题下，不需要合并
                merge_refine_chunk.append(base_chunk)
                base_chunk = next_chunk
        else:
            # 不超长不需要合并
            merge_refine_chunk.append(base_chunk)
            base_chunk = next_chunk


        if base_chunk:
            merge_refine_chunk.append(base_chunk)
    logger.warning(f"经过merge的chunk数量: {len(merge_refine_chunk)}")
    return merge_refine_chunk


@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(title_chunks)->list[dict[str, Any]]:
    """
    精细切割!!
    :param chunks:
    :return:
    """
    refine_chunks: list[dict[str, Any]] = []
    for chunk in title_chunks:
        if len(chunk.get("content")) > CHUNK_SIZE: # 进行精细化切分
            """
            append: 追加一个元素
            extend: 追加一个列表
            _split_chunk_content(chunk): 的返回值还是 list[dict[str, Any]]
            """
            refine_chunks.extend(_split_chunk_content(chunk))
        else: # 直接返回
            refine_chunks.append(chunk)
        logger.warning(f"经过精细化切割后的 refine_chunks 数量 len:{len(refine_chunks)}")

        # 如果经过精细化细分之后，有些chunk的内容长度 < CHUNK_MIN (400),那么对其进行合并，但是要求合并后大小不能超过 CHUNK_MAX_SIZE (1000)
        refine_chunks = _merge_chunk_content(refine_chunks)
    return refine_chunks

@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(refine_chunks):
    for chunk in refine_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk["title"]
        if "part" not in chunk:
            chunk["part"] = 1
    logger.info(f"完成 chunks 的元素补充，所有属性都完整")


def step_5_backup_chunks_json(refine_chunks, md_path):
    # md_path：D:\\pycharm_workplace\\shopkeeper_brain\\output\\hak180产品安全手册\\hak180产品安全手册_new.md
    md_path:Path = Path(md_path)
    md_path_obj:Path = Path(md_path).parent / f"{md_path.stem}.json"

    # json.dumps(refine_chunks,ensure_ascii=False,indent=4) # indent：缩进
    md_path_obj.write_text(json.dumps(refine_chunks,ensure_ascii=False,indent=4),encoding="utf-8")

    logger.info(f"已完成 refine_chunks 备份，存放位置: {md_path_obj}")



@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    """
    # TODO 1.添加到运行时列表
    add_running_task(state.get("task_id"), "node_document_split")
    # TODO 2.从状态中获取数据并进行校验
    md_content, file_title, md_path = step_1_validate_get_data(state)
    # TODO 3.根据标题进行语义切割
    # 3. 根据语义(标题进行内容的切割)
    """
    根据语义切割
    list[{"title":"", "content":"", "file_title":""}]
    title: md文档的标题(1~6级标题)
    file_title: 文件目录
    """
    title_chunks: list[dict[str, Any]] = step_2_split_document_by_title(md_content, file_title)
    # for chunk in chunks:
    #     logger.success(f"chunk:{chunks}")

    # TODO 4.精细化切割
    # 4. 进行chunks的精细切割(切+和)
    refine_chunks: list[dict[str, Any]] = step_3_refine_split_and_merge_chunks(title_chunks)
    # TODO 5.补充元数据信息(parent_title,part)
    step_4_padding_chunks_metadata(refine_chunks)

    # for refine_chunk in refine_chunks:
    #     logger.error(f'*********refine_chunks:{refine_chunk}')

    # TODO 6.将切割后的chunk持久化到磁盘上
    # 6. 备份chunks的内容
    # chunks => [{},{}] -> json字符串 -> 文件 -> 文件名.json
    step_5_backup_chunks_json(refine_chunks, md_path)

    # TODO 7.更新状态
    state['chunks'] = refine_chunks
    # TODO 8.添加到已完成列表
    add_done_task(state.get("task_id"), "node_document_split")
    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    from processor.import_processor.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    # test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
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
            "md_content": "",
            "file_title": "hak180产品安全手册.md", # "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        # logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.warning(f"final_state:{final_state}")
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")