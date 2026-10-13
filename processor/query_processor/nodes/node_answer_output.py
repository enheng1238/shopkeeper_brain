import re
import time
import sys
from typing import Any

from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_running_task, add_done_task

@step_log("answer_exists_in_state")
def answer_exists_in_state(state):
    # 从状态中获取 答案
    answer = state.get("answer")
    has_answer: bool = True if answer else False
    return has_answer

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取数据
    reranked_docs = state.get("reranked_docs")
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    is_stream = state.get("is_stream", False)

    if (not reranked_docs) or (not session_id) or (not item_names) or (not rewritten_query):
        logger.error("核心参数为空,业务无法继续,提前终止!")
        raise ValueError("核心参数为空,业务无法继续,提前终止!")

    return reranked_docs, session_id, item_names, rewritten_query, is_stream

@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id):
    history_list:list[dict[str, Any]] = get_recent_messages(session_id)
    logger.warning(f"根据{session_id}查询到的聊天记录{len(history_list)}")

    # 过滤有效历史记录
    history_list = [history for history in history_list if history.get("item_names")]
    logger.warning(f"根据{session_id}过滤后的有效的的聊天记录{len(history_list)}")

    return history_list

@step_log("step_3_create_answer_prompt")
def step_3_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query):
    """
    【参考内容】
    {context}

    【历史对话】
    {history}

    【相关商品/实体】
    {item_names}

    【用户问题】
    {question}

    """
    # 拼接 context 从reranked_docs获取 [{"title":"", "text":"", "score":"", "type":"", "url":""}, {}, {}]
    # f"标题:xxx,数据类源:{ '联网搜索' or '向量数据库'}, 置信度:xxx,内容:xxx \n"
    context: str = ""
    for chunk in reranked_docs:
        context += f"标题:{chunk.get('title')},数据类源:{ '联网搜索' if chunk.get('type') == 'web' else '向量数据库'}, 置信度:{chunk.get('score')},内容:{chunk.get('text')} \n"

    # 拼接history
    history_text: str = None
    if history_list:
        history_text_list: list[str] = []
        for index, history_item in enumerate(history_list):
            if history_item.get("role") == "user":
                # 用户
                history_text_list.append(
                    f"编号:{index},用户提问,原始问题:{history_item.get('text')},重写后的问题:{history_item.get('rewritten_query')},关联item_name：{','.join(history_item.get('item_names'))}")
            else:
                # 助手
                # 注意：助手回答内容可能比较多，为了避免内容过多可以使用如下方式进行优化
                # 方式一：对历史记录中的回答内容进行截取
                # 方式二： 调整提示词模板的内容
                history_text_list.append(
                    f"编号:{index},助手回答,重写问题:{history_item.get('rewritten_query')},回答结果:{history_item.get('text')[:50]},,关联item_name:{','.join(history_item.get('item_names'))}")
        history_text = '\n'.join(history_text_list)
    else:
        history_text = "没有有效的历史聊天记录!"

    # 拼接 item_names
    item_names += ','.join(item_names)

    answer_prompt = load_prompt(name="answer_out", context=context, history=history_text, item_names=item_names, question=rewritten_query)

    return answer_prompt

@step_log("step_4_call_llm_create_answer")
def step_4_call_llm_create_answer(answer_prompt, is_stream, session_id):
    answer:str = ""

    # 获取模型客户端
    llm_model = get_llm_client()

    # 判断是不是流式处理
    if is_stream:
        # 流式 流式处理，需要调用模型的stream方法来获取结果，并将结果通过 SSE 方式，推送到客户端
        stream_res = llm_model.stream(answer_prompt)
        for chunk in stream_res:
            chunk_str = chunk.content
            # 将流式输出的内容，推到队列
            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={"delta": chunk_str}
            )
            answer += chunk_str
    else:
        # 非流式
        response = llm_model.invoke(answer_prompt)
        answer = response.content
    return answer

@step_log("step_5_extract_chunk_and_url_image")
def step_5_extract_chunk_and_url_image(reranked_docs):
    # ![](images/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)
    # 从 rerank 精排过滤后的内容中提取图片出来
    image_urls: list[str] = []

    # 定义匹配图片的正则表达式
    reps = re.compile(r"\!\[.*?\]\((.*?)\)")

    # 对结果进行遍历
    for chunk in reranked_docs:
        url: str = chunk.get("url")
        text = chunk.get("text")

        if url:
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                image_urls.append(url)

        if text:
            image_url_list:list[str] = reps.findall(text)
            if image_url_list:
                image_urls.extend(image_url_list)

    return image_urls

@step_log("step_6_save_assistant_message")
def step_6_save_assistant_message(state):
    # 写入/更新单条会话记录到MongoDB
    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names"),
        image_urls=state.get("image_urls"),
    )


@node_log("node_answer_output")
def node_answer_output(state:QueryGraphState):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    # logger.info("---node_answer_output 节点处理开始---")
    # add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #
    # session_id = state["session_id"]
    # is_stream = state.get("is_stream", True)
    # base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。"
    # final_text = ""
    #
    # if is_stream:
    #     for ch in base_answer:
    #         final_text += ch
    #         push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
    #         time.sleep(0.03)
    #
    #     image_urls = ["https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png", "https://atguigu.com/images/index_new/logo.png"]
    #     logger.info(f"流式输出完成，总长度: {len(final_text)}")
    # else:
    #     final_text = base_answer
    #
    #
    #
    # # 将系统消息保存到 mongodb
    # save_chat_message(
    #     session_id=state["session_id"],
    #     role="assistant",
    #     text=final_text,
    #     rewritten_query="这是重写的问题",
    #     item_names=["商品A","商品B"],
    #     image_urls=image_urls
    # )
    # add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    # logger.info("---node_answer_output 节点处理结束---")



    # TODO 1. 添加节点到运行中列表
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    # TODO 2. 从状态中获取答案 | 获取和判断answer是否存在
    has_answer: bool = answer_exists_in_state(state)
    # 不存在,模型回答answer以及提取image_urls的过程
    if not has_answer:
        # TODO 3. 从状态中获取数据 session_id、reranked_docs、item_names、rewritten_query、is_stream, 并进行校验
        reranked_docs, session_id, item_names, rewritten_query, is_stream = step_1_validate_and_get_data(state)
        # TODO 4. 根据 session_id 获取历史记录
        history_list = step_2_get_history_by_session_id(session_id)
        # TODO 5. 拼接提示词
        answer_prompt: str = step_3_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query)
        # TODO 6. 调用模型生成答案 流式 | 非流式 注意：如果是流式处理（is_stream=True） 需要调用 push_to_session 将实时生成流式结果放到队列中
        answer: str = step_4_call_llm_create_answer(answer_prompt, is_stream, session_id)
        # TODO 7. 提取图片链接 image_urls
        image_urls: list[str] = step_5_extract_chunk_and_url_image(reranked_docs)
        # TODO 8. 更新状态
        state["answer"] = answer
        state["image_urls"] = image_urls
    # TODO 9. 保存聊天记录到 mongodb 中
    step_6_save_assistant_message(state)
    # TODO 10. 添加节点到已完成列表
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            """,
        }
    ]
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？", "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法"},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)