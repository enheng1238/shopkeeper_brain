import time
import sys

from agents.mcp import MCPServerStreamableHttp

from common.config.bailian_mcp_config import mcp_config
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task,add_running_task
from common.logging.logger import logger, node_log, step_log
import asyncio
import json




@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取 rewritten_query
    rewritten_query = state.get("rewritten_query")

    if not rewritten_query:
        logger.error(f"重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"重写的问题为空,业务无法继续,提前终止!")

    return rewritten_query

@step_log("step_2_call_mcp_tool")
async def step_2_call_mcp_tool(rewritten_query):
    # 1. 创建一个mcp的服务端连接对应mcp
    mcp_server = MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )

    # 2. 主动触发连接
    await mcp_server.connect()

    try:
        # 3. 查询所有工具信息
        list_tool = await mcp_server.list_tools()
        logger.debug(f"查询指定服务器:{mcp_config.mcp_base_url},获取工具列表:{list_tool}")

        # 4. 调用工具返回结果
        result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count":5
            }
        )
        return result
    except Exception as e:
        logger.exception(f"连接mcp报错,错误信息:{str(e)}")
    finally:
        # 释放资源
        await mcp_server.cleanup()




@node_log("node_web_search_mcp")
def node_web_search_mcp(state:QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    # TODO 1. 添加到运行中列表
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    # TODO 2. 从状态中获取数据并校验 rewritten_query
    rewritten_query: str = step_1_validate_and_get_data(state)
    # TODO 3. 调用mcp服务进行工具调用
    result = asyncio.run(step_2_call_mcp_tool(rewritten_query))
    # TODO 4. 搜索结果格式化（扁平化）
    # 3. 结果解析
    text_result: str = result.content[0].text
    web_search_docs = json.loads(text_result).get("pages", [])
    # TODO 5. 添加到已完成列表
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))


    return {"web_search_docs":web_search_docs}

if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)