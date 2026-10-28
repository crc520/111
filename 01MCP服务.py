# 导入json库，用于解析LLM返回的工具调用参数json字符串
import json
# 导入OpenAI客户端，DeepSeek兼容OpenAI接口规范，直接复用这个SDK
from openai import OpenAI
# 导入FastAPI框架基础组件，Query用于接收url查询参数
from fastapi import FastAPI, Query
from mcp import StdioServerParameters
import asyncio
# AsyncExitStack：异步资源管理器，统一管理多个异步上下文、自动关闭连接
from contextlib import AsyncExitStack
# MCP客户端会话对象，用来和MCP服务交互
from mcp import ClientSession
# =========重点：替换成 stdio 客户端，不再用sse_client=========
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import os

load_dotenv()
# =================【配置区】=====================
# 本地weather.py的绝对路径
WEATHER_SCRIPT_PATH = WEATHER_SCRIPT_PATH
# fs 文件MCP脚本绝对路径（新增）
FS_SCRIPT_PATH = FS_SCRIPT_PATH
DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL
DEEPSEEK_MODEL = DEEPSEEK_MODEL

MCP_SERVER_LIST = [
    {
        "service_id": "weather",
        "script_path": WEATHER_SCRIPT_PATH
    },
    {
        "service_id": "fs",
        "script_path": FS_SCRIPT_PATH
    }
]

# ===============================================

app = FastAPI()
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ==========全局MCP对象，服务启动时只连接一次==========
# 异步资源栈，管理MCP的stdio子进程、session会话，程序退出时统一释放
exit_stack = AsyncExitStack()
# 保存多个MCP会话：key是服务标识 "weather" / "fs"
mcp_sessions = dict()
# 工具映射表：用来查找调用哪个session
tool_owner_map = dict()
# 存放从MCP服务拉取并转换格式后的工具列表，直接传给DeepSeek function calling
mcp_tools_schema = []


async def start_single_mcp(service_id, script_path):
    params = StdioServerParameters(
        command=command,
        args=[script_path]
    )
    # 进入异步上下文，拉起子进程，拿到读写流
    streams = await exit_stack.enter_async_context(stdio_client(params))
    read, write = streams
    # 创建MCP会话
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    # 和MCP服务握手初始化
    await session.initialize()
    # 获取相关MCP服务的所有工具
    tool_response = await session.list_tools()
    # 保存查询天气会话对象到mcp_sessions字典，key=weather
    mcp_sessions[service_id] = session
    for tool in tool_response.tools:
        tool_owner_map[tool.name] = service_id
        mcp_tools_schema.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        })

# 服务启动事件：uvicorn启动完成后，自动执行，拉起本地weather.py子进程，建立MCP连接
@app.on_event("startup")
async def startup():
    # 声明使用全局变量
    global mcp_sessions, tool_owner_map, mcp_tools_schema
    # 清空，防止重复加载
    mcp_sessions = {}
    tool_owner_map = {}
    mcp_tools_schema = []
    # 遍历配置清单，批量启动所有MCP
    for cfg in MCP_SERVER_LIST:
        sid = cfg["service_id"]
        spath = cfg["script_path"]
        try:
            await start_single_mcp(sid, spath)
        except Exception as e:
            print(f"❌ MCP服务 [{sid}] 启动失败，跳过该服务: {e}")
    # 全部加载完成，打印调试信息
    print(f"📥 已加载全部工具：{list(tool_owner_map.keys())}")

# 服务关闭事件：uvicorn服务停止时触发，关闭weather子进程、释放资源
@app.on_event("shutdown")
async def shutdown():
    # 关闭所有在exit_stack中管理的异步资源，杀掉weather子进程
    await exit_stack.aclose()


async def get_response(user_query: str):
    max_retry = 5          # 最大重试次数，防止LLM反复调用工具死循环
    current_retry = 0      # 当前重试计数器
    # 消息上下文数组，保存system提示词、用户提问、模型回复、工具返回结果
    messages = [
        {"role": "system", "content":
            "你是一个助手，可以调用工具：查询天气、列出目录、读取文本文件、写入文本文件、在文件内搜索关键词。严格使用工具获取信息，不要编造结果。"

         },
        {"role": "user", "content": user_query},
    ]
    # Agent循环：持续轮询大模型，直到返回最终回答或者达到重试上限
    while current_retry < max_retry:
        try:
            # 请求DeepSeek模型，传入消息上下文 + MCP拉取的工具定义
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                stream=False,
                tools=mcp_tools_schema,       # 使用MCP拉取的工具
                tool_choice="auto"            # auto：模型自主判断是否调用工具
            )
            # 获取模型返回的第一条消息
            msg = response.choices[0].message
            # 判断模型是否要求调用工具
            if msg.tool_calls is not None:
                # 将模型返回的assistant消息转字典，追加到消息上下文
                messages.append(msg.model_dump())
                # 遍历LLM返回全部tool_call，支持一次性并行调用多个工具
                for tool_call in msg.tool_calls:
                    # 获取要调用的工具名称
                    func_name = tool_call.function.name
                    # 解析工具入参json字符串，转为python字典
                    arguments = json.loads(tool_call.function.arguments)

                    # 查表找到对应的MCP会话：工具名 → 找到所属服务标识
                    service_name = tool_owner_map[func_name]
                    # 通过服务标识拿到对应的MCP会话对象
                    target_session = mcp_sessions[service_name]
                    # 使用查到的session调用工具
                    tool_result_raw = await target_session.call_tool(name=func_name, arguments=arguments)

                    # 提取MCP返回的文本内容（MCP返回是数组，取第一个文本块）
                    tool_result = tool_result_raw.content[0].text
                    # 把工具执行结果包装成tool消息，加入消息上下文，传回大模型
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                # 工具调用完成，回到循环开头，再次请求LLM，让模型根据工具结果生成最终回答
                continue
            else:
                # 模型不需要调用工具，直接返回最终回答
                final_answer = msg.content
                return final_answer
        except Exception as e:
            # 捕获异常，打印错误信息
            print("执行异常：", e)
            # 重试次数+1
            current_retry += 1
            # 异常后重置上下文，防止坏消息污染后续请求
            messages = [
                {"role": "system", "content":
                    "你是一个助手，可以调用工具：查询天气、列出目录、读取文本文件、写入文本文件、在文件内搜索关键词。严格使用工具获取信息，不要编造结果。"
},
                {"role": "user", "content": user_query},
            ]
    # 达到最大重试次数，返回失败提示
    return "重试达到上限，生成数据失败"

# 路由接口，改成async异步函数，调用异步Agent逻辑
@app.get("/run-agent")
async def run_agent(query: str = "你好，你能做什么"):
    # 调用agent函数，await等待异步执行完成
    result = await get_response(query)
    # 返回json结果
    return {"success": True, "msg": result}

# 程序入口：直接运行此py文件时，启动uvicorn服务
if __name__ == '__main__':
    import uvicorn
    # 启动服务，reload=True开启热重载（开发调试用；生产环境要关闭）
    uvicorn.run("01MCP服务:app", host="127.0.0.1", port=8000,reload=False)
