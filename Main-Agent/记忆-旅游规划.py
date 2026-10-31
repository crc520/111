from typing import TypedDict, Any
import re
import json
import asyncio
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

from helper2 import save_memory, recall_memory, rebuild_chroma_from_mysql, save_snapshot, get_snapshot, get_max_version

load_dotenv()

PYTHON_EXE_PATH = os.getenv("PYTHON_EXE_PATH")
WEATHER_SCRIPT_PATH = os.getenv("WEATHER_SCRIPT_PATH")
FS_SCRIPT_PATH = os.getenv("FS_SCRIPT_PATH")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

MCP_SERVER_LIST = [
    {"service_id": "weather", "script_path": WEATHER_SCRIPT_PATH},
    {"service_id": "fs", "script_path": FS_SCRIPT_PATH}
]
# MCP 全局资源
exit_stack = AsyncExitStack()
mcp_sessions: dict[str, ClientSession] = {}
tool_owner_map: dict[str, str] = {}
mcp_tools_schema: list[dict] = []
langchain_tool_list: list[Tool] = []

async def start_single_mcp(service_id: str, script_path: str):
    params = StdioServerParameters(
        command=PYTHON_EXE_PATH,
        args=[script_path]
    )
    streams = await exit_stack.enter_async_context(stdio_client(params))
    read, write = streams
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tool_response = await session.list_tools()
    mcp_sessions[service_id] = session
    for tool in tool_response.tools:
        tool_owner_map[tool.name] = service_id
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        }
        mcp_tools_schema.append(schema)

async def init_all_mcp():
    """程序启动时，批量拉起所有MCP服务"""
    global mcp_sessions, tool_owner_map, mcp_tools_schema
    mcp_sessions = {}
    tool_owner_map = {}
    mcp_tools_schema = []
    for cfg in MCP_SERVER_LIST:
        sid = cfg["service_id"]
        spath = cfg["script_path"]
        try:
            await start_single_mcp(sid, spath)
        except Exception as e:
            print(f"❌ MCP服务 [{sid}] 启动失败，跳过该服务: {e}")
    print(f"📥 已加载MCP工具：{list(tool_owner_map.keys())}")

async def shutdown_mcp():
    await exit_stack.aclose()

async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """统一调用MCP工具"""
    service_id = tool_owner_map[tool_name]
    session = mcp_sessions[service_id]
    res = await session.call_tool(name=tool_name, arguments=arguments)
    return res.content[0].text

# ====================== 1. 状态定义，新增 tool_messages 存放工具调用上下文 ======================
class State(TypedDict):
    user_raw_requirement: str
    travel_info: dict
    info_check_result: str
    travel_draft_plan: str
    human_review_input: str
    final_travel_plan: str
    long_memory: list
    base_version: int | None
    tool_messages: list  # 新增：工具调用消息栈

# ====================== 2. LLM 初始化，绑定MCP工具schema ======================
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
)

# ---------------- 节点0：解析用户是否要求基于历史版本修改 ----------------
def parse_base_version(state: State, config):
    user_text = state["user_raw_requirement"]
    session_id = config["configurable"]["thread_id"]
    pattern = re.search(r"第(\d+)版|第(\d+)轮", user_text)
    if pattern:
        v_num = int(pattern.group(1))
        snapshot_data = get_snapshot(session_id, v_num)
        if snapshot_data is not None:
            print(f"📌 识别到基于版本{v_num}修改，加载快照：{snapshot_data}")
            return {
                "base_version": v_num,
                "travel_info": snapshot_data,
                "long_memory": [],
                "tool_messages": []
            }
    return {"base_version": None, "tool_messages": []}

# ---------------- 节点：召回长期记忆 ----------------
def recall_long_memory(state: State, config):
    session_id = config["configurable"]["thread_id"]
    query = state["user_raw_requirement"]
    memory_list = recall_memory(session_id, query, top_k=5)
    return {"long_memory": memory_list}

# ---------------- 节点：提取事实、保存到长期记忆 ----------------
def save_long_memory(state: State, config):
    session_id = config["configurable"]["thread_id"]
    info = state["travel_info"]
    facts = []
    if info.get("目的地"):
        facts.append( (f"用户想去的目的地：{info['目的地']}", "place") )
    if info.get("出行日期"):
        facts.append( (f"用户出行时间：{info['出行日期']}", "time") )
    if info.get("总预算"):
        facts.append( (f"用户出行预算：{info['总预算']}", "money") )
    if info.get("出行人数"):
        facts.append( (f"出行人数：{info['出行人数']}", "people") )
    if info.get("出行偏好"):
        facts.append( (f"用户出行偏好：{info['出行偏好']}", "preference") )

    for text, typ in facts:
        save_memory(session_id, text, memory_type=typ)
    print("✅ 本次出行信息已存入长期事实记忆")
    return {}

# ---------------- 节点：保存整套快照（定稿时执行） ----------------
def save_travel_snapshot(state: State, config):
    session_id = config["configurable"]["thread_id"]
    current_info = state["travel_info"]
    max_v = get_max_version(session_id)
    new_v = max_v + 1
    save_snapshot(session_id, new_v, current_info)
    print(f"📷 快照已保存，版本号：{new_v}")
    return {}

# ====================== 提示词 ======================
info_collect_prompt = ChatPromptTemplate.from_template("""
历史长期记忆：{long_memory}
用户原始需求：{user_raw_requirement}
当前已收集信息：{travel_info}

你需要检查是否收集齐全下面5项信息：
1.目的地城市 
2.出行日期（起止时间） 
3.总预算 
4.出行人数 
5.出行偏好（休闲/徒步/美食打卡等）

如果有缺失，只返回缺失的字段名称，多个字段用中文逗号隔开；
如果全部齐全：直接输出JSON，key为【目的地、出行日期、总预算、出行人数、出行偏好】，不要额外文字。
""")

plan_generate_prompt = ChatPromptTemplate.from_template("""
历史长期记忆：{long_memory}
基于下面用户出行信息，生成一份多日旅游规划方案。
出行信息：{travel_info}
你可以调用工具查询目的地天气、读取本地攻略文件，方案要结合天气情况。

方案包含：每日行程、餐饮建议。方案总字数不要超过120个字。
""")

plan_refine_prompt = ChatPromptTemplate.from_template("""
历史长期记忆：{long_memory}
旧旅游方案草稿：{travel_draft_plan}
用户修改意见：{human_review_input}
出行基础信息：{travel_info}

根据用户意见调整旅游方案，可以调用工具查询天气。新方案总字数不要超过120个字。
""")

# ====================== 节点1：收集 & 校验旅游信息 ======================
def check_travel_info(state: State):
    resp = llm.invoke(info_collect_prompt.format(
        user_raw_requirement=state["user_raw_requirement"],
        travel_info=state.get("travel_info", {}),
        long_memory=state.get("long_memory", [])
    ))
    content = resp.content.strip()
    try:
        parsed_info = json.loads(content)
        return {
            "info_check_result": "信息齐全",
            "travel_info": parsed_info
        }
    except json.JSONDecodeError:
        return {"info_check_result": content}

# ====================== 节点2：信息不足，人工收集补充信息 ======================
def collect_missing_info(state: State):
    user_input_str = interrupt({
        "message": f"信息收集不全，需要补充：{state['info_check_result']}",
        "current_collected": state.get("travel_info", {})
    })
    parse_prompt = ChatPromptTemplate.from_template("""
    历史长期记忆：{long_memory}
    当前已有的出行信息：{old_info}
    用户刚刚补充的内容：{user_text}
    提取里面的旅游字段，只输出严格json，key只能是：目的地、出行日期、总预算、出行人数、出行偏好
    没有提到的key不要写进去，不要额外解释，只返回json
    """)
    parse_resp = llm.invoke(parse_prompt.format(
        old_info=state.get("travel_info", {}),
        user_text=user_input_str,
        long_memory=state.get("long_memory", [])
    ))
    try:
        new_info_part = json.loads(parse_resp.content)
    except Exception:
        new_info_part = {}
    new_travel_info = {**state.get("travel_info", {}), **new_info_part}
    return Command(update={"travel_info": new_travel_info}, goto="check_travel_info")

# ====================== 【新增核心节点：生成方案 + 工具调用循环】 ======================
async def create_draft_plan_with_tools(state: State):
    messages = [
        {"role": "system", "content": "你是旅游规划助手，可以调用工具：查询城市天气、文件读写。严格使用工具获取真实信息，禁止编造。"},
        {"role": "user", "content": plan_generate_prompt.format(
            travel_info=state["travel_info"],
            long_memory=state["long_memory"]
        )}
    ]
    messages.extend(state["tool_messages"])
    max_loop = 5
    for _ in range(max_loop):
        resp =await llm.ainvoke(messages, tools=mcp_tools_schema)
        msg = resp.content
        tool_calls = resp.tool_calls
        if not tool_calls:
            return {"travel_draft_plan": msg, "tool_messages": []}
        # 执行MCP工具
        messages.append(resp.model_dump())
        for tc in tool_calls:
            # ✅ 兼容：如果是对象就转dict，如果已经是dict直接用
            if hasattr(tc, "model_dump"):
                tc = tc.model_dump()
            tool_name = tc["name"]
            args = tc["args"]
            res_text = await call_mcp_tool(tool_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": res_text
            })
    return {"travel_draft_plan": msg, "tool_messages": messages}

# ====================== 节点4：人工审核方案 ======================
def human_audit_plan(state: State):
    human_input = interrupt({
        "message": "请审核旅游方案，输入【同意】定稿；输入【修改:你的修改要求】调整方案",
        "draft_plan": state["travel_draft_plan"],
        "travel_info": state["travel_info"],
        "long_memory": state.get("long_memory", [])
    })
    if human_input.strip() == "同意":
        return Command(
            update={
                "human_review_input": human_input,
                "final_travel_plan": state["travel_draft_plan"]
            },
            goto="save_long_memory"
        )
    else:
        modify_content = human_input.replace("修改:", "").strip()
        return Command(
            update={"human_review_input": modify_content, "tool_messages": []},
            goto="revise_plan"
        )

# ====================== 节点5：修订方案（同样支持工具调用） ======================
async def revise_plan(state: State):
    messages = [
        {"role": "system", "content": "你是旅游规划助手，可以调用工具查询天气、读取本地文件，不要编造信息。"},
        {"role": "user", "content": plan_refine_prompt.format(
            travel_draft_plan=state["travel_draft_plan"],
            human_review_input=state["human_review_input"],
            travel_info=state["travel_info"],
            long_memory=state["long_memory"]
        )}
    ]
    messages.extend(state["tool_messages"])
    max_loop = 5
    for _ in range(max_loop):
        resp =await llm.ainvoke(messages, tools=mcp_tools_schema)
        msg = resp.content
        tool_calls = resp.tool_calls
        if not tool_calls:
            return {"travel_draft_plan": msg, "tool_messages": []}
        messages.append(resp.model_dump())
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                tc = tc.model_dump()
            tool_name = tc["name"]
            args = tc["args"]
            res_text = await call_mcp_tool(tool_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": res_text
            })
    return {"travel_draft_plan": msg, "tool_messages": messages}

# ====================== 路由 ======================
def route_info_check(state: State):
    if "信息齐全" in state["info_check_result"]:
        return "create_draft_plan_with_tools"
    else:
        return "collect_missing_info"

def after_save_fact(state: State):
    return "save_travel_snapshot"

def route_after_parse(state: State):
    if state.get("base_version") is not None:
        return "check_travel_info"
    else:
        return "recall_long_memory"

# ====================== 构图 ======================
builder = StateGraph(State)
builder.add_node("parse_base_version", parse_base_version)
builder.add_node("recall_long_memory", recall_long_memory)
builder.add_node("check_travel_info", check_travel_info)
builder.add_node("collect_missing_info", collect_missing_info)
builder.add_node("create_draft_plan_with_tools", create_draft_plan_with_tools)
builder.add_node("human_audit_plan", human_audit_plan)
builder.add_node("revise_plan", revise_plan)
builder.add_node("save_long_memory", save_long_memory)
builder.add_node("save_travel_snapshot", save_travel_snapshot)

builder.add_edge(START, "parse_base_version")
builder.add_conditional_edges("parse_base_version", route_after_parse)
builder.add_edge("recall_long_memory", "check_travel_info")
builder.add_conditional_edges("check_travel_info", route_info_check)
builder.add_edge("create_draft_plan_with_tools", "human_audit_plan")
builder.add_edge("revise_plan", "human_audit_plan")
builder.add_conditional_edges("save_long_memory", after_save_fact)
builder.add_edge("save_travel_snapshot", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ====================== 交互式运行（主入口，异步启动MCP） ======================
async def main():
    # 初始化MCP服务，拉起weather、fs子进程
    await init_all_mcp()
    try:
        config = {"configurable": {"thread_id": "travel_001"}}
        user_input = input("请输入你的旅行需求（如使用历史版本，请以“第1版”的格式输入）：")
        print("===== 旅游规划Agent开始执行 =====")
        input_payload = {
            "user_raw_requirement": user_input,
            "travel_info": {},
            "long_memory": [],
            "base_version": None,
            "tool_messages": []
        }
        # 异步stream
        async for event in graph.astream(input_payload, config, stream_mode="updates"):
            print(event)
        snapshot = await graph.aget_state(config)
        while snapshot.next:
            print("\n===== 等待你输入 =====")
            user_input = input("请按提示输入内容：")
            await graph.ainvoke(Command(resume=user_input), config)
            snapshot = await graph.aget_state(config)
            print("\n当前状态：", snapshot.values)
    finally:
        # 程序退出，关闭所有MCP子进程
        await shutdown_mcp()

if __name__ == "__main__":
    asyncio.run(main())

