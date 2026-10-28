from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
import os

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY2")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL2")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

# ====================== 1. 定义状态 ======================
class State(TypedDict):
    user_raw_requirement: str       # 用户最开始输入的旅游需求
    travel_info: dict               # 收集到出行信息：目的地、出行日期、预算、出行人数、偏好
    info_check_result: str          # 信息校验结果（是否齐全/缺失字段）
    travel_draft_plan: str          # 生成的旅游方案草稿
    human_review_input: str         # 用户审核输入：同意 / 修改:xxx
    final_travel_plan: str          # 最终定稿方案

# ====================== 2. LLM 初始化 ======================
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY
)

# 提示词：收集并校验旅游必填信息
info_collect_prompt = ChatPromptTemplate.from_template("""
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

# 提示词：生成旅游草稿方案
plan_generate_prompt = ChatPromptTemplate.from_template("""
基于下面用户出行信息，生成一份多日旅游规划方案。
出行信息：{travel_info}

方案包含：每日行程、餐饮建议。
方案总字数不要超过30个字。
""")

# 提示词：根据用户反馈修改方案
plan_refine_prompt = ChatPromptTemplate.from_template("""
旧旅游方案草稿：{travel_draft_plan}
用户修改意见：{human_review_input}
出行基础信息：{travel_info}

根据用户意见调整旅游方案，新方案总字数不要超过30个字。
""")

# ====================== 节点1：收集 & 校验旅游信息 ======================
def check_travel_info(state: State):
    # llm.invoke():同步调用大模型
    # format:把相应的数据填充进提示词模板info_collect_prompt中传给LLM
    resp = llm.invoke(info_collect_prompt.format(
        user_raw_requirement=state["user_raw_requirement"],
        travel_info=state.get("travel_info", {})
    ))
    # 获取LLM返回的数据
    # .strip()用于清除字符串首尾空格、换行符
    content = resp.content.strip()
    try:
        # 尝试解析json = 信息齐全，提取字段
        parsed_info = json.loads(content)
        return {
            "info_check_result": "信息齐全",
            "travel_info": parsed_info
        }
    except json.JSONDecodeError:
        # 解析失败 = 返回的是缺失字段文本
        return {"info_check_result": content}


# ====================== 节点2：信息不足，人工收集补充信息（interrupt暂停） ======================
def collect_missing_info(state: State):
    # interrupt:暂停，等待用户补充缺失的出行信息
    user_input_str = interrupt({
        "message": f"信息收集不全，需要补充：{state['info_check_result']}",
        "current_collected": state.get("travel_info", {})
    })
    # 交给LLM把自然语言转成结构化dict，不再直接把字符串解包
    parse_prompt = ChatPromptTemplate.from_template("""
    当前已有的出行信息：{old_info}
    用户刚刚补充的内容：{user_text}
    提取里面的旅游字段，只输出严格json，key只能是：目的地、出行日期、总预算、出行人数、出行偏好
    没有提到的key不要写进去，不要额外解释，只返回json
    """)
    parse_resp = llm.invoke(parse_prompt.format(
        old_info=state.get("travel_info", {}),
        user_text=user_input_str
    ))
    try:
        # 将LLM返回的内容解析成字典
        new_info_part = json.loads(parse_resp.content)
    except Exception:
        # 解析失败，新信息为空字典
        new_info_part = {}
    # 将原有信息和新增信息合并
    new_travel_info = {**state.get("travel_info", {}), **new_info_part}
    # Command：更新状态，并且指定下一步跳回信息校验节点，重新检查是否齐全
    return Command(update={"travel_info": new_travel_info}, goto="check_travel_info")

# ====================== 节点3：生成旅游草稿方案 ======================
def create_draft_plan(state: State):
    # 使用已经收集完整的travel_info调用LLM生成方案
    resp = llm.invoke(plan_generate_prompt.format(travel_info=state["travel_info"]))
    # 把生成的草稿存入状态的travel_draft_plan字段
    return {"travel_draft_plan": resp.content}

# ====================== 节点4：人工审核方案（核心人在回路） ======================
def human_audit_plan(state: State):
    human_input = interrupt({
        "message": "请审核旅游方案，输入【同意】定稿；输入【修改:你的修改要求】调整方案",
        "draft_plan": state["travel_draft_plan"],
        "travel_info": state["travel_info"]
    })
    # 用户输入判断
    if human_input.strip() == "同意":
        # 用户确认，把草稿作为最终方案，走向END结束
        return Command(
            update={
                "human_review_input": human_input,
                "final_travel_plan": state["travel_draft_plan"]
            },
            goto=END
        )
    else:
        # 用户选择修改，去掉前缀“修改:”拿到修改指令
        modify_content = human_input.replace("修改:", "").strip()
        # 更新修改意见，跳转到修订方案节点
        return Command(
            update={"human_review_input": modify_content},
            goto="revise_plan"
        )

# ====================== 节点5：根据反馈修订旅游方案 ======================
def revise_plan(state: State):
    # 携带旧方案、修改意见、基础信息交给LLM重新生成
    resp = llm.invoke(plan_refine_prompt.format(
        travel_draft_plan=state["travel_draft_plan"],
        human_review_input=state["human_review_input"],
        travel_info=state["travel_info"]
    ))
    # 更新草稿，回到人工审核节点，再次交给用户审阅
    return {"travel_draft_plan": resp.content}

# ====================== 条件路由：信息校验后的分支跳转 ======================
def route_info_check(state: State):
    # 包含匹配，不再严格相等，兼容【信息齐全】这种带符号输出
    if "信息齐全" in state["info_check_result"]:
        # 信息齐全 → 生成方案
        return "create_draft_plan"
    else:
        # 信息缺失 → 进入收集补充节点
        return "collect_missing_info"

# ====================== 构建图 ======================
builder = StateGraph(State)

# 注册全部节点
builder.add_node("check_travel_info", check_travel_info)
builder.add_node("collect_missing_info", collect_missing_info)
builder.add_node("create_draft_plan", create_draft_plan)
builder.add_node("human_audit_plan", human_audit_plan)
builder.add_node("revise_plan", revise_plan)

# 起始边
builder.add_edge(START, "check_travel_info")
# 信息校验后走条件分支
builder.add_conditional_edges("check_travel_info", route_info_check)
# 草稿生成完成，进入人工审核
builder.add_edge("create_draft_plan", "human_audit_plan")
# 修改完方案，回到人工审核
builder.add_edge("revise_plan", "human_audit_plan")

# LangGraph 内置的检查点存储实现，数据存在程序内存里（Python 进程内存）
memory = MemorySaver()
# checkpointer:图编译的开关
graph = builder.compile(checkpointer=memory)

# ====================== 交互式运行 ======================
if __name__ == "__main__":
    # thread_id 区分会话，记忆保存在这个会话内，用于后续上线FastAPI时区分不同会话
    config = {"configurable": {"thread_id": "travel_001"}}
    # 控制台接收用户第一条旅行需求输入
    user_input = input("请输入你的旅行需求：")

    print("===== 旅游规划Agent开始执行 =====")
    # 把用户初始输入传入提示词
    input_payload = {
        "user_raw_requirement": user_input,
        "travel_info": {}
    }
    # 启动graph
    for event in graph.stream(input_payload, config, stream_mode="updates"):
        print(event)

    # 一对情侣在2026.11.1日到达长沙，玩3天，喜欢自然风光，预算3000

    # 循环处理所有interrupt暂停点
    snapshot = graph.get_state(config)
    while snapshot.next:
        print("\n===== 等待你输入 =====")
        user_input = input("请按提示输入内容：")
        # resume=用户输入，恢复被interrupt暂停的图
        graph.invoke(Command(resume=user_input), config)
        # 重新拉取最新状态快照
        snapshot = graph.get_state(config)
        print("\n当前状态：", snapshot.values)
