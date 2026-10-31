# LangGraph 旅游规划 Agent + MCP 工具服务
基于 Python + LangGraph + DeepSeek + MCP stdio 实现的人在回路旅游规划智能体
配套 FastAPI 服务，集成天气查询、本地文件读写MCP工具。

## ✨ 功能特性
- 旅游信息收集、参数校验、缺失信息人工补充
- 自动生成旅游方案草稿，支持人工审核、修改迭代
- MCP本地工具：天气查询、Markdown报表生成
- FastAPI 对外接口，支持http调用Agent
