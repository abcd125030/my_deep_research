# 深度研究助手（LangGraph 简版）

## ✨ 项目概述

基于 LangChain 与 LangGraph 的轻量级深度研究助手。它通过 LLM 自动完成以下流程：

- 生成结构化的网络搜索计划
- 执行多步 Web 搜索并汇总要点
- 产出中文 Markdown 研究报告

本目录包含两种实现：

- `mini_deepresearch.py`：函数式顺序实现，便于快速理解与改造
- `simple_deep_research_langgraph.py`：基于 LangGraph 的状态机工作流实现

## 🎯 功能特性

- **规划搜索**：LLM 生成 8–10 条搜索子任务（结构化 JSON）
- **执行搜索**：集成 `TavilySearch`，按子任务检索并生成精炼摘要
- **撰写报告**：高级研究员角色生成中文 Markdown 报告（结构化输出）
- **可扩展工作流**：LangGraph 节点化编排，易于新增校验、优化、分支等节点

## 🏗 架构与文件

- `mini_deepresearch.py`：规划 → 搜索 → 写作 的函数式串联
- `simple_deep_research_langgraph.py`：`plan` → `search` → `report` 的状态图工作流

核心数据结构：

- `WebSearchItem`（搜索项）
- `WebSearchPlan`（搜索计划）
- `ReportData`（报告数据）
- `ResearchState`（工作流状态，仅 LangGraph 版本）

## 🚀 快速开始

```bash
# 创建 .env 并写入密钥
TAVILY_API_KEY=你的Tavily密钥
OPENAI_API_KEY=你的DashScope或OpenAI兼容密钥
MODEL_NAME=你的DashScope模型名称，如qwen-plus

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（任选其一）
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装 Python 依赖
pip install -r requirements.txt

# 运行代码
python simple_deep_research_langgraph.py
# 或
python mini_deepresearch.py
```



## 🔧 工作流说明

- **规划阶段**：根据原始查询生成结构化搜索计划（JSON），每项包含 `reason` 与 `query`
- **搜索阶段**：逐项调用搜索代理，产出 5–10 段、<1000 字的精炼摘要
- **写作阶段**：汇总搜索结果，生成中文 Markdown 报告，包含摘要与后续问题

LangGraph 版本通过 `StateGraph` 将三阶段编排为节点，自动合并状态：

- `plan` 节点：生成 `search_plan`
- `search` 节点：输出 `search_results`
- `report` 节点：写出 `report_content`

## 📚 关键类与函数（文件定位）

- `WebSearchItem`：`simple_deep_research_langgraph.py:14`，`mini_deepresearch.py:13`
- `WebSearchPlan`：`simple_deep_research_langgraph.py:21`，`mini_deepresearch.py:20`
- `ReportData`：`simple_deep_research_langgraph.py:78`，`mini_deepresearch.py:77`
- `ResearchState`：`simple_deep_research_langgraph.py:106`
- `plan_node`：`simple_deep_research_langgraph.py:114`
- `search_node`：`simple_deep_research_langgraph.py:121`
- `report_node`：`simple_deep_research_langgraph.py:135`
- `build_research_graph`：`simple_deep_research_langgraph.py:144`
- `_plan_searches`：`mini_deepresearch.py:106`
- `_perform_searches`：`mini_deepresearch.py:112`
- `_search`：`mini_deepresearch.py:122`
- `_write_report`：`mini_deepresearch.py:131`

## ⚠️ 注意事项

- `mini_deepresearch.py` 的 `_perform_searches` 目前返回的是最后一个结果而非列表，建议将返回值改为 `results`（位置：`mini_deepresearch.py:112-120`）
- 不要在代码中硬编码密钥，推荐使用环境变量并从代码读取
- Tavily 与模型 API 可能存在速率与配额限制，建议合理控制并发
- 写作提示词已固定要求中文输出，如需英文可调整 `write_template`

## 📎 参考与扩展

- 更详细的函数式说明：`mini_deepresearch.md`
- LangGraph 工作流细节：`simple_deep_research_langgraph.md`

欢迎根据你的场景扩展节点（校验、优化、分支）或替换搜索与模型后端。
