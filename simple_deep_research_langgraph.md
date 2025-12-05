# 基于LangGraph的深度研究助手

## 概述

这个模块实现了一个基于LangChain和LangGraph的深度研究助手，它能够自动完成从研究规划、网络搜索到报告生成的全流程。该系统采用LangGraph的StateGraph构建工作流，由三个主要节点构成：搜索规划器(Planner)、搜索执行器(Searcher)和报告生成器(Writer)，通过这三个节点的协同工作，可以为用户提供全面、深入的研究报告。

## 系统架构

整个深度研究助手系统由以下三个主要阶段组成：

1. **搜索规划阶段**：根据用户查询生成结构化的搜索计划
2. **搜索执行阶段**：执行搜索计划中的每个搜索步骤并获取结果
3. **报告生成阶段**：基于搜索结果生成综合性研究报告

系统使用LangGraph的StateGraph构建工作流，实现了节点间的状态传递和流程控制。

## 核心组件

### 1. 数据模型

#### WebSearchItem 类

`WebSearchItem` 类用于表示搜索计划中的单个搜索步骤，包含以下字段：

- `reason`: 搜索原因，解释为什么这个搜索对于查询很重要
- `query`: 用于Web搜索的搜索词

```python
class WebSearchItem(BaseModel):
    """表示研究过程中的一个步骤"""
    reason: str = Field(description="Your reasoning for why this search is important to the query")
    query: str = Field(description="The search term to use for the web search.")
```

#### WebSearchPlan 类

`WebSearchPlan` 类用于表示完整的搜索计划，包含以下字段：

- `searches`: WebSearchItem对象的列表，表示为了最好地回答查询而需要执行的Web搜索列表

```python
class WebSearchPlan(BaseModel):
    """表示完整的研究计划"""
    searches: List[WebSearchItem] = Field(default_factory=list, description="A list of web searches to perform to best answer the query")
```

#### ReportData 类

`ReportData` 类用于表示最终生成的研究报告数据，包含以下字段：

- `short_summary`: 研究发现的简短摘要（2-3句话）
- `markdown_report`: Markdown格式的完整报告
- `follow_up_questions`: 建议进一步研究的主题列表

```python
class ReportData(BaseModel):
    """表示研究过程中的一个步骤"""
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings")
    markdown_report: str = Field(description="A markdown final report")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested topics to research further")
```

#### ResearchState 类

`ResearchState` 是一个TypedDict，用于定义LangGraph工作流中的状态：

```python
class ResearchState(TypedDict):
    query: str
    search_plan: Union[WebSearchPlan, None]
    search_results: Union[list, None]
    report_content: Union[str, None]  # 避免与节点名称冲突
    status: str
```

### 2. 提示词模板

#### Planner 提示词模板

`planner_template` 定义了如何指导LLM生成Web搜索计划。模板包括：

- 角色定义：有帮助的研究助手
- 输入参数：用户查询
- 输出要求：包含8-10个搜索步骤的JSON格式搜索计划，每个步骤包含搜索原因和搜索词
- 示例输出：提供了JSON格式的示例，展示了如何构建搜索步骤

#### Search 提示词模板

`search_template` 定义了如何指导LLM执行搜索并生成搜索结果摘要：

- 角色定义：有帮助的研究助手
- 任务描述：根据搜索项进行网络搜索并生成结果的简洁摘要
- 输出要求：5-10段落，少于1000字，捕捉主要观点，简洁表达

#### Writer 提示词模板

`write_template` 定义了如何指导LLM生成最终研究报告：

- 角色定义：高级研究员
- 任务描述：为研究查询编写连贯的报告
- 输出要求：先创建报告大纲，然后生成详细报告，使用Markdown格式，内容详尽（5-10页，至少500字）
- 输出格式：JSON格式，包含short_summary、markdown_report和follow_up_questions字段
- 语言要求：最终结果用中文输出

### 3. 工具与代理

#### TavilySearch 工具

系统使用 `TavilySearch` 作为网络搜索工具，配置为每次搜索返回5个结果：

```python
search_tool = TavilySearch(max_results=5, topic="general")
```

#### ReAct Agent

系统使用 LangGraph 的 `create_react_agent` 创建一个能够执行搜索的代理：

```python
search_agent = create_react_agent(model=llm, tools=[search_tool], prompt=search_template)
```

### 4. LangGraph 工作流节点

#### plan_node 函数

根据用户查询生成搜索计划：

```python
def plan_node(state: ResearchState) -> ResearchState:
    """规划搜索步骤"""
    query = state["query"]
    planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)
    search_plan = planner_chain.invoke({"query": query})
    return {"search_plan": search_plan, "status": "planned"}
```

#### search_node 函数

执行搜索计划中的每个搜索步骤：

```python
def search_node(state: ResearchState) -> ResearchState:
    """执行搜索"""
    search_plan = state["search_plan"]
    results = []
    for item in search_plan.searches:
        print(f"child_query: {item.query}")
        try:
            final_query = f"Search term: {item.query}\nReason for searching: {item.reason}"
            result = search_agent.invoke({"messages": [{"role": "user", "content": final_query}]})
            results.append(str(result['messages'][-1].content))
        except Exception:
            pass
    return {"search_results": results, "status": "searched"}
```

#### report_node 函数

基于原始查询和搜索结果生成研究报告：

```python
def report_node(state: ResearchState) -> ResearchState:
    """生成报告"""
    query = state["query"]
    search_results = state["search_results"]
    final_query = f"Original query: {query}\nSummarized search results: {search_results}"
    result = writer_chain.invoke({"query": final_query})
    return {"report_content": result.markdown_report, "status": "completed"}
```

### 5. StateGraph 构建

`build_research_graph` 函数创建并配置LangGraph的StateGraph：

```python
def build_research_graph():
    # 创建图
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("report", report_node)
    
    # 设置边和流程
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "report")
    workflow.add_edge("report", END)
    
    # 设置入口节点
    workflow.set_entry_point("plan")
    
    # 编译图
    return workflow.compile()
```

## 使用方法

### 基本用法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from simple_deep_research_langgraph import build_research_graph, ResearchState

# 设置必要的API密钥
import os
os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"

# 创建LLM实例
llm = ChatOpenAI(
    model_name="qwen-plus",  # 可按需更换模型名称
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    temperature=0.7,
    openai_api_key="your_api_key_here"  # 替换为您的API密钥
)

# 创建研究流程图
research_graph = build_research_graph()

# 执行研究流程
query = "请问你对AI+教育有何看法?"
result = research_graph.invoke({"query": query, "search_plan": None, "search_results": None, "report_content": None, "status": "start"})

# 输出报告
markdown_report = result["report_content"]
print("------------------研究报告--------------------------")
print(markdown_report)
```

### 完整流程示例

以下是一个完整的示例，展示了从查询到生成报告的整个流程：

```python
if __name__ == "__main__":
    query = "请问你对AI+教育有何看法?"
    
    # 创建研究流程图
    research_graph = build_research_graph()
    
    # 执行研究流程
    result = research_graph.invoke({"query": query, "search_plan": None, "search_results": None, "report_content": None, "status": "start"})
    
    # 输出报告
    markdown_report = result["report_content"]
    print("------------------markdown_report_begin--------------------------")
    print(markdown_report)
    print("------------------markdown_report_end--------------------------")
```

## LangGraph 实现特点

### 1. 状态管理

使用 `TypedDict` 定义 `ResearchState` 类型，明确指定工作流中的状态结构：

- `query`: 用户的原始查询
- `search_plan`: 搜索计划
- `search_results`: 搜索结果列表
- `report_content`: 生成的报告内容
- `status`: 当前工作流状态

### 2. 节点函数设计

每个节点函数都接收完整的状态对象，并只返回需要更新的状态部分：

- `plan_node`: 返回 `{"search_plan": search_plan, "status": "planned"}`
- `search_node`: 返回 `{"search_results": results, "status": "searched"}`
- `report_node`: 返回 `{"report_content": result.markdown_report, "status": "completed"}`

### 3. 工作流构建

使用 `StateGraph` 构建有向图工作流：

1. 创建 `StateGraph` 实例，指定状态类型
2. 添加节点，将函数与节点名称关联
3. 添加边，定义节点间的转换关系
4. 设置入口点和结束条件
5. 编译图，生成可执行的工作流

### 4. 状态键命名注意事项

在LangGraph中，状态键名称不能与节点名称相同，否则会导致冲突。在本实现中，将状态键命名为 `report_content` 而非 `report`，以避免与 `report` 节点名称冲突。

## 自定义扩展

### 修改提示词模板

您可以根据需要修改各个提示词模板，以生成不同格式或内容的搜索计划、搜索结果或研究报告：

```python
# 修改planner提示词模板示例
planner_template = """You are a helpful research assistant. Given a query, come up with a set of web searches to perform to best answer the query. Output between 5 and 7 search terms, focusing on academic sources.

Your response MUST be in JSON format with a list of search steps, each containing a 'reason' and a 'query' field.
"""

# 修改search提示词模板示例
search_template = """You are a helpful research assistant. Given a search item, you search the web for that term and produce a concise summary of the results.

The summary must be 3-5 paragraphs and less than 500 words. Focus on academic and reliable sources.
"""

# 修改write提示词模板示例
write_template = """You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some initial research done by a research assistant.
The final output should be in markdown format, and it should be concise but informative. Aim for 2-3 pages of content.
Please format your response as JSON. The response should include short_summary, markdown_report, and follow_up_questions fields.
最终结果请用中文输出，并注重学术性和客观性。"""
```

### 使用不同的LLM模型

您可以通过修改`ChatOpenAI`的参数来使用不同的模型或调整温度等参数：

```python
# 使用OpenAI的GPT-4模型
llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)

# 使用阿里云的Qwen-Max模型
llm = ChatOpenAI(
    model_name="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    temperature=0.7,
    openai_api_key="your_api_key_here"
)
```

### 扩展工作流节点

您可以扩展现有的工作流，添加更多节点以实现更复杂的功能：

```python
def validate_node(state: ResearchState) -> dict:
    """验证搜索计划"""
    search_plan = state["search_plan"]
    # 验证逻辑
    return {"status": "validated"}

def refine_node(state: ResearchState) -> dict:
    """优化搜索结果"""
    search_results = state["search_results"]
    # 优化逻辑
    return {"search_results": refined_results, "status": "refined"}

# 在build_research_graph中添加新节点
workflow.add_node("validate", validate_node)
workflow.add_node("refine", refine_node)

# 修改边的连接
workflow.add_edge("plan", "validate")
workflow.add_edge("validate", "search")
workflow.add_edge("search", "refine")
workflow.add_edge("refine", "report")
```

### 添加条件分支

您可以使用LangGraph的条件分支功能，根据状态内容动态决定下一步执行哪个节点：

```python
def branch_fn(state: ResearchState) -> str:
    """根据搜索结果数量决定下一步"""
    if len(state["search_results"]) > 5:
        return "refine"
    else:
        return "report"

# 在build_research_graph中添加条件分支
workflow.add_conditional_edges(
    "search",
    branch_fn,
    {"refine": "refine", "report": "report"}
)
```

## 系统优化建议

1. **错误处理增强**：当前的`search_node`函数在发生异常时简单跳过，可以增强错误处理逻辑，记录具体错误信息并提供更优雅的降级策略。

2. **并行搜索**：可以使用LangGraph的并行执行功能并行执行多个搜索，提高系统效率。

3. **结果缓存**：实现搜索结果缓存机制，避免重复搜索相同的查询。

4. **交互式反馈**：扩展工作流以支持用户对搜索计划或中间结果提供反馈，通过添加新的节点和条件分支实现。

5. **可视化工作流**：利用LangGraph提供的可视化功能，生成工作流图表，帮助理解和调试工作流。

## 注意事项

- 使用此模块需要有效的Tavily API密钥和OpenAI API密钥（或兼容的API密钥，如阿里云的Qwen模型API密钥）
- 生成的搜索计划和报告质量取决于提供的查询的清晰度和具体程度
- 对于复杂的查询，建议使用更高级的模型（如gpt-4或qwen-max）
- 确保在提示词模板中明确指定输出格式，并包含示例，以确保模型生成符合预期的结构化输出
- 在LangGraph中，状态键名称不能与节点名称相同，否则会导致冲突
- 节点函数只需返回需要更新的状态部分，LangGraph会自动合并到完整状态中
- 使用`with_structured_output`方法时，提示词中必须包含相关的格式指示词（如"json"），否则可能会导致错误