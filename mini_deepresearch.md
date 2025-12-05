# 深度研究助手 (Deep Research Assistant)

## 概述

这个模块实现了一个基于LangChain和LangGraph的深度研究助手，它能够自动完成从研究规划、网络搜索到报告生成的全流程。该系统由三个主要组件构成：搜索规划器(Planner)、搜索执行器(Searcher)和报告生成器(Writer)，通过这三个组件的协同工作，可以为用户提供全面、深入的研究报告。

## 系统架构

整个深度研究助手系统由以下三个主要阶段组成：

1. **搜索规划阶段**：根据用户查询生成结构化的搜索计划
2. **搜索执行阶段**：执行搜索计划中的每个搜索步骤并获取结果
3. **报告生成阶段**：基于搜索结果生成综合性研究报告

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

### 4. 核心功能函数

#### _plan_searches 函数

根据用户查询生成搜索计划：

```python
def _plan_searches(query: str) -> WebSearchPlan:
    planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)
    planner_result = planner_chain.invoke({"query": query})
    return planner_result
```

#### _perform_searches 函数

执行搜索计划中的每个搜索步骤：

```python
def _perform_searches(search_plan: WebSearchPlan):
    results = []
    for item in search_plan.searches:
        print(f"child_query: {item.query}")
        result = _search(item)
        if result is not None:
            results.append(result)
    return results  # 注意：原代码这里有bug，应该返回results列表而不是最后一个result
```

#### _search 函数

执行单个搜索步骤并返回结果：

```python
def _search(item: WebSearchItem) -> str | None:
    try:
        final_query = f"Search term: {item.query}\nReason for searching: {item.reason}"
        result = search_agent.invoke({"messages": [{"role": "user", "content": final_query}]})
        return str(result['messages'][-1].content)
    except Exception:
        return None
```

#### _write_report 函数

基于原始查询和搜索结果生成研究报告：

```python
def _write_report(query: str, search_results) -> str:
    final_query = f"Original query: {query}\nSummarized search results: {search_results}"
    result = writer_chain.invoke({"query": final_query})
    return result.markdown_report
```

## 使用方法

### 基本用法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from mini_deepresearch import WebSearchItem, WebSearchPlan, ReportData, _plan_searches, _perform_searches, _write_report

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

# 执行深度研究流程
query = "请问你对AI+教育有何看法?"
planner_result = _plan_searches(query)
print(f"生成了{len(planner_result.searches)}个搜索步骤")

search_results = _perform_searches(planner_result)
print(f"完成了{len(search_results)}个搜索")

markdown_report = _write_report(query, search_results)
print("------------------研究报告--------------------------")
print(markdown_report)
```

### 完整流程示例

以下是一个完整的示例，展示了从查询到生成报告的整个流程：

```python
if __name__ == "__main__":
    query = "请问你对AI+教育有何看法?"
    planner_result = _plan_searches(query)
    search_results = _perform_searches(planner_result)
    markdown_report = _write_report(query, search_results)
    print("------------------markdown_report_begin--------------------------")
    print(markdown_report)
    print("------------------markdown_report_end--------------------------")
```

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

### 扩展数据模型

您可以扩展现有的数据模型，添加更多字段或方法以满足特定需求：

```python
class EnhancedWebSearchItem(WebSearchItem):
    priority: int = Field(description="Priority of this search (1-5, with 5 being highest)")
    expected_sources: List[str] = Field(default_factory=list, description="Expected sources for this search")
    max_results: int = Field(default=5, description="Maximum number of results to return for this search")

class EnhancedReportData(ReportData):
    key_findings: List[str] = Field(default_factory=list, description="Key findings from the research")
    data_visualizations: List[str] = Field(default_factory=list, description="URLs or descriptions of data visualizations")
    executive_summary: str = Field(description="A one-paragraph executive summary")
```

### 修改搜索工具

您可以调整TavilySearch工具的参数或使用其他搜索工具：

```python
# 调整TavilySearch参数
search_tool = TavilySearch(max_results=10, topic="academic")

# 使用其他搜索工具，如Google Search API
from langchain_google_search import GoogleSearchAPIWrapper
google_search = GoogleSearchAPIWrapper()
```

## 系统优化建议

1. **搜索结果聚合**：当前系统在`_perform_searches`函数中有一个bug，它只返回最后一个搜索结果而不是所有结果的列表。应修改为返回完整的结果列表。

2. **错误处理增强**：当前的`_search`函数在发生异常时简单返回None，可以增强错误处理逻辑，记录具体错误信息并提供更优雅的降级策略。

3. **并行搜索**：可以使用异步编程或多线程技术并行执行多个搜索，提高系统效率。

4. **结果缓存**：实现搜索结果缓存机制，避免重复搜索相同的查询。

5. **交互式反馈**：允许用户对搜索计划或中间结果提供反馈，以优化后续搜索和报告生成过程。

## 注意事项

- 使用此模块需要有效的Tavily API密钥和OpenAI API密钥（或兼容的API密钥，如阿里云的Qwen模型API密钥）
- 生成的搜索计划和报告质量取决于提供的查询的清晰度和具体程度
- 对于复杂的查询，建议使用更高级的模型（如gpt-4或qwen-max）
- 确保在提示词模板中明确指定输出格式，并包含示例，以确保模型生成符合预期的结构化输出
- 使用`with_structured_output`方法时，提示词中必须包含相关的格式指示词（如"json"），否则可能会导致错误
- 在使用管道操作符 `|` 连接提示词模板和LLM时，确保使用正确的方法（`from_messages`而非`from_template`）
- 网络搜索结果可能受到各种因素的影响，包括搜索引擎的可用性、网络连接状态和搜索词的质量