from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Annotated, TypedDict, Literal, Union, Dict, Any
import os
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 定义用于保存planner结果的第一个类
class WebSearchItem(BaseModel):
    """表示研究过程中的一个步骤"""
    reason: str = Field(description="Your reasoning for why this search is important to the query")
    query: str = Field(description="The search term to use for the web search.")


# 定义用于保存planner结果的第二个类
class WebSearchPlan(BaseModel):
    """表示完整的研究计划"""
    searches: List[WebSearchItem] = Field(default_factory=list, description="A list of web searches to perform to best answer the query")


# 创建planner提示词模板
planner_template = """You are a helpful research assistant. Given a query, come up with a set of web searches to perform to best answer the query. Output between 8 and 10 search terms.

Your response MUST be in JSON format with a list of search steps, each containing a 'reason' and a 'query' field. For example:

```json
{{
  "searches": [
    {{
      "reason": "To understand the basic concept",
      "query": "What is AI in education"
    }},
    {{
      "reason": "To learn about current applications",
      "query": "AI applications in modern classrooms"
    }}
  ]
}}
```

Ensure your response follows this exact JSON structure."""


# 创建planner提示词
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", planner_template),
    ("human", "{query}")
])


# 创建search提示词模板
search_template = """You are a helpful research assistant. Given a search item, you search the web for that term and produce a concise summary of the results.

The summary must 5-10 paragraphs and less then 1000 words. Capture the main points. write succinctly, no need to have complete sentences or good grammar.This will be consumed by 

some synthesizing a report, so its vital your capture the essence and ignore any fluff. Do not include any additional commentary other than tge summary."""


# 创建LLM
llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME", "qwen-plus"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 使用TavilySearch搜索网页
search_tool = TavilySearch(max_results=5, topic="general")
# 创建并使用search_agent
search_agent = create_react_agent(model=llm, tools=[search_tool], prompt=search_template)

# 定义用于保存write结果的第一个类
class ReportData(BaseModel):
    """表示研究过程中的一个步骤"""
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings")
    markdown_report: str = Field(description="A markdown final report")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested topics to research further")


# 创建write提示词模板
write_template = """You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research "
    "assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 500 words.\n"
    "Please format your response as JSON. The response should include short_summary, markdown_report, and follow_up_questions fields."
    "最终结果请用中文输出。"""


# 创建planner提示词
write_prompt = ChatPromptTemplate.from_messages([
    ("system", write_template),
    ("human", "{query}")
])

writer_chain = write_prompt | llm.with_structured_output(ReportData)

# 定义状态类型
class ResearchState(TypedDict):
    query: str
    search_plan: Union[WebSearchPlan, None]
    search_results: Union[list, None]
    report_content: Union[str, None]
    status: str

# 定义节点函数
def plan_node(state: ResearchState) -> ResearchState:
    """规划搜索步骤"""
    query = state["query"]
    planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)
    search_plan = planner_chain.invoke({"query": query})
    return {"search_plan": search_plan, "status": "planned"}

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

def report_node(state: ResearchState) -> ResearchState:
    """生成报告"""
    query = state["query"]
    search_results = state["search_results"]
    final_query = f"Original query: {query}\nSummarized search results: {search_results}"
    result = writer_chain.invoke({"query": final_query})
    return {"report_content": result.markdown_report, "status": "completed"}

def build_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("report", report_node)
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "report")
    workflow.add_edge("report", END)
    workflow.set_entry_point("plan")
    return workflow.compile()

# 示例调用
if __name__ == "__main__":
    query = "请问你对AI+教育有何看法?"
    
    research_graph = build_research_graph()
    result = research_graph.invoke({"query": query, "search_plan": None, "search_results": None, "report_content": None, "status": "start"})
    '''
    child_query: AI在教育中的应用
    child_query: AI如何影响现代教学方法
    child_query: AI在个性化学习中的作用
    child_query: AI如何促进教育公平
    child_query: 智能辅导系统的技术基础
    child_query: AI在教育中的伦理挑战
    child_query: AI在语言教育中的应用
    child_query: AI在教育管理中的应用
    '''

    # 输出报告
    markdown_report = result["report_content"]
    print("------------------markdown_report_begin--------------------------")
    print(markdown_report)
    '''
    # AI+教育：智能化转型与未来展望

    ## 一、引言

    随着人工智能（AI）技术的迅猛发展，其在教育领域的应用日益广泛。AI不仅提升了教学效率，还推动了教育模式的深刻变革。从个性化学习到智能评估，从虚拟导师到语言教育，AI正在重塑教育的各个环节。然而，AI在教育中的广泛应用也带来了伦理、公平性、隐私保护等方面的挑战。本文将系统梳理AI在教育中的主要应用场景、技术基础、优势与挑战，并探讨其未来发展趋势。

    ## 二、AI在教育中的主要应用场景

    ### 1. 课程与教案制定
    AI工具如Top Hat、Education Copilot和ChatGPT等，能够帮助教师创建个性化课程和教学计划。这些工具通过分析教学目标、学生水平和学习风格，自动生成教学 内容和教学活动，提高课程设计的效率和质量。

    ### 2. 差异化学习
    AI工具能够分析学生的学习数据，识别其知识掌握情况、学习风格和薄弱环节，从而推荐针对性的学习资源和活动。例如，Squirrel AI能够为学生提供个性化支持 ，帮助他们进行针对性练习，提升学习效果。

    ### 3. 自动化评分与反馈
    AI可以自动批改作业、测验和考试，节省教师时间并减少评分偏差。例如，Quizlet、Turnitin Feedback Studio和Gradescope等工具能够提供即时反馈，帮助学生 及时纠正错误。

    ### 4. 智能导师与学习助手
    虚拟助手和聊天机器人如Tutor.ai和Syntea，能够提供即时解答，帮助学生解决问题。这些工具通过自然语言处理技术，实现与学生的互动交流，提升学习体验。  

    ### 5. 自适应学习平台
    AI平台如Wolfram Alpha、Smart Sparrow和Docebo能够根据学生的学习模式和需求分析，创建个性化学习路径。这种自适应机制提高了学习效率，使学生能够按照自己的节奏进行学习。

    ### 6. 语言学习
    AI在语言学习中的应用尤为突出。Duolingo和Rosetta Stone等应用程序能够根据学习者的水平定制个性化学习计划，并通过语音识别技术评估口语，提供即时纠正 或建议。

    ### 7. 辅助残障学生
    AI技术能够调整界面以适应不同学习障碍，启用语音转文本或文本转语音技术。例如，Microsoft的学习工具和JAWS、Otter.ai提供实时字幕，帮助残障学生更好地 参与学习。

    ### 8. 预测分析与学习干预
    AI驱动的技术能够分析学生数据，提前识别学习挑战并进行干预。例如，通过分析学生的学习行为，AI可以预测可能的学习困难，并提供相应的学习资源或建议。  

    ### 9. 智能考试系统
    AI能够生成试卷、自动批改并提供反馈。例如，AI可以通过拍照识别答卷内容，自动生成总结报告，分析学生掌握情况，提升考试效率。

    ### 10. 个性化在线学习平台
    基于AI的在线学习平台能够根据学生的需求和进度推荐课程和练习题，提高学习效率和个性化体验。例如，Coursera、Udacity等平台结合AI技术，提供定制化的学 习路径。

    ## 三、AI在现代教学方法中的影响

    ### 1. 个性化学习
    AI通过分析学生的学习数据，能够识别其学习风格、兴趣和能力，从而设计出更加个性化和灵活的课程内容。这种“超个性化教育”不仅提高了学生的学习参与度，还增强了他们的学习动机和成果。

    ### 2. 智能辅导系统
    AI可以作为智能辅导工具，提供即时反馈和指导。它能够实时跟踪学生的学习进度，帮助他们及时纠正错误并不断进步。这使得学校教育逐渐打破传统的班级概念，转变为因材施教的个性化教育场所。

    ### 3. 教学资源的丰富
    AI技术通过网络爬虫、自然语言处理等技术，自动搜集、整理和分析各种优质的教学资源，如教材、课件、练习题等，为教师提供多样化的教学素材。这不仅提升了教学质量，也为学生提供了更多的学习内容和方式选择。

    ### 4. 自动化评估与反馈
    AI技术实现了自动化评估与反馈，提高了评估的效率和准确性，减少了教师的工作量。AIGC（人工智能生成内容）可以利用自然语言处理、图像识别等技术，自动快速批改学生的作业、论文和试卷，并给出详细的分级和建议。

    ### 5. 教学方法的创新
    AI技术促进了教学方法的创新，使教学过程更加生动、直观，提高学生的学习兴趣和参与度。新的教育教学方式将层出不穷，为学生提供更加多样化的学习体验。例如，AI与编程教育的结合培养了学生的逻辑思维和创新能力。

    ### 6. 教育渠道的拓展
    AI技术拓展了教育渠道，使学习变得更加灵活、便捷。在线学习平台和移动学习应用程序，以及社交媒体和在线社区等为学生提供了交流、分享和学习的新渠道，促进了师生之间以及学生之间的互动和合作。

    ### 7. 教育理论的创新
    AI与教育的深度融合必然会带来教育理论的创新。从个性化教学条件到利用数据科学指导教学实践，再到强化实践性教学和关注学生情感教育，以及根据学生学习反馈动态调整教学内容等应用实践创新，都将推动新的教育理论形成。

    ## 四、AI在促进教育公平中的作用

    ### 1. 个性化学习路径设计
    AI能够根据每位学生的特点和需求提供定制化教学内容，从而实现教育资源的优化配置。这种技术的应用不仅提升了教学质量，也为不同背景的学生提供了平等的学习机会。

    ### 2. 自动化评估系统
    AI能够快速准确地评价学生的学习成果，并为教师提供数据支持，帮助他们优化教学策略。此外，AI还能为偏远地区的学生提供接触优质教育资源的机会，通过网络平台打破地理限制，实现教育服务的普及化。

    ### 3. 特殊需求学生的支持
    对于特殊需求的学生，AI同样提供了个性化的支持方案，例如为语言学习障碍者提供翻译系统，改善他们的学习体验。这进一步促进了教育公平，确保所有学生都能获得适合自己的教育方式。

    ## 五、AI在教育中的伦理挑战

    ### 1. 数据隐私问题
    AI系统高频采集学生敏感信息，存在泄露风险，特别是面部识别等生物特征数据的非法收集，可能侵犯师生隐私权益。此外，教育数据在政府与企业间的流转缺乏有效监管，资本对数据的垄断加剧了隐私风险。

    ### 2. 算法偏见与歧视
    AI系统的训练数据若包含性别、种族或社会经济地位的偏见，可能放大这些偏见，导致不公平的评估结果。例如，某些AI评分系统可能误判非母语学生的作品为AI生成，引发学术诚信指控，进一步加剧教育不平等。

    ### 3. 情感疏离
    AI的深度应用可能导致教学“去人性化”，师生互动简化为人机交互，削弱教育过程中的情感联结。数字世界的特点可能使学生的个性化成长泡沫化，影响真实情感体验和共鸣的生成。

    ### 4. 信息茧房效应
    算法推荐机制可能导致信息同质化，限制学生获取多元观点，影响其认知发展和自主选择权。学生可能在算法操控的信息洪流中失去批判性思维能力，教育者也可能丧失对教学过程的控制权。

    ### 5. 应对策略
    为应对这些伦理挑战，需构建AI在教育中应用的法律规制体系和协同综合治理体系。技术创新应与人文价值传承平衡，避免技术对教育的负面影响。数据安全措施和隐私保护政策需加强，确保最小化数据收集和知情同意原则的落实。同时，AI系统的透明性和可解释性应提升，减少算法偏见和歧视。

    ## 六、AI在语言教育中的应用

    AI在语言教育中的应用主要体现在提升学习效率和个性化教学上。AI可以通过模拟真实的语言环境，与学习者进行交流，提升口语学习效果。此外，AI还能够根据学习者的水平定制个性化学习计划，如Duolingo和Rosetta Stone等应用程序。AI在语言学习中的另一个应用是通过语音识别技术评估口语，并提供即时纠正或建议。 

    ## 七、AI在教育管理中的应用

    ### 1. 教学内容设计
    AI通过大数据和机器学习，能够个性化和精准化教学资源，帮助教师快速找到符合学生需求的内容，并根据学习进度动态调整难度和深度。

    ### 2. 教学过程管理
    AI技术通过实时跟踪学生的学习进度、时间和状态，为教师提供即时的学习数据反馈，帮助教师了解学生动态，及时发现困难并调整教学策略。

    ### 3. 学习效果评估
    AI技术通过分析学生的学习数据和行为习惯，提供个性化的评估和反馈，涵盖知识掌握情况、学习态度、习惯和能力等多个方面，为教师提供更为全面、准确的教学评估结果。

    ## 八、结论与展望

    AI正在以多种方式改变教育，为学生提供更高效、更有效的学习体验，同时也为教师减轻了工作负担，简化了行政任务。未来，AI与教育的深度融合将推动教育理论创新、教学方法变革，同时需要加强伦理监管与技术普及，确保AI在教育中的公平、公正与可持续发展。

    ## 九、参考文献

    1. AI在教育中的应用概述
    2. AI对现代教学方法的影响
    3. AI在个性化学习中的作用
    4. AI在促进教育公平中的潜力
    5. AI在智能辅导系统中的技术基础
    6. AI在教育中的伦理挑战
    7. AI在语言教育中的应用
    8. AI在教育管理中的应用
    '''
    print("------------------markdown_report_end--------------------------")
