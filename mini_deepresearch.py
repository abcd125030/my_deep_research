from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import List
from langchain_tavily import TavilySearch
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


#
def _plan_searches(query: str) -> WebSearchPlan:
    planner_chain = planner_prompt | llm.with_structured_output(WebSearchPlan)
    planner_result = planner_chain.invoke({"query": query})
    return planner_result

#
def _perform_searches(search_plan: WebSearchPlan):
    results = []
    for item in search_plan.searches:
        print(f"child_query: {item.query}")
        result = _search(item)
        if result is not None:
            results.append(result)
    return result

#
def _search(item: WebSearchItem) -> str | None:
    try:
        final_query = f"Search term: {item.query}\nReason for searching: {item.reason}"
        result = search_agent.invoke({"messages": [{"role": "user", "content": final_query}]})
        return str(result['messages'][-1].content)
    except Exception:
        return None

#
def _write_report(query: str, search_results) -> ReportData:
    final_query = f"Original query: {query}\nSummarized search results: {search_results}"
    result = writer_chain.invoke({"query": final_query})
    return result.markdown_report


# 示例调用
if __name__ == "__main__":
    query = "请问你对AI+教育有何看法?"
    planner_result = _plan_searches(query)
    search_results = _perform_searches(planner_result)
    markdown_report = _write_report(query, search_results)
    print("------------------markdown_report_begin--------------------------")
    print(markdown_report)
    print("------------------markdown_report_end--------------------------")

    '''
    # AI+教育：未来趋势、应用场景与挑战分析

    ## 一、引言

    随着人工智能（AI）技术的快速发展，教育领域正经历深刻的变革。AI不仅改变了教学内容的呈现方式，也重塑了学习过程、评估机制和教育公平性。特别是在生成式AI、大语言模型、语音识别等技术不断成熟的背景下，AI+教育正成为教育创新的重要方向。本文将从AI+教育的发展趋势、应用场景、技术挑战、伦理问题等多个维度进行深入探讨。

    ## 二、AI+教育的发展前景

    ### 1. 产业规模与市场潜力

    据2023年数据显示，中国人工智能产业规模已达到2137亿元，预计到2028年将突破8110亿元。AI技术在教育领域的应用，尤其是大语言模型和语音大模型的应用，正在加速拓展。生成式AI产品的出现，重新引发了产学研界对AI教育的关注，显示出其在知识传授、学习辅助、智能评估等方面的重要价值。

    ### 2. 教育供给端的重构

    AI技术正在推动教育供给侧的重构。传统教育中教师资源有限、教学内容固化等问题，有望通过AI实现个性化、智能化教学。例如，AI可以根据学生的学习习惯、认知水平、兴趣爱好等数据，提供定制化的学习路径和内容推荐。

    ## 三、AI+教育的应用场景

    ### 1. 职业教育与企业培训

    当前我国职业教育仍以理论授课为主，缺乏实操训练。AI技术的引入，有望实现“1V1实时虚拟师傅”模式，为学生提供个性化的操作指导和实时反馈。此外，在企业 培训领域，AI可以多维度分析员工潜力，提升企业人才管理效率，同时为员工提供更公平的发展机会。

    ### 2. 学科教育与素质教育

    在学科教育方面，AI不受思维惯性限制，可以为学生提供多元化的引导思路。而在素质教育领域，AI可通过收集学生的成长数据，实现过程性评估与个性化陪伴，提升教育的灵活性与温度。

    ### 3. 情绪智能与心理健康

    AI在情绪理解与陪伴方面的能力也在不断提升。未来，AI有望在儿童成长过程中提供情感支持、心理疏导等服务，真正实现“多维一体”的成长陪伴。

    ## 四、AI+教育的深度挑战

    ### 1. 算法黑箱与教育公平

    AI教育系统中存在“算法黑箱”问题，即模型的决策过程不透明，导致部分使用者对其持怀疑态度。这种不透明性可能加剧教育不公平现象。未来需通过算法可解释性提升、训练数据多样性增强等方式，建立更具信任感的AI教育系统。

    ### 2. 技术适用性与数据稀疏性

    AI在教育场景中的应用往往受限于学习数据的稀疏性，导致个性化推荐不够精准，甚至出现以偏概全的现象。此外，AI模型在复杂多变的教育场景中，往往难以满足实际教学需求。

    ### 3. 教育本质与人文关怀的缺失

    AI虽然在知识传授方面表现出色，但在情感交流、价值引导、批判性思维培养等方面仍存在局限。大学生的社会性与情感性发展无法完全依赖AI，自然人教师在教育中仍具有不可替代的作用。

    ## 五、AI时代教师的新使命

    在AI时代，教师的角色将从“知识传授者”转变为“学习引导者”和“技术批判者”。教师需要为学生理解技术本质，发展对AI的批判性思维，帮助学生在AI辅助下保持独立思考能力。

    ## 六、教育公平与AI的助力

    AI技术打破了地理与时间的限制，使得偏远地区的学生也能享受到优质教育资源。通过教学直播、录播、回放等方式，教师的教学不再受限于物理空间，极大促进了教育公平的实现。

    ## 七、未来展望与建议

    ### 1. 技术与教育深度融合

    未来AI+教育的发展方向应是技术与教育理念的深度融合，推动教育范式的根本性变革。教育系统应构建以学生为中心、数据驱动、持续评估新型教学模式。     

    ### 2. 建立AI教育伦理与监管机制

    为保障AI教育健康发展，需健全的伦理规范与监管机制，防范算法歧视、数据滥用等问题，确保AI教育系统的公平性与透明度。

    ### 3. 强化教师培训与技术素养提升

    教师应成为AI教育的引导者与监督者。因此，必须加强对教师的技术培训，提升其对AI工具的理解与使用能力，使其能够在AI辅助下更好地完成教育使命。        

    ## 八、结语

    AI+教育正站在技术变革与教育创新的交汇点上。虽然AI在教育领域展现出巨大潜力，但其发展仍需谨慎推进，注重技术与人文的平衡、公平与效率的统一。只有在 多方努力下，AI才能真正成为推动教育公平、提升教育质量、促进个体全面发展的有力工具。

    ## 九、参考文献

    - 中国人工智能产业发展报告（2023）
    - 教育部《人工智能+教育发展白皮书》
    - 国际教育技术协会（ISTE）AI教育应用指南
    - IEEE关于AI在教育中的伦理与监管研究
    - 多所高校关于AI辅助教学的实证研究

    ## 十、附录

    - AI+教育典型应用场景案例分析
    - 国内外AI教育平台功能对比
    - 教师AI素养培训课程设计建议
    - AI教育伦理准则草案

    ---

    > 本文为AI+教育领域的综合研究报告，旨在为教育从业者、政策制定者、技术开发者提供参考与思考方向。
    '''





  

