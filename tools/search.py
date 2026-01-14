from langchain_core.tools import tool
from tavily import TavilyClient
from config.settings import Config

config = Config()

@tool
def tavily_wrapper(query: str) -> str:
    """
    A search engine that also provides a comprehensive, AI-generated answer based on the top search results.
    Use this as your primary tool to get a detailed, synthesized overview of a topic.
    Input should be a search query string.
    The tool returns a structured response containing both a direct answer and a list of source snippets.
    The output of this tool is the final answer for the research task.
    """
    print(f"\n--- [Tool Wrapper] Calling TavilySearch with query: {query} ---\n")

    # 直接使用 TavilyClient，因为它能让我们轻松访问 'answer' 字段
    client = TavilyClient(api_key=tavily_api_key)

    try:
        # 使用 include_answer="advanced" 来获取综合性答案
        response = client.search(
            query=query,
            search_depth="advanced",
            include_answer="advanced",
            max_results=3
        )
    except Exception as e:
        return f"Tavily search failed: {e}"
    print(response)

    # 提取 'answer'
    ai_generated_answer = response.get('answer', 'No direct answer was generated.')
    print(f"AI-Generated Answer: {ai_generated_answer}")

    # 提取 'results' 中的摘要
    results = response.get('results', [])
    source_snippets = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            snippet = result.get('content', 'No snippet available.')
            source_snippets.append(f"Source {i + 1} Snippet: {snippet}")

    # 将 'answer' 和摘要组合成一个结构化的、信息丰富的字符串
    final_output = (
            f"**AI-Generated Summary Answer:**\n{ai_generated_answer}\n\n"
            f"--- \n"
            f"**Supporting Source Snippets:**\n" + "\n---\n".join(source_snippets)
    )

    return final_output