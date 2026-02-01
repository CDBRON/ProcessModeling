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

   
    client = TavilyClient(api_key=tavily_api_key)

    try:
       
        response = client.search(
            query=query,
            search_depth="advanced",
            include_answer="advanced",
            max_results=3
        )
    except Exception as e:
        return f"Tavily search failed: {e}"
    print(response)


    ai_generated_answer = response.get('answer', 'No direct answer was generated.')
    print(f"AI-Generated Answer: {ai_generated_answer}")

 
    results = response.get('results', [])
    source_snippets = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            snippet = result.get('content', 'No snippet available.')
            source_snippets.append(f"Source {i + 1} Snippet: {snippet}")

   
    final_output = (
            f"**AI-Generated Summary Answer:**\n{ai_generated_answer}\n\n"
            f"--- \n"
            f"**Supporting Source Snippets:**\n" + "\n---\n".join(source_snippets)
    )

    return final_output
