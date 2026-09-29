from typing import Any, Dict

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    
    documents.append(web_results)
    
    return {"documents": documents, "question": question}
