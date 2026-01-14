from langchain_tavily import TavilySearch


def load_search_tool(time_range=None) -> TavilySearch:
    """
    Loads the Tavily Search tool with predefined parameters.
    Args:
        time_range (str, optional): Time range for the search. Defaults to None.
    Returns:
        TavilySearch: An instance of the TavilySearch tool.
    """
    tavily_search = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        time_range=time_range,
        include_domains=None,
        exclude_domains=None,
    )
    return tavily_search
