import yaml
from pydantic import Field
from typing import List
from data_models import FedWatchInfo
from tools import load_search_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage


class FedWatchState(MessagesState):
    """
    State model for gathering information about fed rate changes.
    """

    assets: List[str] = Field(
        description="List of assets in the portfolio to consider when summarizing the impact of fed rate changes on personal finance and investments."
    )
    fed_watch_info: FedWatchInfo = Field(description="The fed watch information.")
    basic_search_response: dict | None = Field(
        default=None,
        description="The response from the search tool for basic fed watch information.",
    )
    expectations_search_response: dict | None = Field(
        default=None,
        description="The response from the search tool for market expectations on fed rate changes",
    )


class FedWatchGraph:
    """
    Graph to gather the latest information about fed rate changes.
    """

    def __init__(
        self,
        instructions_yml_file: str = "./instructions.yml",
        model_name: str = "gpt-5-mini",
    ):
        self.load_instructions(instructions_yml_file)
        self.load_tools()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        return None

    def load_instructions(self, instructions_yml_file: str) -> None:
        """
        Loads instructions from a YAML file.
        Args:
            file_path (str): Path to the YAML file containing instructions.
        Returns:
            None
        """

        with open(instructions_yml_file, "r") as f:
            instructions_data = yaml.safe_load(f)

        self.fed_watch_summarize_instructions = instructions_data[
            "fed_watch_summarize_instructions"
        ]

        return None

    def load_tools(self) -> None:
        """
        Loads the necessary tools for the agent.
        Returns:
            None
        """
        self.search_tool = load_search_tool(time_range="week")
        return None

    def contruct_graph(self) -> StateGraph:
        """
        Constructs and returns the state graph for gathering fed rate changes information.
        Returns:
            StateGraph: The constructed state graph.
        """

        fed_watch_graph = StateGraph(FedWatchState)
        fed_watch_graph.add_node("gather_fed_watch_info", self.gather_fed_watch_info)
        fed_watch_graph.add_node(
            "summarize_fed_watch_info", self.summarize_fed_watch_info
        )
        fed_watch_graph.add_edge(START, "gather_fed_watch_info")
        fed_watch_graph.add_edge("gather_fed_watch_info", "summarize_fed_watch_info")
        fed_watch_graph.add_edge("summarize_fed_watch_info", END)

        return fed_watch_graph

    def gather_fed_watch_info(self, state: FedWatchState) -> FedWatchState:
        """
        Gathers the latest inforamtion about the fed rate changes.
        Args:
            state (FedWatchState): Initial state containing fed watch information.
        Returns:
            FedWatchState: The updated state with gathered fed watch information.
        """

        basic_search_query = (
            "When is the next FOMC meeting and what is the current federal funds rate?"
        )
        basic_search_response = self.search_tool.run(basic_search_query)

        expectations_search_query = "What are the market expectations for the next federal funds rate change (e.g., No Change, Increase, Decrease)?"
        expectations_search_response = self.search_tool.run(expectations_search_query)

        return {
            "basic_search_response": basic_search_response,
            "expectations_search_response": expectations_search_response,
        }

    def summarize_fed_watch_info(
        self,
        state: FedWatchState,
    ) -> FedWatchState:
        """
        Summarize the latest information about the fed watch.
        Args:
            state (FedWatchState): The current state containing relevant information to summarize.
        Returns:
            FedWatchState: Updated state with the summarized latest fed watch information.
        """

        llm_copy = self.llm.with_structured_output(FedWatchInfo)
        instructions = self.fed_watch_summarize_instructions.format(
            assets=state["assets"],
            basic_search_result=state["basic_search_response"],
            expectations_search_result=state["expectations_search_response"],
        )
        human_msg = HumanMessage(
            content="Please summarize the information you have gathered so far."
        )
        sys_msg = SystemMessage(content=instructions)

        message = llm_copy.invoke([sys_msg] + state["messages"] + [human_msg])

        return {
            "fed_watch_info": [message],
        }
