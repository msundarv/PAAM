import yaml
import operator
from pathlib import Path
from pydantic import Field
from typing import Annotated
from typing import List
from .data_models import Asset, AssetNewsInfo
from .tools import load_search_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


class OverallAssetNewsState(MessagesState):
    """
    State model for gathering the latest news about all assets.
    """

    assets: list[Asset] = Field(description="List of assets to gather news for.")
    asset_news: Annotated[list[AssetNewsInfo], operator.add] = Field(
        default=[], description="List of gathered asset news information."
    )


class IndividualAssetNewsState(MessagesState):
    """
    State model for gathering the news about a single asset.
    """

    asset: Asset = Field(description="The asset to gather news for.")
    search_response: dict | None = Field(
        default=None, description="The response from the search tool."
    )
    asset_news: Annotated[list[AssetNewsInfo], operator.add] = Field(
        default=[], description="List of gathered asset news information."
    )


class AssetNewsGraph:
    """
    Graph to gather the latest news about assets.
    """

    def __init__(
        self,
        instructions_yml_file: str = CONFIG_DIR / "instructions.yml",
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

        self.asset_news_summarize_instructions = instructions_data[
            "asset_news_summarize_instructions"
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
        Constructs and returns the state graph for gathering asset news.
        Returns:
            StateGraph: The constructed state graph.
        """

        # Construct individual asset news sub graph
        individual_asset_news_graph = StateGraph(IndividualAssetNewsState)
        individual_asset_news_graph.add_node(
            "gather_asset_news", self.gather_asset_news
        )
        individual_asset_news_graph.add_node(
            "summarize_asset_news", self.summarize_asset_news
        )
        individual_asset_news_graph.add_edge(START, "gather_asset_news")
        individual_asset_news_graph.add_edge(
            "gather_asset_news", "summarize_asset_news"
        )
        individual_asset_news_graph.add_edge("summarize_asset_news", END)

        # Construct overall asset news graph
        overall_asset_news_graph = StateGraph(OverallAssetNewsState)
        overall_asset_news_graph.add_node(
            "individual_asset_news",
            individual_asset_news_graph.compile(),
        )
        overall_asset_news_graph.add_conditional_edges(
            START,
            self.forward_to_asset_news_gather,
            ["individual_asset_news"],
        )

        return overall_asset_news_graph

    def forward_to_asset_news_gather(
        self,
        state: OverallAssetNewsState,
    ) -> List[Send]:
        """
        Create Send actions to forward each asset to the asset news gathering sub graph.
        Args:
            state (OverallAssetNewsState): The current state containing the asset list.
        Returns:
            List[Send]: A list of Send actions, each targeting the "gather_asset_news" node with the state for a specific asset.
        """

        asset_states = []
        for asset in state["assets"]:
            asset_state = IndividualAssetNewsState(
                asset=asset,
                search_response=None,
            )
            asset_states.append(asset_state)

        return [
            Send("individual_asset_news", asset_state) for asset_state in asset_states
        ]

    def gather_asset_news(
        self, state: IndividualAssetNewsState
    ) -> IndividualAssetNewsState:
        """
        Gathers the latest news about the asset.
        Args:
            state (IndividualAssetNewsState): The current state containing asset information.
        Returns:
            IndividualAssetNewsState: The updated state with gathered asset news.
        """

        search_query = "What is the latest news about " + state["asset"].asset
        search_response = self.search_tool.run(search_query)

        return {"search_response": search_response}

    def summarize_asset_news(
        self,
        state: IndividualAssetNewsState,
    ) -> OverallAssetNewsState:
        """
        Summarize the latest news about the asset.
        Args:
            state (IndividualAssetNewsState): The current state containing relevant information to summarize.
        Returns:
            OverallAssetNewsState: Updated state with the summarized latest news.
        """

        llm_copy = self.llm.with_structured_output(AssetNewsInfo)
        instructions = self.asset_news_summarize_instructions.format(
            asset=state["asset"].asset,
            description=state["asset"].description,
            asset_class=state["asset"].asset_class,
            industry=state["asset"].industry,
            market=state["asset"].market,
            investment_unit=state["asset"].investment_unit,
            currency=state["asset"].currency,
            search_result=state["search_response"],
        )
        human_msg = HumanMessage(
            content="Please summarize the information you have gathered so far."
        )
        sys_msg = SystemMessage(content=instructions)

        message = llm_copy.invoke([sys_msg] + state["messages"] + [human_msg])

        return {
            "asset_news": [message],
        }
