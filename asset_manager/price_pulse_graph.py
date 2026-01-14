import yaml
import operator
from pydantic import Field
from typing import Annotated
from typing import List, Literal
from data_models import Asset, AssetPriceInfo
from tools import load_search_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class OverallPricePulseState(MessagesState):
    """
    State model for gathering the price of all assets.
    """

    assets: list[Asset] = Field(description="List of assets to gather prices for.")
    asset_prices: Annotated[list[AssetPriceInfo], operator.add] = Field(
        default=[], description="List of gathered asset price information."
    )


class IndividualPricePulseState(MessagesState):
    """
    State model for gathering the price of a single asset.
    """

    asset: Asset = Field(description="The asset to gather price for.")
    tool_use_count: int = Field(
        default=0, description="Number of times tools have been used."
    )
    asset_prices: Annotated[list[AssetPriceInfo], operator.add] = Field(
        default=[], description="List of gathered asset price information."
    )


class PricePulseGraph:
    """
    Graph to gather the current prices of assets.
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

        self.price_pulse_gather_instructions = instructions_data[
            "price_pulse_gather_instructions"
        ]
        self.price_pulse_summarize_instructions = instructions_data[
            "price_pulse_summarize_instructions"
        ]

        return None

    def load_tools(self) -> None:
        """
        Loads the necessary tools for the agent.
        Returns:
            None
        """
        self.tools = [load_search_tool(time_range="week")]
        return None

    def contruct_graph(self) -> StateGraph:
        """
        Constructs and returns the state graph for gathering asset prices.
        Returns:
            StateGraph: The constructed state graph.
        """

        # Construct individual asset price pulse sub graph
        individual_price_pulse_graph = StateGraph(IndividualPricePulseState)
        individual_price_pulse_graph.add_node(
            "gather_asset_price", self.gather_asset_price
        )
        individual_price_pulse_graph.add_node(
            "summarize_asset_price", self.summarize_asset_price
        )
        individual_price_pulse_graph.add_node("tools", ToolNode(self.tools))
        individual_price_pulse_graph.add_edge(START, "gather_asset_price")
        individual_price_pulse_graph.add_conditional_edges(
            "gather_asset_price", self.is_tool_use_needed
        )
        individual_price_pulse_graph.add_edge("tools", "gather_asset_price")
        individual_price_pulse_graph.add_edge("summarize_asset_price", END)

        # Construct overall price pulse graph
        overall_price_pulse_graph = StateGraph(OverallPricePulseState)
        overall_price_pulse_graph.add_node(
            "individual_asset_price_pulse",
            individual_price_pulse_graph.compile(),
        )
        overall_price_pulse_graph.add_conditional_edges(
            START,
            self.forward_to_asset_price_gather,
            ["individual_asset_price_pulse"],
        )

        return overall_price_pulse_graph

    def forward_to_asset_price_gather(
        self,
        state: OverallPricePulseState,
    ) -> List[Send]:
        """
        Create Send actions to forward each asset to the asset price gathering sub graph.
        Args:
            state (OverallPricePulseState): The current state containing the asset list.
        Returns:
            List[Send]: A list of Send actions, each targeting the "gather_asset_price" node with the state for a specific asset.
        """

        asset_states = []
        for asset in state["assets"]:
            asset_state = IndividualPricePulseState(
                asset=asset,
                tool_use_count=0,
            )
            asset_states.append(asset_state)

        return [
            Send("individual_asset_price_pulse", asset_state)
            for asset_state in asset_states
        ]

    def gather_asset_price(
        self, state: IndividualPricePulseState
    ) -> IndividualPricePulseState:
        """
        Gathers the current price of the assets.
        Args:
            state (IndividualPricePulseState): The current state containing asset information.
        Returns:
            IndividualPricePulseState: The updated state with gathered asset price.
        """

        llm_copy = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        instructions = self.price_pulse_gather_instructions.format(
            asset=state["asset"].asset,
            description=state["asset"].description,
            asset_class=state["asset"].asset_class,
            industry=state["asset"].industry,
            market=state["asset"].market,
            investment_unit=state["asset"].investment_unit,
            currency=state["asset"].currency,
        )
        human_msg = HumanMessage(
            content="Please gather the current unit price information."
        )
        sys_msg = SystemMessage(content=instructions)

        message = llm_copy.invoke([sys_msg] + state["messages"] + [human_msg])

        tool_usage = state["tool_use_count"]
        if message.tool_calls:
            tool_usage = tool_usage + 1

        if tool_usage >= 5:
            message = AIMessage(
                content="Maximum tool usage reached. Proceeding to summarize the current price."
            )

        return {"messages": [message], "tool_use_count": tool_usage}

    def summarize_asset_price(
        self,
        state: IndividualPricePulseState,
    ) -> OverallPricePulseState:
        """
        Summarize the current price of the asset.
        Args:
            state (IndividualPricePulseState): The current state containing relevant information to summarize.
        Returns:
            OverallPricePulseState: Updated state with the summarized current price.
        """

        llm_copy = self.llm.with_structured_output(AssetPriceInfo)
        instructions = self.price_pulse_summarize_instructions.format(
            asset=state["asset"].asset,
            description=state["asset"].description,
            asset_class=state["asset"].asset_class,
            industry=state["asset"].industry,
            market=state["asset"].market,
            investment_unit=state["asset"].investment_unit,
            currency=state["asset"].currency,
        )
        human_msg = HumanMessage(
            content="Please summarize the information you have gathered so far."
        )
        sys_msg = SystemMessage(content=instructions)

        message = llm_copy.invoke([sys_msg] + state["messages"] + [human_msg])

        return {
            "asset_prices": [message],
        }

    def is_tool_use_needed(
        self,
        state: IndividualPricePulseState,
    ) -> Literal["tools", "summarize_asset_price"]:
        """
        Determines whether to use any tool or proceed to next step.
        Args:
            state (IndividualPricePulseState): The current state of the agent.
        Returns:
            Literal: Decision on the next step.
        """

        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"

        return "summarize_asset_price"
