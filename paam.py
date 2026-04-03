import yaml
import random
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from currency_converter import CurrencyConverter, SINGLE_DAY_ECB_URL

from asset_manager.data_models import Asset
from asset_manager.price_pulse_graph import PricePulseGraph, OverallPricePulseState
from asset_manager.asset_news_graph import AssetNewsGraph, OverallAssetNewsState
from asset_manager.fed_watch_graph import FedWatchGraph, FedWatchState


def load_app_config(yml_file_path="config/app_config.yml") -> dict:
    """
    Load application configuration from a YAML file.
    Args:
        yml_file_path (str): The path to the YAML file containing app configuration.
    Returns:
        dict: A dictionary containing the application configuration.
    """

    app_config = None

    # Load app configuration from YAML file
    with open(yml_file_path, "r") as file:
        app_config = yaml.safe_load(file)

    return app_config


def load_assets(yml_file_path="data/my_assets.yml") -> pd.DataFrame:
    """
    Load assets from a YAML file and return them as a DataFrame.
    Args:
        yml_file_path (str): The path to the YAML file containing asset data.
    Returns:
        pd.DataFrame: A DataFrame containing the asset data.
    """

    assets_yml = None
    assets_df = None

    # Load assets from YAML file
    with open(yml_file_path, "r") as file:
        assets_yml = yaml.safe_load(file)

    # Convert YAML data to DataFrame if assets_yml is not empty
    if assets_yml:
        assets_df = (
            pd.DataFrame.from_dict(assets_yml, orient="index")
            .reset_index()
            .rename(columns={"index": "asset_id"})
        )

    return assets_df


def filter_unique_assets(
    assets_df,
    sel_cols=[
        "asset",
        "description",
        "asset_class",
        "industry",
        "market",
        "investment_unit",
        "currency",
    ],
) -> pd.DataFrame:
    """
    Filter unique assets based on selected columns.
    Args:
        assets_df (pd.DataFrame): The DataFrame containing asset data.
        sel_cols (list): List of column names to consider for uniqueness.
    Returns:
        pd.DataFrame: A DataFrame with unique rows based on selected columns.
    """

    if assets_df is None or assets_df.empty:
        return None

    if sel_cols is None or not all(col in assets_df.columns for col in sel_cols):
        raise ValueError(
            "Selected columns must be provided and exist in the DataFrame."
        )

    unique_assets = assets_df.drop_duplicates(
        subset=sel_cols, keep="first"
    ).reset_index(drop=True)
    unique_assets = unique_assets[sel_cols]

    # Aggregate quantity and buy price for duplicate assets
    aggregated = assets_df.groupby(sel_cols, as_index=False).agg(
        {
            "quantity": "sum",
            "buy_price_per_unit": lambda x: (
                x * assets_df.loc[x.index, "quantity"]
            ).sum()
            / assets_df.loc[x.index, "quantity"].sum(),
        }
    )
    unique_assets = unique_assets.merge(
        aggregated[sel_cols + ["quantity", "buy_price_per_unit"]],
        on=sel_cols,
        how="left",
    )

    # Place holder columns for agent results
    unique_assets["current_unit_price"] = "NA"
    unique_assets["price_sources"] = "NA"
    unique_assets["news"] = "NA"
    unique_assets["news_sources"] = "NA"

    return unique_assets


def populate_portfolio_cols(
    data_df,
    quantity_col="quantity",
    buy_price_col="buy_price_per_unit",
    current_price_col="current_unit_price",
) -> pd.DataFrame:
    """
    Populate additional columns to the portfolio DataFrame based on quantity, buy price, and current price.
    Args:
        data_df (pd.DataFrame): The DataFrame to populate.
        quantity_col (str): The name of the column containing quantity information.
        buy_price_col (str): The name of the column containing buy price information.
        current_price_col (str): The name of the column containing current price information.
    Returns:
        pd.DataFrame: The DataFrame with an additional columns.
    """

    if data_df is None or data_df.empty:
        return None

    if not all(
        col in data_df.columns
        for col in [quantity_col, buy_price_col, current_price_col]
    ):
        raise ValueError(
            "Quantity, buy price, and current price columns must exist in the DataFrame."
        )

    data_df["total_cost"] = data_df[quantity_col] * data_df[buy_price_col]
    data_df["market_value"] = data_df[quantity_col] * data_df[current_price_col]
    data_df["unrealized_gain_loss"] = data_df["market_value"] - data_df["total_cost"]
    data_df["unrealized_gain_loss_percent"] = (
        data_df["unrealized_gain_loss"] / data_df["total_cost"]
    ) * 100

    return data_df


def clean_currency_formatting(data_df, clean_cols, currency_col) -> pd.DataFrame:
    """
    Clean currency formatting in the DataFrame.
    Args:
        data_df (pd.DataFrame): The DataFrame to clean.
        clean_cols (list): List of column names to clean.
        currency_col (str): The name of the column containing currency information.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    if data_df is None or data_df.empty:
        return None

    # Create a copy of the df
    res_df = data_df.copy()

    # Merge currency information into the specified columns
    for col in clean_cols:
        if col in res_df.columns:
            res_df[col] = res_df[col].apply(lambda x: f"{x:,.2f}")
            res_df[col] = res_df[col].apply(lambda x: x.replace(".00", ""))
            res_df[col] = res_df[col] + " " + res_df[currency_col]

    # Drop the original currency column after merging
    res_df = res_df.drop(columns=[currency_col])

    return res_df


def clean_percent_formatting(data_df, clean_cols) -> pd.DataFrame:
    """
    Clean percentage formatting in the DataFrame.
    Args:
        data_df (pd.DataFrame): The DataFrame to clean.
        clean_cols (list): List of column names to clean.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    if data_df is None or data_df.empty:
        return None

    # Create a copy of the df
    res_df = data_df.copy()

    # Format percentage columns
    for col in clean_cols:
        if col in res_df.columns:
            res_df[col] = res_df[col].apply(lambda x: f"{x:.0f}%")

    return res_df


def highlight_portfolio_rows(row) -> list:
    """
    Highlight rows in the portfolio DataFrame based on unrealized gain/loss percentage.
    Args:
        row (pd.Series): A row of the DataFrame.
    Returns:
        list: A list of styles to apply to the row.
    """

    # If the agent has not yet populated the unrealized gain/loss percentage, do not apply any highlighting
    value = row["Current Unit Price"]
    if value.split()[0] == "-1":
        return [""] * len(row)

    value = row["Unrealized P/L %"]
    if isinstance(value, str):
        value = float(value.rstrip("%"))
    if value > 0:
        return ["background-color: olivedrab"] * len(row)
    elif value < 0:
        return ["background-color: indianred"] * len(row)
    return [""] * len(row)


def display_portfolio_aggregate() -> None:
    """
    Display aggregated portfolio metrics.
    """

    # Prepare the data for aggregate metrics calculation and visualization
    data_df = st.session_state.get("unique_assets").copy()
    base_currency = st.session_state.get("app_config", {}).get("base_currency", "USD")
    currency_col = "currency"
    value_cols = ["buy_price_per_unit"]
    # Add additional value columns from price pulse agent results if available
    if st.session_state.get("is_pulse_agent_completed"):
        value_cols.extend(["current_unit_price"])

    # Convert all values to base currency using the currency converter if they are not already in base currency
    if data_df is not None and not data_df.empty and currency_col in data_df.columns:
        for idx, row in data_df.iterrows():
            if row.get(currency_col) != base_currency:
                # Convert relevant numeric columns
                for col in value_cols:
                    if col in data_df.columns and pd.notna(row[col]):
                        converted_value = st.session_state["converter"].convert(
                            row[col], row.get(currency_col), base_currency
                        )
                        data_df.at[idx, col] = converted_value
                # Update currency column
                data_df.at[idx, "currency"] = base_currency
    else:
        st.warning(
            "No data available to display aggregated portfolio metrics.", icon="⚠️"
        )
        return None

    # Display warning if the price pulse agent couldn't retrieve current prices for any assets
    if (
        st.session_state.get("is_pulse_agent_completed")
        and (data_df["current_unit_price"] == -1).any()
    ):
        st.warning(
            "Agent was unable to retrieve current prices for some assets. Please note that these assets will not be included in the Total Portfolio Value and Overall P/L % metrics.",
            icon="⚠️",
        )

    st.space(size="small")
    _, col1, _, col2, _ = st.columns([1, 2, 1, 2, 1])
    with col1:
        # Total Cost Metric
        total_cost = (data_df["quantity"] * data_df["buy_price_per_unit"]).sum()
        st.metric(
            "Total Portfolio Cost", f"{total_cost:,.0f} {base_currency}", border=True
        )

        # Total Value Metric
        if st.session_state.get("is_pulse_agent_completed"):
            total_value = (data_df["quantity"] * data_df["current_unit_price"]).sum()
            st.metric(
                "Total Portfolio Value",
                f"{total_value:,.0f} {base_currency}",
                border=True,
            )
        else:
            st.metric("Total Portfolio Value", "NA", border=True)

        # Overall P/L % Metric
        overall_gain_loss_percentage = (
            (total_value - total_cost) / total_cost * 100
            if st.session_state.get("is_pulse_agent_completed") and total_cost != 0
            else None
        )
        st.metric(
            "Overall P/L %",
            (
                f"{overall_gain_loss_percentage:.0f}%"
                if overall_gain_loss_percentage is not None
                else "NA"
            ),
            border=True,
        )

    with col2:

        # Asset class distribution pie chart
        if not data_df.empty:
            asset_class_distribution = (
                data_df.groupby("asset_class")
                .apply(lambda x: (x["quantity"] * x["buy_price_per_unit"]).sum())
                .reset_index(name="cost")
            )
            fig, ax = plt.subplots()
            ax.figure.set_size_inches(2, 2)
            ax.pie(
                asset_class_distribution["cost"],
                labels=asset_class_distribution["asset_class"],
                autopct="%1.1f%%",
                colors=plt.get_cmap("tab20").colors[: len(asset_class_distribution)],
                textprops={"fontsize": 8},
            )
            ax.axis("equal")
            st.pyplot(fig, use_container_width=True)

    st.space(size="small")

    return None


def display_portfolio() -> None:
    """
    Display the portfolio of assets.
    """

    # Display aggregated portfolio metrics
    display_portfolio_aggregate()

    # Clean up the DataFrame
    # Populate additional columns if price pulse agent has completed
    # Display the formatted portfolio
    portfolio_df = st.session_state.get("unique_assets")
    portfolio_df = portfolio_df[
        [
            "asset",
            "asset_class",
            "quantity",
            "investment_unit",
            "buy_price_per_unit",
            "current_unit_price",
            "currency",
        ]
    ]
    if st.session_state.get("is_pulse_agent_completed"):
        portfolio_df = populate_portfolio_cols(portfolio_df)
        portfolio_df = clean_currency_formatting(
            portfolio_df,
            clean_cols=[
                "buy_price_per_unit",
                "current_unit_price",
                "total_cost",
                "market_value",
                "unrealized_gain_loss",
            ],
            currency_col="currency",
        )
        portfolio_df = clean_percent_formatting(
            portfolio_df,
            clean_cols=[
                "unrealized_gain_loss_percent",
            ],
        )
        portfolio_df = portfolio_df.rename(
            columns={
                "asset": "Asset",
                "asset_class": "Asset Class",
                "quantity": "Quantity",
                "investment_unit": "Investment Unit",
                "buy_price_per_unit": "Average Unit Buy Price",
                "current_unit_price": "Current Unit Price",
                "total_cost": "Total Cost",
                "market_value": "Market Value",
                "unrealized_gain_loss": "Unrealized P/L",
                "unrealized_gain_loss_percent": "Unrealized P/L %",
            }
        )
        st.dataframe(
            portfolio_df.style.apply(highlight_portfolio_rows, axis=1),
            hide_index=True,
            width="stretch",
        )
    else:
        portfolio_df = clean_currency_formatting(
            portfolio_df,
            clean_cols=[
                "buy_price_per_unit",
            ],
            currency_col="currency",
        )
        portfolio_df = portfolio_df.rename(
            columns={
                "asset": "Asset",
                "asset_class": "Asset Class",
                "quantity": "Quantity",
                "investment_unit": "Investment Unit",
                "buy_price_per_unit": "Average Unit Buy Price",
                "current_unit_price": "Current Unit Price",
            }
        )
        st.dataframe(
            portfolio_df,
            hide_index=True,
            width="stretch",
        )

    # Display price sources
    with st.expander("📝 See Price Sources", expanded=False):
        price_sources_df = (
            st.session_state["unique_assets"][["asset", "price_sources"]]
            .rename(columns={"asset": "Asset", "price_sources": "Price Sources"})
            .set_index("Asset")
        )
        st.table(price_sources_df)

    return None


def load_price_pulse_state() -> OverallPricePulseState:
    """
    Load the state for the price pulse agent.
    Returns:
        OverallPricePulseState: Initial state of the price pulse agent.
    """

    unique_assets = st.session_state.get("unique_assets")

    asset_list = []
    for row in unique_assets.to_dict(orient="records"):
        try:
            asset_list.append(Asset(**row))
        except TypeError:
            asset_list.append(row)

    return OverallPricePulseState(assets=asset_list)


def get_current_value() -> None:
    """
    Run price pulse agent to get current prices and calculate current portfolio value.
    """

    price_pulse_result = None

    if st.button("Run Price Pulse Agent", icon="▶️"):

        with st.spinner("Running Price Pulse Agent..."):
            try:
                price_pulse_state = load_price_pulse_state()
                price_pulse_graph = PricePulseGraph().contruct_graph().compile()
                thread = {"configurable": {"thread_id": random.randint(0, 9999)}}

                # Run the price pulse agent and stream results
                for chunk in price_pulse_graph.stream(
                    price_pulse_state, thread=thread, stream_mode="values"
                ):
                    if "asset_prices" in chunk:
                        price_pulse_result = chunk["asset_prices"]

                # Populate the price pulse results
                if price_pulse_result:

                    for asset_price in price_pulse_result:
                        st.session_state["unique_assets"].loc[
                            st.session_state["unique_assets"]["asset"]
                            == asset_price.asset,
                            "current_unit_price",
                        ] = asset_price.current_unit_price
                        st.session_state["unique_assets"].loc[
                            st.session_state["unique_assets"]["asset"]
                            == asset_price.asset,
                            "price_sources",
                        ] = ", ".join(asset_price.price_source)

                        st.session_state["is_pulse_agent_completed"] = True

                    # Replace if any current unit price is still NA with '-1' after agent run with buy price as a fallback
                    st.session_state["unique_assets"][
                        "current_unit_price"
                    ] = st.session_state["unique_assets"].apply(
                        lambda row: (
                            -1
                            if row["current_unit_price"] == "NA"
                            else row["current_unit_price"]
                        ),
                        axis=1,
                    )

            except Exception as e:
                st.error(f"Error running Price Pulse Agent: {e}", icon="🚨")

    return None


def load_asset_news_state() -> OverallAssetNewsState:
    """
    Load the state for the asset news agent.
    Returns:
        OverallAssetNewsState: Initial state of the asset news agent.
    """

    unique_assets = st.session_state.get("unique_assets")

    asset_list = []
    for row in unique_assets.to_dict(orient="records"):
        try:
            asset_list.append(Asset(**row))
        except TypeError:
            asset_list.append(row)

    return OverallAssetNewsState(assets=asset_list)


def get_latest_asset_news() -> None:
    """
    Run asset news agent to get latest news for each asset.
    """

    asset_news_result = None

    if st.button("Run Asset News Agent", icon="▶️"):

        with st.spinner("Running Asset News Agent..."):
            try:
                asset_news_state = load_asset_news_state()
                asset_news_graph = AssetNewsGraph().contruct_graph().compile()
                thread = {"configurable": {"thread_id": random.randint(0, 9999)}}

                # Run the asset news agent and stream results
                for chunk in asset_news_graph.stream(
                    asset_news_state, thread=thread, stream_mode="values"
                ):
                    if "asset_news" in chunk:
                        asset_news_result = chunk["asset_news"]

                # Populate the asset news results
                if asset_news_result:
                    for asset_news in asset_news_result:
                        st.session_state["unique_assets"].loc[
                            st.session_state["unique_assets"]["asset"]
                            == asset_news.asset,
                            "news",
                        ] = asset_news.news
                        st.session_state["unique_assets"].loc[
                            st.session_state["unique_assets"]["asset"]
                            == asset_news.asset,
                            "news_sources",
                        ] = ", ".join(asset_news.news_source)

                        st.session_state["is_asset_news_agent_completed"] = True
            except Exception as e:
                st.error(f"Error running Asset News Agent: {e}", icon="🚨")

    return None


def display_news() -> None:
    """
    Display the latest news for each asset.
    """

    if st.session_state.get("is_asset_news_agent_completed"):

        news_df = (
            st.session_state["unique_assets"][["asset", "news"]]
            .rename(
                columns={
                    "asset": "Asset",
                    "news": "Latest News",
                }
            )
            .set_index("Asset")
        )
        st.table(news_df)

        with st.expander("📝 See News Sources", expanded=False):
            news_sources_df = (
                st.session_state["unique_assets"][["asset", "news_sources"]]
                .rename(
                    columns={
                        "asset": "Asset",
                        "news_sources": "News Sources",
                    }
                )
                .set_index("Asset")
            )
            st.table(news_sources_df)

    return None


def get_fed_watch_result() -> None:
    """
    Run fed watch agent to get the latest fed watch result.
    """

    fed_watch_result = None

    if st.button("Run Fed Watch Agent", icon="▶️"):

        with st.spinner("Running Fed Watch Agent..."):
            try:
                fed_watch_state = FedWatchState(
                    assets=st.session_state.get("unique_assets")["asset"].tolist()
                )
                fed_watch_graph = FedWatchGraph().contruct_graph().compile()
                thread = {"configurable": {"thread_id": random.randint(0, 9999)}}

                # Run the fed watch agent and stream results
                for chunk in fed_watch_graph.stream(
                    fed_watch_state, thread=thread, stream_mode="values"
                ):
                    if "fed_watch_info" in chunk:
                        fed_watch_result = chunk["fed_watch_info"][0]

                # Populate the fed watch result
                if fed_watch_result:
                    st.session_state["fed_watch_result"] = fed_watch_result

            except Exception as e:
                st.error(f"Error running Fed Watch Agent: {e}", icon="🚨")

    return None


def display_fed_watch_result() -> None:
    """
    Display the latest fed watch result.
    """

    if st.session_state.get("fed_watch_result"):

        fed_watch_info = st.session_state["fed_watch_result"]
        impact_text = fed_watch_info.personal_finance_impact
        impact_lines = impact_text.split("\n")
        if len(impact_lines) > 1:
            impact_text = (
                impact_lines[0]
                + "\n"
                + "\n".join(["> " + line for line in impact_lines[1:]])
            )

        st.markdown(
            f"""
            - ***Current Fed Rate***: {fed_watch_info.current_fed_rate}
            
            \n- ***Next Meeting Date***: {fed_watch_info.next_meeting_date}
            
            \n- ***Expected Rate Change***: {fed_watch_info.expected_rate_change}
            
            \n- ***Impact***:\n> {impact_text}
            """
        )

    with st.expander("📝 See Fed Watch Sources", expanded=False):
        if st.session_state.get("fed_watch_result"):
            fed_watch_info = st.session_state["fed_watch_result"]
            sources = fed_watch_info.source
            sources_md = "\n".join([f"- {source}" for source in sources])
            st.markdown(sources_md)

    return None


st.set_page_config(page_title="PAAM", layout="wide", page_icon="🤖")
st.title("PAAM - Personal AI Asset Manager")

# Load page level data into session state if not already loaded
if st.session_state.get("assets") is None:

    load_dotenv()

    try:
        st.session_state["app_config"] = load_app_config()
        st.session_state["converter"] = CurrencyConverter(SINGLE_DAY_ECB_URL)
        st.session_state["assets"] = load_assets()
        st.session_state["unique_assets"] = filter_unique_assets(
            st.session_state["assets"]
        )
        st.session_state["is_pulse_agent_completed"] = False
        st.session_state["is_asset_news_agent_completed"] = False
        st.session_state["fed_watch_result"] = False
    except Exception as e:
        st.error(f"Error processing assets data: {e}", icon="🚨")

if st.session_state.get("assets") is not None:

    # Portfolio Overview
    st.subheader("📊 Portfolio Overview", divider="blue")
    get_current_value()
    display_portfolio()

    # Asset News
    st.subheader("📰 Latest Asset News", divider="blue")
    get_latest_asset_news()
    display_news()

    # Fed Watch
    st.subheader("🏦 Fed Watch", divider="blue")
    get_fed_watch_result()
    display_fed_watch_result()

else:
    st.warning("No assets data available to display.", icon="⚠️")
