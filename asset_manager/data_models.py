from pydantic import BaseModel, Field


class Asset(BaseModel):
    """
    Represents an asset in the portfolio.
    """

    asset: str = Field(description="Name of the asset")
    description: str = Field(description="Description of the asset")
    asset_class: str = Field(
        description="Class of the asset (e.g., 'Equity', 'Fixed Income')"
    )
    industry: str = Field(description="Industry associated with the asset")
    market: str = Field(description="Market where the asset is traded")
    investment_unit: str = Field(
        description="Unit of investment (e.g., 'share', 'gram')"
    )
    currency: str = Field(description="Currency unit for pricing (e.g., 'USD', 'SGD')")


class AssetPriceInfo(Asset):
    """
    Extends Asset class to include pricing information.
    """

    current_unit_price: float = Field(description="The current unit price of the asset")
    price_source: list[str] = Field(
        description="Links to the source from which the current unit price was obtained"
    )


class AssetNewsInfo(Asset):
    """
    Extends Asset class to include news information.
    """

    news: str = Field(description="Summarized news about the asset")
    news_source: list[str] = Field(
        description="Links to the source from which the news information was obtained"
    )


class FedWatchInfo(BaseModel):
    """
    Represents the information about the Federal Reserve's interest rate decisions.
    """

    current_fed_rate: float = Field(
        description="The current federal funds rate in percentage"
    )
    next_meeting_date: str = Field(description="The date of the next FOMC meeting")
    expected_rate_change: str = Field(
        description="Market expectations for the next rate change (e.g., 'No Change', 'Increase', 'Decrease')"
    )
    personal_finance_impact: str = Field(
        description="Summary of how the current federal funds rate impacts personal finance and investments"
    )
    source: list[str] = Field(
        description="Links to the source from which the federal funds rate information was obtained"
    )
