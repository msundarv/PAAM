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
        description="Source from which the current unit price was obtained"
    )
