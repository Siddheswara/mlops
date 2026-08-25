from __future__ import annotations

from pydantic import BaseModel, Field


class ReturnRequest(BaseModel):
    product_category: str
    sub_category: str
    brand: str
    product_price: float = Field(ge=0)
    discount_percent: float = Field(ge=0, le=100)
    product_rating: float | None = Field(default=None, ge=0, le=5)
    review_count: float = Field(ge=0)
    fragile_item: int = Field(ge=0, le=1)
    warranty_available: int = Field(ge=0, le=1)
    product_return_rate: float = Field(ge=0, le=1)
    category_return_rate: float = Field(ge=0, le=1)
    brand_return_rate: float = Field(ge=0, le=1)
    defect_rate: float = Field(ge=0, le=1)
    seller_rating: float | None = Field(default=None, ge=0, le=5)
    seller_return_rate: float = Field(ge=0, le=1)
    fulfillment_type: str
    payment_method: str
    quantity: float = Field(gt=0)
    shipping_distance_km: float = Field(ge=0)
    delayed_delivery: int = Field(ge=0, le=1)
    wishlist_before_purchase: int = Field(ge=0, le=1)
    product_page_views: float = Field(ge=0)
    customer_support_calls: float = Field(ge=0)
    chat_interactions: float = Field(ge=0)


class PredictionResponse(BaseModel):
    returned: bool
    probability: float
    model: str = "logistic_regression"
