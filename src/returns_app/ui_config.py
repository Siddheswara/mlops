"""Form options and published model-card metrics for the prediction UI."""

from __future__ import annotations

# Holdout scores from project.ipynb (logistic regression on the notebook test split).
MODEL_NAME = "logistic_regression"
MODEL_LABEL = "Logistic Regression"
MODEL_METRICS = {
    "accuracy": 0.8558,
    "precision": 0.7194,
    "recall": 0.8615,
    "f1": 0.7840,
    "source": "Notebook holdout test set",
}

CATEGORICAL_OPTIONS = {
    "product_category": [
        "Beauty",
        "Books",
        "Electronics",
        "Fashion",
        "Home",
        "Sports",
        "Toys",
    ],
    "sub_category_by_category": {
        "Beauty": ["Haircare", "Makeup", "Skincare"],
        "Books": ["Academic", "Fiction", "Non-Fiction"],
        "Electronics": ["Audio", "Laptops", "Mobiles"],
        "Fashion": ["Accessories", "Footwear", "Kids", "Men", "Women"],
        "Home": ["Decor", "Furniture", "Kitchen"],
        "Sports": ["Fitness", "Outdoor", "Team Sports"],
        "Toys": ["Action Figures", "Educational", "Outdoor Play"],
    },
    "brand": [f"Brand_{i}" for i in range(1, 31)],
    "fulfillment_type": ["Marketplace Fulfilled", "Seller Fulfilled"],
    "payment_method": ["COD", "Credit Card", "Debit Card", "UPI", "Wallet"],
}

FORM_DEFAULTS = {
    "product_category": "Fashion",
    "sub_category": "Women",
    "brand": "Brand_1",
    "product_price": 597.47,
    "discount_percent": 33.5,
    "product_rating": 3.9,
    "review_count": 69,
    "fragile_item": 0,
    "warranty_available": 1,
    "product_return_rate": 0.134,
    "category_return_rate": 0.12,
    "brand_return_rate": 0.135,
    "defect_rate": 0.051,
    "seller_rating": 4.1,
    "seller_return_rate": 0.121,
    "fulfillment_type": "Marketplace Fulfilled",
    "payment_method": "COD",
    "quantity": 3,
    "shipping_distance_km": 715.2,
    "delayed_delivery": 0,
    "wishlist_before_purchase": 0,
    "product_page_views": 12,
    "customer_support_calls": 0,
    "chat_interactions": 0,
}
