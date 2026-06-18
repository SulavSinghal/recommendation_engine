"""Shared data loading and cleaning for all four engines.

Dataset: UCI "Online Retail II" (CSV). Loads, cleans, and reshapes the
transactions into the formats each engine needs: baskets (Task 1),
user-item matrix (Task 2), and RFM features (Task 3).
"""
import pandas as pd

import config

# Raw -> clean column names, used everywhere downstream.
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# StockCodes that are not real products (postage, fees, adjustments, vouchers).
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "BANK CHARGES", "D", "S",
    "AMAZONFEE", "ADJUST", "ADJUST2", "B", "CRUK", "TEST001", "TEST002",
}


def load_raw() -> pd.DataFrame:
    """Load the raw CSV (add encoding="ISO-8859-1" if you hit a decode error)."""
    return pd.read_csv(config.RAW_FILE)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows (no customer, cancellations, non-positive qty/price) and add line_revenue."""
    df = df.rename(columns=COLUMN_MAP).copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype("int64")
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].str.strip()
    df = df[df["description"] != ""]
    df["line_revenue"] = df["quantity"] * df["price"]
    return df.reset_index(drop=True)


def filter_to_products(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-product lines so association rules cover real products only."""
    code = df["stock_code"].astype(str)
    is_junk = code.isin(NON_PRODUCT_CODES) | code.str.startswith("gift_")
    return df[~is_junk].copy()


def get_baskets(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot basket matrix for Task 1 (rows = invoices, cols = products)."""
    from mlxtend.preprocessing import TransactionEncoder
    df = filter_to_products(df)
    transactions = df.groupby("invoice")["description"].apply(list).tolist()
    te = TransactionEncoder()
    encoded = te.fit_transform(transactions)
    return pd.DataFrame(encoded, columns=te.columns_)


def get_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Customer x product matrix for Task 2 (values = total quantity)."""
    return df.pivot_table(
        index="customer_id",
        columns="stock_code",
        values="quantity",
        aggfunc="sum",
        fill_value=0,
    )


def get_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer RFM features for Task 3 (recency in days, frequency, monetary)."""
    reference_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    return df.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (reference_date - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("line_revenue", "sum"),
    )


if __name__ == "__main__":
    raw = load_raw()
    print("Raw columns:", raw.columns.tolist())
    print("Raw shape:  ", raw.shape)
    clean = clean_transactions(raw)
    print("Clean shape:", clean.shape)
    print("Dropped rows:", len(raw) - len(clean))
    print(clean.head())
