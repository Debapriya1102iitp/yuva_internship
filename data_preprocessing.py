"""
Week 2 - Logistics Data Collection, Cleaning and Preprocessing
Project: Yuva Internship - Logistics Data Analytics

This script demonstrates a reproducible preprocessing workflow using
the Brazilian E-Commerce Public Dataset by Olist.

Expected raw files:
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_customers_dataset.csv
    olist_sellers_dataset.csv
    olist_products_dataset.csv

Place the CSV files in a local data/raw/ directory before execution.
The full public dataset should not be uploaded to GitHub.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# 2. Data Collection / Loading
# -------------------------------------------------------------------

def load_data():
    """Load the main Olist tables from the raw-data directory."""
    files = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "products": "olist_products_dataset.csv",
    }

    tables = {}

    for name, filename in files.items():
        path = RAW_DIR / filename

        if not path.exists():
            print(f"[WARNING] Missing file: {path}")
            continue

        tables[name] = pd.read_csv(path)
        print(f"{name:12} -> {tables[name].shape}")

    return tables


# -------------------------------------------------------------------
# 3. Initial Data Quality Profiling
# -------------------------------------------------------------------

def profile_data(tables):
    """Display basic data-quality information for every loaded table."""
    print("\n========== DATA QUALITY PROFILE ==========")

    for name, df in tables.items():
        print(f"\n--- {name.upper()} ---")
        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("Duplicate rows:", df.duplicated().sum())

        missing = (
            df.isna()
            .sum()
            .sort_values(ascending=False)
        )

        print("Top missing-value counts:")
        print(missing.head(10))


# -------------------------------------------------------------------
# 4. Timestamp Cleaning
# -------------------------------------------------------------------

def clean_order_dates(orders):
    """Convert order timestamp columns to Pandas datetime."""
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(
                orders[column],
                errors="coerce"
            )

    return orders


# -------------------------------------------------------------------
# 5. Duplicate Handling
# -------------------------------------------------------------------

def remove_exact_duplicates(tables):
    """Remove exact duplicate rows while preserving table-specific grain."""
    cleaned = {}

    for name, df in tables.items():
        before = len(df)
        df = df.drop_duplicates().copy()
        after = len(df)

        print(
            f"{name:12}: removed {before - after} "
            f"exact duplicate rows"
        )

        cleaned[name] = df

    return cleaned


# -------------------------------------------------------------------
# 6. Categorical Standardization
# -------------------------------------------------------------------

def standardize_categories(customers, sellers):
    """Standardize common geographic text fields."""
    for df, city_col, state_col in [
        (customers, "customer_city", "customer_state"),
        (sellers, "seller_city", "seller_state"),
    ]:
        if city_col in df.columns:
            df[city_col] = (
                df[city_col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

        if state_col in df.columns:
            df[state_col] = (
                df[state_col]
                .astype("string")
                .str.strip()
                .str.upper()
            )

    return customers, sellers


# -------------------------------------------------------------------
# 7. Missing-Value Logic
# -------------------------------------------------------------------

def create_delivery_status(orders):
    """
    Create an explicit delivery indicator.

    A missing actual delivery date is not automatically imputed,
    because it can represent an undelivered or unrecorded order.
    """
    orders["is_delivered"] = (
        orders["order_delivered_customer_date"].notna()
    ).astype(int)

    return orders


# -------------------------------------------------------------------
# 8. Feature Engineering
# -------------------------------------------------------------------

def create_delivery_features(orders):
    """Create logistics-focused time and delay features."""

    delivered = orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()

    delivered["lead_time_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    delivered["late_flag"] = (
        delivered["delay_days"] > 0
    ).astype(int)

    delivered["purchase_month"] = (
        delivered["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    delivered["purchase_weekday"] = (
        delivered["order_purchase_timestamp"]
        .dt.day_name()
    )

    return delivered


# -------------------------------------------------------------------
# 9. Outlier Detection
# -------------------------------------------------------------------

def iqr_bounds(series):
    """Return lower and upper IQR bounds."""
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


def find_outliers(df, column):
    """Identify potential statistical outliers using the IQR rule."""
    if column not in df.columns:
        return pd.DataFrame()

    lower, upper = iqr_bounds(df[column])

    return df[
        (df[column] < lower) |
        (df[column] > upper)
    ].copy()


# -------------------------------------------------------------------
# 10. Domain Validation
# -------------------------------------------------------------------

def validate_numeric_values(items):
    """Check for obviously invalid negative logistics values."""
    checks = {}

    if "price" in items.columns:
        checks["negative_price"] = int(
            (items["price"] < 0).sum()
        )

    if "freight_value" in items.columns:
        checks["negative_freight"] = int(
            (items["freight_value"] < 0).sum()
        )

    return checks


# -------------------------------------------------------------------
# 11. Freight Feature
# -------------------------------------------------------------------

def create_freight_ratio(items):
    """Create freight-to-price ratio where price is non-zero."""
    items = items.copy()

    if {"freight_value", "price"}.issubset(items.columns):
        items["freight_ratio"] = np.where(
            items["price"] > 0,
            items["freight_value"] / items["price"],
            np.nan,
        )

    return items


# -------------------------------------------------------------------
# 12. Scaling Example
# -------------------------------------------------------------------

def scale_numeric_features(df, columns):
    """
    Standardize numeric features.

    In a machine-learning project, fit the scaler only on the
    training set and use transform() on validation/test data.
    """
    available = [c for c in columns if c in df.columns]

    if not available:
        return None, None

    working = df[available].copy()
    working = working.fillna(working.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(working)

    scaled_df = pd.DataFrame(
        scaled,
        columns=available,
        index=df.index,
    )

    return scaled_df, scaler


# -------------------------------------------------------------------
# 13. KPI Summary
# -------------------------------------------------------------------

def calculate_kpis(delivered):
    """Calculate basic delivery-performance KPIs."""
    if delivered.empty:
        return {}

    on_time_rate = (
        1 - delivered["late_flag"].mean()
    ) * 100

    late_rate = delivered["late_flag"].mean() * 100
    avg_lead_time = delivered["lead_time_days"].mean()

    return {
        "on_time_delivery_rate_percent": round(on_time_rate, 2),
        "late_delivery_rate_percent": round(late_rate, 2),
        "average_delivery_lead_time_days": round(avg_lead_time, 2),
    }


# -------------------------------------------------------------------
# 14. Main Pipeline
# -------------------------------------------------------------------

def main():
    print("==============================================")
    print(" WEEK 2 LOGISTICS PREPROCESSING PIPELINE")
    print("==============================================")

    tables = load_data()

    if not tables:
        print(
            "\nNo raw CSV files were found."
            "\nDownload the Olist dataset and place the CSV files "
            "inside data/raw/ before running the pipeline."
        )
        return

    profile_data(tables)

    tables = remove_exact_duplicates(tables)

    if "orders" in tables:
        orders = clean_order_dates(tables["orders"])
        orders = create_delivery_status(orders)
        tables["orders"] = orders

    if "customers" in tables and "sellers" in tables:
        customers, sellers = standardize_categories(
            tables["customers"],
            tables["sellers"],
        )
        tables["customers"] = customers
        tables["sellers"] = sellers

    if "items" in tables:
        items = create_freight_ratio(tables["items"])
        tables["items"] = items

        print("\nDomain validation:")
        print(validate_numeric_values(items))

    if "orders" in tables:
        delivered = create_delivery_features(tables["orders"])

        print("\n========== KPI SUMMARY ==========")
        for name, value in calculate_kpis(delivered).items():
            print(f"{name}: {value}")

        # Save the cleaned order-level analytical table.
        output = PROCESSED_DIR / "cleaned_delivery_orders.csv"
        delivered.to_csv(output, index=False)

        print(f"\nSaved: {output}")

        # Example outlier detection.
        outliers = find_outliers(delivered, "delay_days")
        print(
            f"Potential delay outliers detected: {len(outliers)}"
        )

    print("\nPreprocessing pipeline completed.")


if __name__ == "__main__":
    main()
