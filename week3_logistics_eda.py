"""
Week 3 - Advanced Data Analysis and Visualization in Logistics

This script creates a reproducible hypothetical logistics dataset,
performs exploratory data analysis, calculates logistics KPIs,
and generates six visualizations.

The dataset is synthetic and is intended for the internship task.
No real customer or company data is used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
OUTPUT_DIR = Path("week3_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_dataset(n=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    regions = np.array(["North", "South", "East", "West", "Central"])
    modes = np.array(["Road", "Rail", "Air", "Sea"])
    categories = np.array(
        ["Electronics", "Apparel", "Grocery", "Furniture", "Industrial"]
    )
    months = pd.date_range("2026-01-01", periods=12, freq="MS")

    region = rng.choice(
        regions, n, p=[0.18, 0.24, 0.19, 0.21, 0.18]
    )
    mode = rng.choice(
        modes, n, p=[0.55, 0.18, 0.17, 0.10]
    )
    category = rng.choice(
        categories, n, p=[0.22, 0.24, 0.18, 0.18, 0.18]
    )
    month = rng.choice(months, n)

    distance = np.clip(
        rng.gamma(2.5, 180, n), 30, 1800
    )
    volume = np.clip(
        rng.lognormal(3.0, 0.65, n), 5, 180
    )
    weight = np.clip(
        volume * rng.uniform(0.4, 1.8, n), 2, 300
    )

    mode_delay = pd.Series(mode).map({
        "Road": 0.8,
        "Rail": 1.6,
        "Air": -0.8,
        "Sea": 2.4
    }).to_numpy()

    region_delay = pd.Series(region).map({
        "North": 0.3,
        "South": 1.2,
        "East": 0.6,
        "West": 0.9,
        "Central": 0.1
    }).to_numpy()

    base_days = (
        1.8 + distance / 360 + mode_delay + region_delay
    )

    delivery_time = np.clip(
        base_days + rng.normal(0, 1.25, n),
        0.7, 16
    )

    estimated_time = np.clip(
        base_days + rng.normal(0.7, 0.5, n),
        1.5, 18
    )

    delay = delivery_time - estimated_time

    fuel_factor = pd.Series(mode).map({
        "Road": 1.0,
        "Rail": 0.62,
        "Air": 1.8,
        "Sea": 0.45
    }).to_numpy()

    transport_cost = np.clip(
        70
        + distance * 0.18 * fuel_factor
        + volume * 4.5
        + weight * 0.75
        + rng.normal(0, 35, n),
        40,
        None
    )

    return pd.DataFrame({
        "shipment_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "date": month,
        "region": region,
        "transport_mode": mode,
        "product_category": category,
        "distance_km": np.round(distance, 1),
        "shipment_volume_units": np.round(volume, 1),
        "weight_kg": np.round(weight, 1),
        "delivery_time_days": np.round(delivery_time, 2),
        "estimated_delivery_days": np.round(estimated_time, 2),
        "delay_days": np.round(delay, 2),
        "transport_cost": np.round(transport_cost, 2),
        "on_time": (delay <= 0).astype(int)
    })


def print_eda(df):
    numeric = [
        "distance_km",
        "shipment_volume_units",
        "weight_kg",
        "delivery_time_days",
        "delay_days",
        "transport_cost"
    ]

    print("\n=== DATASET OVERVIEW ===")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df[numeric].describe().round(2))

    print("\n=== KPI SUMMARY ===")
    print(f"On-time delivery rate: {df['on_time'].mean() * 100:.2f}%")
    print(f"Late delivery rate: {(1 - df['on_time'].mean()) * 100:.2f}%")
    print(f"Average delivery time: {df['delivery_time_days'].mean():.2f} days")
    print(f"Average delay: {df['delay_days'].mean():.2f} days")
    print(f"Average transport cost: {df['transport_cost'].mean():.2f}")

    print("\n=== CORRELATIONS ===")
    print(df[numeric].corr().round(2))


def create_visualizations(df):
    # 1. Distribution
    plt.figure()
    plt.hist(df["delivery_time_days"], bins=30)
    plt.xlabel("Delivery time (days)")
    plt.ylabel("Number of shipments")
    plt.title("Distribution of Delivery Times")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_delivery_time_distribution.png", dpi=180)
    plt.close()

    # 2. Monthly trend
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["on_time"]
        .mean()
        .mul(100)
    )

    plt.figure()
    plt.plot(monthly.index.astype(str), monthly, marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("On-time delivery rate (%)")
    plt.title("Monthly On-Time Delivery Performance")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_monthly_on_time_rate.png", dpi=180)
    plt.close()

    # 3. Cost vs distance
    plt.figure()
    plt.scatter(
        df["distance_km"],
        df["transport_cost"],
        alpha=0.55
    )
    plt.xlabel("Distance (km)")
    plt.ylabel("Transport cost")
    plt.title("Transportation Cost vs. Distance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_cost_vs_distance.png", dpi=180)
    plt.close()

    # 4. Delay by mode
    mode_delay = (
        df.groupby("transport_mode")["delay_days"]
        .mean()
        .sort_values()
    )

    plt.figure()
    plt.bar(mode_delay.index, mode_delay.values)
    plt.xlabel("Transport mode")
    plt.ylabel("Average delay (days)")
    plt.title("Average Delivery Delay by Transport Mode")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_delay_by_transport_mode.png", dpi=180)
    plt.close()

    # 5. Regional performance
    region_otd = (
        df.groupby("region")["on_time"]
        .mean()
        .mul(100)
        .sort_values()
    )

    plt.figure()
    plt.barh(region_otd.index, region_otd.values)
    plt.xlabel("On-time delivery rate (%)")
    plt.ylabel("Region")
    plt.title("Regional On-Time Delivery Performance")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_regional_on_time_rate.png", dpi=180)
    plt.close()

    # 6. Correlation matrix
    numeric = [
        "distance_km",
        "shipment_volume_units",
        "weight_kg",
        "delivery_time_days",
        "delay_days",
        "transport_cost"
    ]

    corr = df[numeric].corr()

    plt.figure(figsize=(8.5, 6.5))
    image = plt.imshow(
        corr,
        interpolation="nearest",
        aspect="auto"
    )
    plt.colorbar(image, label="Correlation")
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(corr.index)),
        corr.index
    )
    plt.title("Correlation Matrix of Key Logistics Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_correlation_matrix.png", dpi=180)
    plt.close()


def main():
    df = create_dataset()

    csv_output = Path("week3_hypothetical_logistics_dataset.csv")
    df.to_csv(csv_output, index=False)

    print_eda(df)
    create_visualizations(df)

    print("\nDataset saved to:", csv_output)
    print("Visualizations saved to:", OUTPUT_DIR)
    print("\nWeek 3 analysis completed successfully.")


if __name__ == "__main__":
    main()
