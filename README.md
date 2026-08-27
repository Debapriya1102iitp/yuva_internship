# Logistics Data Analytics — Yuva Internship

A Python-based logistics data analytics project developed as part of the Yuva Internship.

## Project Overview

This project studies logistics performance using the **Brazilian E-Commerce Public Dataset by Olist**. The overall objective is to prepare reliable logistics data for analysis of delivery performance, delay prediction, operational segmentation, and route optimization.

The project is being developed progressively across the internship tasks.

### Current Progress

- **Week 1:** Strategic planning and data exploration
- **Week 2:** Data collection, cleaning, and preprocessing
- **Next stages:** Exploratory analysis, predictive modeling, clustering, and route optimization

## Dataset

Reference dataset:

**Brazilian E-Commerce Public Dataset by Olist**

The dataset contains approximately 100,000 orders from 2016–2018 and includes information about:

- Orders and order status
- Purchase and delivery timestamps
- Order items
- Product information
- Sellers
- Customers
- Freight values
- Customer reviews
- Brazilian geographic information

Dataset source:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### Important

The raw Olist CSV files are **not included in this GitHub repository** because they are large and are not required for the code submission.

To run the preprocessing script:

1. Download the dataset from the source above.
2. Extract the CSV files.
3. Create this folder:

```text
data/raw/
```

4. Place the required CSV files inside it.

## Repository Structure

```text
yuva_internship/
│
├── Week_1/
│   └── Week 1 strategic planning report
│
├── Week_2/
│   └── data preprocessing code
│
├── reports/
│   └── internship reports
│
├── data/
│   └── raw/              # Local dataset; not uploaded to GitHub
│
└── README.md
```

## Week 2 — Data Preprocessing

The Week 2 Python implementation demonstrates:

1. Loading multiple logistics-related CSV files
2. Initial dataset profiling
3. Missing-value inspection
4. Duplicate detection and removal
5. Timestamp conversion
6. Delivery-status creation
7. Categorical data standardization
8. Domain validation of numeric values
9. Outlier detection using the IQR method
10. Feature engineering
11. Delivery KPI calculation
12. Standardization using `StandardScaler`
13. Saving a cleaned analytical dataset

### Logistics Features

The preprocessing workflow creates features such as:

- `lead_time_days`
- `delay_days`
- `late_flag`
- `purchase_month`
- `purchase_weekday`
- `freight_ratio`

These features will support the predictive and exploratory stages of the project.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Running the Code

Install the required libraries:

```bash
pip install pandas numpy scikit-learn
```

Then place the Olist CSV files inside:

```text
data/raw/
```

Run:

```bash
python Week_2/data_preprocessing.py
```

The cleaned delivery-order dataset will be generated under:

```text
data/processed/cleaned_delivery_orders.csv
```

## Data Quality Principles

The project follows several important preprocessing principles:

- Do not blindly impute missing logistics events.
- Validate timestamps before calculating delivery durations.
- Distinguish exact duplicates from legitimate one-to-many records.
- Investigate outliers instead of automatically deleting them.
- Standardize categorical values before grouping.
- Validate table relationships before joining.
- Fit preprocessing transformations only on training data during machine-learning stages to prevent data leakage.

## Planned Future Work

### Week 3 — Exploratory Data Analysis

- KPI analysis
- Delivery-delay distributions
- Seller and regional comparisons
- Correlation analysis
- Visualization

### Week 4 — Predictive Analytics

- Delivery-delay prediction
- Classification of late/on-time orders
- Model evaluation

### Later Stages

- Seller and geographic clustering
- Geographic distance analysis
- Vehicle Routing Problem prototype
- Business recommendations

## Project Goal

The final objective is to develop a practical logistics analytics workflow that can help answer:

> **Where are delivery problems occurring, why are they occurring, which orders are at risk, and how can logistics resources be allocated more efficiently?**

## References

- Olist Brazilian E-Commerce Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Pandas Documentation: https://pandas.pydata.org/docs/
- NumPy Documentation: https://numpy.org/doc/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
