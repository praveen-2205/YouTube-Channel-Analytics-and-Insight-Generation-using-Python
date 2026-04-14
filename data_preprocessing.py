# ============================================================
# YouTube Channel Analytics - Data Preprocessing in Python
# Dataset: Global YouTube Statistics (1,006 rows x 29 columns)
# Dependencies: pandas, numpy
# ============================================================

import pandas as pd
import numpy as np
from datetime import date
import re

# ----------------------------------------------------------
# 1. Load the Dataset
# ----------------------------------------------------------
df = pd.read_csv(
    "GlobalYouTubeStatistics.csv",
    encoding="latin1",
    na_values=["", "nan", "NaN", "NA"]
)

print("============================================")
print("  DATA PREPROCESSING - YouTube Analytics")
print("============================================\n")
print("[LOAD] Dataset Loaded")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}\n")

# ----------------------------------------------------------
# 2. Inspect the Dataset Structure
# ----------------------------------------------------------
print("-- Structure of the Dataset --")
df.info()
print("\n-- First 5 Rows --")
print(df.head(5))
print("\n-- Summary Statistics --")
print(df.describe(include="all"))
print()

# ----------------------------------------------------------
# 3. Column Names - Normalize & Clean
# ----------------------------------------------------------
original_names = df.columns.tolist()

df.columns = (
    df.columns
    .str.lower()
    .str.replace(r"[^a-z0-9]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

print("-- Cleaned Column Names --")
name_table = pd.DataFrame({"Original": original_names, "Cleaned": df.columns.tolist()})
print(name_table.to_string(index=False))
print()

# ----------------------------------------------------------
# 4. Drop Redundant Columns
# ----------------------------------------------------------
# 'title' is redundant with 'youtuber' and has encoding issues (106 rows)
# 'country' is redundant with 'country_of_origin' and has inconsistencies
drop_cols = [c for c in ["title", "country"] if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"[DROP] Removed redundant columns: {', '.join(drop_cols)}\n")

# ----------------------------------------------------------
# 5. Fix Encoding Corruption in Text Columns
# ----------------------------------------------------------
def fix_encoding(series):
    """Remove non-ASCII encoding artifacts and extra whitespace."""
    return (
        series
        .apply(lambda x: x.encode("latin1").decode("utf-8", errors="ignore") if isinstance(x, str) else x)
        .apply(lambda x: x.encode("utf-8").decode("ascii", errors="ignore") if isinstance(x, str) else x)
        .apply(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)
    )

text_cols = df.select_dtypes(include="object").columns.tolist()
for col in text_cols:
    df[col] = fix_encoding(df[col])
print(f"[ENCODE] Fixed encoding in text columns: {', '.join(text_cols)}\n")

# ----------------------------------------------------------
# 6. Missing Values Analysis
# ----------------------------------------------------------
print("-- Missing Values Summary --")
missing_count = df.isnull().sum()
missing_pct   = (df.isnull().mean() * 100).round(2)
missing_summary = pd.DataFrame({
    "Column":  missing_count.index,
    "Missing": missing_count.values,
    "Pct":     missing_pct.values
}).sort_values("Missing", ascending=False).reset_index(drop=True)
print(missing_summary.to_string(index=False))
print()

# ----------------------------------------------------------
# 7. Handle Missing Values
# ----------------------------------------------------------

# --- 7a. Drop columns with >30% missing ---
high_miss_cols = missing_pct[missing_pct > 30].index.tolist()
for col in high_miss_cols:
    print(f"[DROP] '{col}' has {missing_pct[col]:.1f}% missing - dropping column")
    df.drop(columns=[col], inplace=True)
if high_miss_cols:
    print()

# --- 7b. Mode function for categorical imputation ---
def get_mode(series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode()[0]

# --- 7c. Impute CATEGORICAL columns with MODE ---
cat_cols = [c for c in ["category", "channel_type", "country_of_origin", "abbreviation"] if c in df.columns]
for col in cat_cols:
    na_count = df[col].isnull().sum()
    if na_count > 0:
        mode_val = get_mode(df[col])
        df[col].fillna(mode_val, inplace=True)
        print(f"[IMPUTE] '{col}': {na_count} NAs filled with mode = '{mode_val}'")

# --- 7d. Impute NUMERIC columns with MEDIAN ---
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    na_count = df[col].isnull().sum()
    if na_count > 0:
        med_val = df[col].median()
        df[col].fillna(med_val, inplace=True)
        print(f"[IMPUTE] '{col}': {na_count} NAs filled with median = {med_val:.2f}")

print("\n[OK] Missing values handled\n")

# ----------------------------------------------------------
# 8. Remove Duplicate Rows
# ----------------------------------------------------------
dup_count = df.duplicated().sum()
print(f"[DUP] Duplicate rows found: {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"  Removed {dup_count} duplicates. New row count: {len(df)}")

# Handle duplicate channel names
if "youtuber" in df.columns:
    dup_names = df["youtuber"][df["youtuber"].duplicated()].tolist()
    if dup_names:
        print(f"[DUP] {len(dup_names)} duplicate channel names detected")
        print(f"  Duplicate names: {', '.join(set(dup_names))}")
        df.drop_duplicates(subset="youtuber", keep="first", inplace=True)
        print(f"  Kept first occurrence. New row count: {len(df)}")
print()

# ----------------------------------------------------------
# 9. Data Type Conversions
# ----------------------------------------------------------
print("-- Data Type Conversions --")

# Ensure numeric columns are numeric
numeric_target = [
    "subscribers", "video_views", "uploads",
    "video_views_for_the_last_30_days",
    "lowest_monthly_earnings", "highest_monthly_earnings",
    "lowest_yearly_earnings", "highest_yearly_earnings",
    "video_views_rank", "country_rank", "channel_type_rank",
    "gross_tertiary_education_enrollment___",
    "population", "unemployment_rate", "urban_population",
    "latitude", "longitude"
]
numeric_target = [c for c in numeric_target if c in df.columns]

for col in numeric_target:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  [NUM] Converted '{col}' to numeric")

# Convert categorical columns to category dtype (equivalent to R factor)
factor_target = [c for c in ["category", "channel_type", "country_of_origin", "abbreviation"] if c in df.columns]
for col in factor_target:
    df[col] = df[col].astype("category")
    print(f"  [CAT] Converted '{col}' to category ({df[col].nunique()} levels)")

# Convert created_year and created_date to integer
for col in [c for c in ["created_year", "created_date"] if c in df.columns]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    print(f"  [INT] Converted '{col}' to integer")

print()

# ----------------------------------------------------------
# 10. Feature Engineering
# ----------------------------------------------------------
print("-- Feature Engineering --")

# --- 10a. Combine date columns into a single Date ---
date_cols = ["created_year", "created_month", "created_date"]
if all(c in df.columns for c in date_cols):
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["created_month_num"] = df["created_month"].map(month_map).fillna(1).astype(int)

    df["channel_created_date"] = pd.to_datetime(
        df["created_year"].astype(str) + "-" +
        df["created_month_num"].astype(str) + "-" +
        df["created_date"].astype(str),
        errors="coerce"
    )
    print("  [DATE] Created 'channel_created_date' from year/month/date")

    # Fix 1970 outlier
    mask_1970 = df["created_year"] == 1970
    n1970 = mask_1970.sum()
    if n1970 > 0:
        df.loc[mask_1970, "channel_created_date"] = pd.NaT
        print(f"  [FIX] Set {n1970} channel(s) with year=1970 to NaT (erroneous)")

    df.drop(columns=["created_month_num"], inplace=True)

# --- 10b. Channel Age (years) ---
if "channel_created_date" in df.columns:
    today = pd.Timestamp(date.today())
    df["channel_age_years"] = (
        (today - df["channel_created_date"]).dt.days / 365.25
    ).round(2)
    print("  [NEW] Created 'channel_age_years'")

# --- 10c. Average Monthly Subscriber Gain ---
if all(c in df.columns for c in ["subscribers", "channel_age_years"]):
    df["avg_monthly_sub_gain"] = np.where(
        df["channel_age_years"] > 0,
        (df["subscribers"] / (df["channel_age_years"] * 12)).round(0),
        np.nan
    )
    print("  [NEW] Created 'avg_monthly_sub_gain'")

# --- 10d. Earnings midpoint ---
if all(c in df.columns for c in ["lowest_monthly_earnings", "highest_monthly_earnings"]):
    df["avg_monthly_earnings"] = (
        (df["lowest_monthly_earnings"] + df["highest_monthly_earnings"]) / 2
    ).round(2)
    print("  [NEW] Created 'avg_monthly_earnings'")

if all(c in df.columns for c in ["lowest_yearly_earnings", "highest_yearly_earnings"]):
    df["avg_yearly_earnings"] = (
        (df["lowest_yearly_earnings"] + df["highest_yearly_earnings"]) / 2
    ).round(2)
    print("  [NEW] Created 'avg_yearly_earnings'")

print()

# ----------------------------------------------------------
# 11. Log Transformations (for skewed data)
# ----------------------------------------------------------
print("-- Log Transformations --")
log_cols = [c for c in [
    "subscribers", "video_views", "uploads",
    "video_views_for_the_last_30_days",
    "avg_monthly_earnings", "avg_yearly_earnings"
] if c in df.columns]

for col in log_cols:
    new_col = f"log_{col}"
    df[new_col] = np.log1p(df[col])   # log(1+x) to handle zeros
    print(f"  [LOG] Created '{new_col}'")
print()

# ----------------------------------------------------------
# 12. Outlier Detection (IQR Method)
# ----------------------------------------------------------
print("-- Outlier Detection (IQR Method) --")

def detect_outliers(series):
    s = series.dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR_val = Q3 - Q1
    lower = Q1 - 1.5 * IQR_val
    upper = Q3 + 1.5 * IQR_val
    count = ((s < lower) | (s > upper)).sum()
    return {"count": count, "lower": lower, "upper": upper}

outlier_cols = [c for c in ["subscribers", "video_views", "uploads", "avg_yearly_earnings"] if c in df.columns]
for col in outlier_cols:
    res = detect_outliers(df[col])
    print(f"  [OUT] '{col}': {res['count']} outliers detected")

# Flag outliers in yearly earnings
if "avg_yearly_earnings" in df.columns:
    Q1 = df["avg_yearly_earnings"].quantile(0.25)
    Q3 = df["avg_yearly_earnings"].quantile(0.75)
    IQR_val = Q3 - Q1
    df["earnings_outlier"] = (
        (df["avg_yearly_earnings"] < (Q1 - 1.5 * IQR_val)) |
        (df["avg_yearly_earnings"] > (Q3 + 1.5 * IQR_val))
    )
    print(f"  [FLAG] 'earnings_outlier' column added ({df['earnings_outlier'].sum()} outliers)")
print()

# ----------------------------------------------------------
# 13. Handle Zero Values
# ----------------------------------------------------------
print("-- Zero Value Investigation --")
zero_views   = (df["video_views"] == 0).sum() if "video_views" in df.columns else 0
zero_uploads = (df["uploads"] == 0).sum()     if "uploads"     in df.columns else 0
print(f"  Channels with 0 video views : {zero_views}")
print(f"  Channels with 0 uploads     : {zero_uploads}")

if "avg_monthly_earnings" in df.columns:
    zero_earn = (df["avg_monthly_earnings"] == 0).sum()
    print(f"  Channels with $0 earnings   : {zero_earn}")

# Flag inactive channels
df["is_inactive"] = (df["uploads"] == 0) | (df["video_views"] == 0)
print(f"  [FLAG] Flagged {df['is_inactive'].sum()} channels as inactive\n")

# ----------------------------------------------------------
# 14. Normalization (Min-Max Scaling)
# ----------------------------------------------------------
print("-- Min-Max Normalization --")

def min_max_scale(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0, index=series.index)
    return (series - mn) / (mx - mn)

norm_cols = [c for c in [
    "subscribers", "video_views", "uploads",
    "video_views_rank", "country_rank", "channel_type_rank"
] if c in df.columns]

for col in norm_cols:
    new_col = f"norm_{col}"
    df[new_col] = min_max_scale(df[col]).round(4)
    print(f"  [NORM] Created '{new_col}'")
print()

# ----------------------------------------------------------
# 15. Verify Earnings Consistency
# ----------------------------------------------------------
if all(c in df.columns for c in ["avg_monthly_earnings", "avg_yearly_earnings"]):
    print("-- Earnings Consistency Check --")
    df["earnings_consistent"] = (
        abs(df["avg_yearly_earnings"] - df["avg_monthly_earnings"] * 12) < 1
    )
    pct = round(df["earnings_consistent"].mean() * 100, 1)
    print(f"  Yearly ~ Monthly x 12 : {pct}% consistent\n")

# ----------------------------------------------------------
# 16. Final Dataset Overview
# ----------------------------------------------------------
print("============================================")
print("       FINAL CLEANED DATASET SUMMARY       ")
print("============================================")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"  Total NAs: {df.isnull().sum().sum()}\n")

print("-- Column Names & Types --")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Type":   df.dtypes.astype(str).values,
    "NAs":    df.isnull().sum().values
})
print(col_info.to_string(index=False))

print("\n-- Numeric Summary --")
print(df.select_dtypes(include=[np.number]).describe())

# ----------------------------------------------------------
# 17. Export Cleaned Dataset
# ----------------------------------------------------------
output_file = "GlobalYouTubeStatistics_Cleaned.csv"
df.to_csv(output_file, index=False)
print(f"\n[SAVE] Cleaned dataset saved to: {output_file}")
print(f"  Final dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
print("\n>> Data Preprocessing Complete!")
