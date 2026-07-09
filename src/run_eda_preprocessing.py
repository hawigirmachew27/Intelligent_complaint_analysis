"""
run_eda_preprocessing.py

Task 1 CLI entry point: load the raw CFPB complaint dataset, run EDA,
filter to CrediTrust's four target products, clean narratives, and
save the result to data/filtered_complaints.csv.

Usage:
    python src/run_eda_preprocessing.py \
        --input data/raw/complaints.csv \
        --output data/filtered_complaints.csv \
        --figures-dir data/processed
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script execution
import matplotlib.pyplot as plt

from preprocessing import (
    TARGET_PRODUCTS,
    clean_narrative,
    map_product_category,
)


def run_eda(df: pd.DataFrame, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("EDA: Complaint distribution by product")
    print("=" * 70)
    product_counts = df["Product"].value_counts()
    print(product_counts.to_string())

    fig, ax = plt.subplots(figsize=(11, 6))
    product_counts.plot(kind="barh", ax=ax, color="#2C6E91")
    ax.set_xlabel("Number of complaints")
    ax.set_title("Complaint volume by product (full raw dataset)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "eda_product_distribution.png"), dpi=120)
    plt.close(fig)

    has_narrative = df["Consumer complaint narrative"].notna() & (
        df["Consumer complaint narrative"].str.strip() != ""
    )
    word_counts = df.loc[has_narrative, "Consumer complaint narrative"].str.split().str.len()

    print("\n" + "=" * 70)
    print("EDA: Narrative word count distribution")
    print("=" * 70)
    print(word_counts.describe().to_string())
    print(f"\nNarratives under 5 words: {(word_counts < 5).sum():,}")
    print(f"Narratives over 1000 words: {(word_counts > 1000).sum():,}")

    fig, ax = plt.subplots(figsize=(9, 5))
    clipped = word_counts.clip(upper=word_counts.quantile(0.99))
    ax.hist(clipped.dropna(), bins=60, color="#E07A5F", edgecolor="white")
    ax.set_title("Narrative word count (clipped at p99)")
    ax.set_xlabel("Word count")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "eda_narrative_length.png"), dpi=120)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("EDA: Narrative presence")
    print("=" * 70)
    narrative_counts = has_narrative.value_counts()
    print(narrative_counts.to_string())
    print(f"% with narrative: {narrative_counts.get(True, 0) / len(df):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Task 1: EDA and preprocessing of CFPB complaints")
    parser.add_argument("--input", default="data/raw/complaints.csv", help="Path to raw CFPB CSV")
    parser.add_argument("--output", default="data/filtered_complaints.csv", help="Path for cleaned output CSV")
    parser.add_argument("--figures-dir", default="data/processed", help="Directory to save EDA charts")
    parser.add_argument("--skip-eda-plots", action="store_true", help="Skip generating/saving EDA charts")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    if not args.skip_eda_plots:
        run_eda(df, args.figures_dir)

    print("\n" + "=" * 70)
    print("Filtering to target products + non-empty narratives")
    print("=" * 70)
    df["product_category"] = df["Product"].apply(map_product_category)
    has_narrative = df["Consumer complaint narrative"].notna() & (
        df["Consumer complaint narrative"].str.strip() != ""
    )
    filtered = df[df["product_category"].isin(TARGET_PRODUCTS) & has_narrative].copy()
    print(f"Retained {len(filtered):,} / {len(df):,} rows ({len(filtered) / len(df):.1%})")
    print(filtered["product_category"].value_counts().to_string())

    print("\nCleaning narrative text...")
    filtered["cleaned_narrative"] = filtered["Consumer complaint narrative"].apply(clean_narrative)
    before = len(filtered)
    filtered = filtered[filtered["cleaned_narrative"].str.len() > 0].copy()
    print(f"Dropped {before - len(filtered)} rows empty after cleaning")
    filtered["word_count"] = filtered["cleaned_narrative"].str.split().str.len()

    output_cols = {
        "Complaint ID": "complaint_id",
        "product_category": "product_category",
        "Product": "product",
        "Issue": "issue",
        "Sub-issue": "sub_issue",
        "Company": "company",
        "State": "state",
        "Date received": "date_received",
        "cleaned_narrative": "cleaned_narrative",
        "word_count": "word_count",
    }
    final_df = filtered[list(output_cols.keys())].rename(columns=output_cols)
    final_df = final_df.dropna(subset=["complaint_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(final_df):,} cleaned complaints to {args.output}")


if __name__ == "__main__":
    main()
