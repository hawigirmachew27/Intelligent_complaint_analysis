"""
preprocessing.py

Reusable data-cleaning and product-mapping utilities for Task 1
(EDA and preprocessing of the CFPB complaint dataset).

This module is imported by:
    - notebooks/task1_eda_preprocessing.ipynb
    - src/run_eda_preprocessing.py (CLI/script entry point)
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Product mapping
# ---------------------------------------------------------------------------
# The CFPB has renamed several `Product` labels over the years. This map
# normalizes all known historical and current variants into the four
# canonical CrediTrust product categories used throughout this project.

PRODUCT_MAP = {
    # Credit Card
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Prepaid card": "Credit Card",

    # Personal Loan
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",

    # Savings Account
    "Checking or savings account": "Savings Account",
    "Bank account or service": "Savings Account",

    # Money Transfer
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
    "Virtual currency": "Money Transfer",
}

TARGET_PRODUCTS = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]


def map_product_category(raw_product: str) -> str:
    """Map a raw CFPB `Product` string to one of the four canonical
    CrediTrust categories. Returns 'Other' for anything out of scope
    (e.g. Mortgage, Student loan, Debt collection, Credit reporting)."""
    if pd.isna(raw_product):
        return "Other"
    return PRODUCT_MAP.get(raw_product.strip(), "Other")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Common boilerplate openers found across many CFPB narratives. Matched
# case-insensitively, anchored near the start of the text.
BOILERPLATE_PATTERNS = [
    r"^\s*i am writing to file a complaint[^.]*\.\s*",
    r"^\s*i am writing to (file|submit|lodge) a complaint[^.]*\.\s*",
    r"^\s*this (complaint|is a complaint) is (in regard|regarding|about)[^.]*\.\s*",
    r"^\s*i would like to file a complaint[^.]*\.\s*",
    r"^\s*to whom it may concern[,:]?\s*",
    r"^\s*dear (sir|madam|cfpb)[,:]?\s*",
    r"^\s*i am submitting this complaint[^.]*\.\s*",
]

# CFPB redacts PII with sequences like XX/XX/XXXX (dates) and XXXX (names,
# account numbers). These tokens carry no semantic content for retrieval.
REDACTION_PATTERN = re.compile(r"\bx{2,}([/\-]x{2,}){0,2}\b", flags=re.IGNORECASE)

# Keep letters, numbers, basic punctuation that carries meaning (. , ! ? ' -)
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-z0-9\s.,!?'\-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    """Clean a single consumer complaint narrative for embedding.

    Steps:
        1. Lowercase
        2. Strip boilerplate complaint-filing openers
        3. Remove CFPB redaction tokens (XX/XX/XXXX, XXXX, etc.)
        4. Remove special characters (keep basic punctuation)
        5. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.lower().strip()

    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = REDACTION_PATTERN.sub(" ", cleaned)
    cleaned = SPECIAL_CHARS_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    """Convenience function: load the raw CFPB CSV, map product categories,
    filter to target products + non-empty narratives, and clean text.

    Returns the cleaned, filtered DataFrame with columns:
        complaint_id, product_category, product, issue, sub_issue,
        company, state, date_received, cleaned_narrative, word_count
    """
    df = pd.read_csv(raw_csv_path, low_memory=False)

    df["product_category"] = df["Product"].apply(map_product_category)
    has_narrative = (
        df["Consumer complaint narrative"].notna()
        & (df["Consumer complaint narrative"].str.strip() != "")
    )

    filtered = df[df["product_category"].isin(TARGET_PRODUCTS) & has_narrative].copy()
    filtered["cleaned_narrative"] = filtered["Consumer complaint narrative"].apply(clean_narrative)
    filtered = filtered[filtered["cleaned_narrative"].str.len() > 0].copy()
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

    return final_df
