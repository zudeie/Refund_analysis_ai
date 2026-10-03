"""
Vireo Audio Data Cleaning Pipeline
==========================================
This is the PREPROCESSING step before the agent.

What it does:
1. Loads all CSVs
2. Removes migration duplicates
3. Converts legacy_fd from paise → rupees
4. Keeps only real refund tickets
5. Adds helper columns
6. Saves a clean dataset ready for the agent
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = Path("data")                  # folder with the original CSVs
OUTPUT_DIR = Path("cleaned_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
def load_raw_data():
    tickets = pd.read_csv(
        DATA_DIR / "tickets.csv",
        parse_dates=["created_at", "first_response_at", "resolved_at"]
    )
    agents = pd.read_csv(
        DATA_DIR / "agents.csv",
        parse_dates=["from_date", "to_date"]
    )
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(DATA_DIR / "products.csv", parse_dates=["launch_date"])

    print("=== RAW DATA LOADED ===")
    print(f"tickets   : {tickets.shape}")
    print(f"agents    : {agents.shape}")
    print(f"orders    : {orders.shape}")
    print(f"customers : {customers.shape}")
    print(f"products  : {products.shape}")
    return tickets, agents, orders, customers, products


# ----------------------------------------------------------------------
# CORE CLEANING
# ----------------------------------------------------------------------
def clean_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning logic:
    - Remove exact duplicate ticket_ids (migration re-import)
    - Convert legacy_fd refunds from paise to rupees
    - Keep only tickets with a real refund (> 0)
    - Create analysis-ready helper columns
    """
    df = tickets.copy()
    print("\n=== STARTING CLEANING ===")

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["ticket_id"], keep="first")
    print(f"✓ Removed {before - len(df)} duplicate tickets")

    # 2. Fix legacy_fd money (paise → rupees)
    legacy_mask = df["source_system"] == "legacy_fd"
    df.loc[legacy_mask, "refund_amount_inr"] = (
        df.loc[legacy_mask, "refund_amount_inr"] / 100
    )
    print("✓ Converted legacy_fd refunds from paise → rupees")

    # 3. Keep only real refunds
    df = df[
        df["refund_amount_inr"].notna() &
        (df["refund_amount_inr"] > 0)
    ].copy()
    print(f"✓ Kept {len(df):,} tickets with actual refunds")

    # 4. Helper columns for the agent
    df["year_month"] = df["created_at"].dt.to_period("M").astype(str)
    df["quarter"] = df["created_at"].dt.to_period("Q").astype(str)
    df["is_gw_other"] = (
        df["refund_reason_code"]
        .fillna("")
        .str.upper()
        .str.contains("GW-OTHER|OTHER|GOODWILL")
    )
    df["has_replacement"] = (
        df["replacement_issued"].fillna("N").str.upper() == "Y"
    )

    # 5. Final sanity check
    print("\n--- Cleaned data summary ---")
    print(df.groupby("source_system")["refund_amount_inr"]
            .agg(["count", "mean", "median", "sum"])
            .round(0))

    total = df["refund_amount_inr"].sum()
    print(f"\nTotal cleaned refund amount: ₹{total:,.0f}")
    return df


# ----------------------------------------------------------------------
# ATTACH USEFUL REFERENCE DATA
# ----------------------------------------------------------------------
def enrich(clean: pd.DataFrame, agents: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Add agent name and product info so the agent has everything in one place."""

    # Latest agent record
    agent_latest = (
        agents
        .sort_values("to_date", ascending=False)
        .drop_duplicates("agent_id")
        [["agent_id", "name", "site", "team", "tier"]]
    )
    clean = clean.merge(agent_latest, on="agent_id", how="left")

    # Product info
    clean = clean.merge(
        products[["sku", "product_name", "family", "retail_price_inr"]],
        left_on="product_sku",
        right_on="sku",
        how="left"
    ).drop(columns=["sku"], errors="ignore")

    return clean


# ----------------------------------------------------------------------
# SAVE CLEAN DATA FOR THE AGENT
# ----------------------------------------------------------------------
def save_clean_data(clean: pd.DataFrame, agents: pd.DataFrame, 
                    orders: pd.DataFrame, customers: pd.DataFrame, 
                    products: pd.DataFrame):

    # Main cleaned refunds file (this is what the agent will mostly use)
    clean.to_csv(OUTPUT_DIR / "clean_refunds.csv", index=False)

    # Also save the reference tables (unchanged but in one place)
    agents.to_csv(OUTPUT_DIR / "agents.csv", index=False)
    orders.to_csv(OUTPUT_DIR / "orders.csv", index=False)
    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    products.to_csv(OUTPUT_DIR / "products.csv", index=False)

    print(f"\n✅  Clean data saved to folder: {OUTPUT_DIR}/")
    print("   - clean_refunds.csv      ← main file for the agent")
    print("   - agents.csv")
    print("   - orders.csv")
    print("   - customers.csv")
    print("   - products.csv")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tickets, agents, orders, customers, products = load_raw_data()

    clean = clean_tickets(tickets)
    clean = enrich(clean, agents, products)

    save_clean_data(clean, agents, orders, customers, products)

    print("\n=== DATA CLEANING COMPLETE ===")
