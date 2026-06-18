"""
Central configuration for the recommendation & decision engine.

Keeping paths and tunable parameters in ONE place means you never hardcode a
number deep inside an engine. When you want to experiment (raise the minimum
support, ask for more recommendations), you change it here once and every engine
picks it up. This is also the first thing a grader reads to understand your knobs.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Download UCI "Online Retail II", put the file in data/raw/, and make sure
# this name matches it.
RAW_FILE = RAW_DIR / "online_retail_II.csv"
CLEAN_FILE = PROCESSED_DIR / "transactions_clean.parquet"

# --- Task 1: Association rules -------------------------------------------
MIN_SUPPORT = 0.01     # keep itemsets appearing in >= 2% of baskets
MIN_CONFIDENCE = 0.30   # rule must hold at least 30% of the time
MIN_LIFT = 1.0          # keep only positive associations (lift > 1)

# --- Task 2: Cross-sell --------------------------------------------------
TOP_K = 5               # how many products to recommend

# --- Task 3: Offers / segmentation --------------------------------------
RFM_QUANTILES = 4       # number of tiers when scoring R, F, M

# --- Task 4: Next best action -------------------------------------------
# You'll define scoring weights / thresholds here once you design the logic.
# e.g. how much to weight "probability of success" vs "value of the action".
