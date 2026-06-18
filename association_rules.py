"""Task 1 - Association rule mining (market basket analysis).

Finds products frequently bought together, then ranks the resulting rules by
lift (strongest associations first). Uses FP-Growth instead of Apriori: it finds
the same frequent itemsets but builds a compressed tree rather than testing every
candidate combination, so it stays memory-safe at low support thresholds.
"""
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

import config
import data_prep


def mine_rules(baskets: pd.DataFrame) -> pd.DataFrame:
    """Return association rules, filtered by lift and sorted strongest-first."""
    itemsets = fpgrowth(baskets, min_support=config.MIN_SUPPORT, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence",
                              min_threshold=config.MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= config.MIN_LIFT]
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def rules_for_product(rules: pd.DataFrame, product: str, k: int = config.TOP_K) -> pd.DataFrame:
    """Top-k products most associated with `product` (rule-based cross-sell lookup)."""
    mask = rules["antecedents"].apply(lambda items: product in items)
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return rules[mask].sort_values("lift", ascending=False).head(k)[cols]


if __name__ == "__main__":
    df = data_prep.clean_transactions(data_prep.load_raw())
    df = df[df["country"] == "United Kingdom"]          # keep it tractable; ~90% of data
    baskets = data_prep.get_baskets(df)
    print("Basket matrix:", baskets.shape)

    rules = mine_rules(baskets)
    print("Rules found:", len(rules))
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15).to_string())






