"""Task 3 - Offer recommendation engine.

Two steps: segment each customer with RFM (Recency, Frequency, Monetary), then
map each segment to an offer via a small, explainable rules table.
"""
import pandas as pd

import config
import data_prep


def assign_segment(r, fm):
    """Map a recency score and a combined frequency-monetary score to a segment."""
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3:
        return "Loyal"
    if r >= 4 and fm <= 2:
        return "New"
    if r <= 2 and fm >= 3:
        return "At Risk"
    if r <= 2 and fm <= 2:
        return "Hibernating"
    return "Needs Attention"


def score_rfm(features, quantiles=config.RFM_QUANTILES):
    """Score R, F, M into 1..q tiers and assign a segment.

    Recency is inverted (a recent customer gets a HIGH score). Values are ranked
    before bucketing so tied counts (e.g. the many customers who ordered once)
    don't collapse the quantile edges.
    """
    f = features.copy()
    asc = list(range(1, quantiles + 1))
    f["r_score"] = pd.qcut(f["recency"].rank(method="first"), quantiles, labels=asc[::-1]).astype(int)
    f["f_score"] = pd.qcut(f["frequency"].rank(method="first"), quantiles, labels=asc).astype(int)
    f["m_score"] = pd.qcut(f["monetary"].rank(method="first"), quantiles, labels=asc).astype(int)
    f["fm_score"] = ((f["f_score"] + f["m_score"]) / 2).round().astype(int)
    f["segment"] = [assign_segment(r, fm) for r, fm in zip(f["r_score"], f["fm_score"])]
    return f


# Offers as data: segment -> (offer, business reason). Easy to read and defend.
OFFER_RULES = {
    "Champions":       ("Early access + loyalty perk", "Reward best customers; don't discount buyers who'd buy anyway"),
    "Loyal":           ("Bundle / cross-sell offer",   "Grow basket size of already-engaged customers"),
    "New":             ("Welcome 10% on next order",   "Convert first-time buyers into repeat customers"),
    "At Risk":         ("Win-back 20% discount",       "Re-engage lapsing high-value customers before they're lost"),
    "Hibernating":     ("Light re-engagement email",   "Low-cost nudge; not worth a deep discount"),
    "Needs Attention": ("Targeted 10% reminder",       "Mid-tier customers cooling off; a modest incentive"),
}


def recommend_offer(segment):
    """Return the offer and reason for a segment."""
    offer, reason = OFFER_RULES.get(segment, ("Standard newsletter", "Default for unknown segment"))
    return {"segment": segment, "offer": offer, "reason": reason}


if __name__ == "__main__":
    df = data_prep.clean_transactions(data_prep.load_raw())
    features = data_prep.get_customer_features(df)
    scored = score_rfm(features)

    print("Segment sizes:")
    print(scored["segment"].value_counts())

    print("\nExample customers:")
    for cid in [12346, 12347, 12350]:
        if cid in scored.index:
            seg = scored.loc[cid, "segment"]
            row = scored.loc[cid]
            print(f"  {cid}: R={row.recency:.0f}d F={row.frequency:.0f} M={row.monetary:.0f} "
                  f"-> {seg} -> {recommend_offer(seg)['offer']}")