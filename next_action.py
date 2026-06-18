"""Task 4 - Next Best Action engine (the orchestrator).

For a customer, generate candidate actions (cross-sell a product, send the
segment offer, or do nothing), score each by expected value
(probability of success x value if it succeeds), and pick the best. A decision
tree is then fitted over the resulting decisions to give an interpretable,
auditable policy.
"""
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier, export_text

import config
import data_prep
import cross_sell
import offer_engine


# Assumed conversion rates. A real deployment calibrates these from historical
# campaign data; here they are documented assumptions, not measured values.
CROSS_SELL_CONVERSION = 0.10
OFFER_TAKE_RATE = {
    "Champions": 0.05, "Loyal": 0.10, "New": 0.20,
    "At Risk": 0.15, "Hibernating": 0.02, "Needs Attention": 0.08,
}


@dataclass
class Action:
    type: str
    payload: dict
    score: float
    reason: str


def candidate_actions(customer_id, scored, user_item, item_sim, prices):
    """Build candidate actions for one customer: cross-sell, offer, no-action."""
    actions = []

    recs = cross_sell.recommend_for_customer(customer_id, user_item, item_sim, k=1)
    if recs:
        product = recs[0]
        price = float(prices.get(product, 0.0))
        actions.append(Action(
            "cross_sell", {"product": product, "price": price},
            CROSS_SELL_CONVERSION * price,
            f"Recommend {product} (p={CROSS_SELL_CONVERSION}, value=£{price:.2f})"))

    if customer_id in scored.index:
        row = scored.loc[customer_id]
        segment = row["segment"]
        avg_order_value = row["monetary"] / max(row["frequency"], 1)
        take = OFFER_TAKE_RATE.get(segment, 0.05)
        offer = offer_engine.recommend_offer(segment)
        actions.append(Action(
            "offer", {"segment": segment, "offer": offer["offer"]},
            take * avg_order_value,
            f"{offer['offer']} (p={take}, value=£{avg_order_value:.2f})"))

    actions.append(Action("no_action", {}, 0.0, "No worthwhile action"))
    return actions


def next_best_action(customer_id, scored, user_item, item_sim, prices):
    """Return the single highest-scoring action for a customer."""
    actions = candidate_actions(customer_id, scored, user_item, item_sim, prices)
    return max(actions, key=lambda a: a.score)


def fit_policy_tree(scored, user_item, item_sim, prices, max_depth=4, sample=None):
    """Fit an interpretable decision tree over the chosen actions (RFM -> action).

    The tree distils the expected-value policy into auditable if/then rules. With
    real outcome data it would instead be trained to predict which action succeeds.
    """
    customers = list(scored.index)[:sample] if sample else list(scored.index)
    X, y = [], []
    for cid in customers:
        action = next_best_action(cid, scored, user_item, item_sim, prices)
        X.append(scored.loc[cid, ["r_score", "f_score", "m_score"]].tolist())
        y.append(action.type)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, y)
    return tree


if __name__ == "__main__":
    df = data_prep.clean_transactions(data_prep.load_raw())
    df_uk = df[df["country"] == "United Kingdom"]
    products = data_prep.filter_to_products(df_uk)

    user_item = data_prep.get_user_item_matrix(products)
    item_sim = cross_sell.build_item_similarity(user_item)
    prices = products.groupby("stock_code")["price"].mean()
    scored = offer_engine.score_rfm(data_prep.get_customer_features(df_uk))

    print("Next best action for benchmark customers:")
    for cid in [12346, 12347, 12350]:
        if cid in scored.index:
            a = next_best_action(cid, scored, user_item, item_sim, prices)
            print(f"  {cid}: {a.type:10s} | score {a.score:8.2f} | {a.reason}")

    print("\nInterpretable policy (decision tree over RFM scores):")
    tree = fit_policy_tree(scored, user_item, item_sim, prices, sample=1500)
    print(export_text(tree, feature_names=["r_score", "f_score", "m_score"]))