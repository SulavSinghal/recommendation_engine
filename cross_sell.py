"""Task 2 - Cross-sell recommendation engine.

Two strategies:
  - recommend_from_rules: reuse Task 1. Given items already in a basket, suggest
    the next product (item-to-item, ranked by association lift).
  - item-based collaborative filtering: given a CUSTOMER, suggest products that
    similar customers bought, via cosine similarity on the user-item matrix.

Quality is measured with precision@k / recall@k on a held-out split (evaluate()).
"""
import random

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
import data_prep
import association_rules


def recommend_from_rules(basket_items, rules, k=config.TOP_K):
    """Strategy A: recommend products for the items already in a basket (uses Task 1)."""
    scores = {}
    for item in basket_items:
        matches = association_rules.rules_for_product(rules, item, k=len(rules))
        for _, row in matches.iterrows():
            for consequent in row["consequents"]:
                if consequent not in basket_items:
                    scores[consequent] = max(scores.get(consequent, 0.0), row["lift"])
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def build_item_similarity(user_item):
    """Item-item cosine similarity (binarized: items sharing customers score high)."""
    binary = (user_item > 0).astype(int)
    sim = cosine_similarity(binary.T)
    return pd.DataFrame(sim, index=user_item.columns, columns=user_item.columns)


def recommend_for_customer(customer_id, user_item, item_sim, k=config.TOP_K):
    """Strategy B: recommend products to a customer via item-based collaborative filtering."""
    if customer_id not in user_item.index:
        return []
    owned = user_item.loc[customer_id]
    owned = owned[owned > 0].index
    if len(owned) == 0:
        return []
    scores = item_sim[owned].sum(axis=1)
    scores = scores.drop(index=owned, errors="ignore")
    return scores.sort_values(ascending=False).head(k).index.tolist()


def precision_at_k(recommended, actual, k=config.TOP_K):
    """Of the top-k recommendations, the fraction the customer actually bought."""
    recommended = recommended[:k]
    if not recommended:
        return 0.0
    return len(set(recommended) & set(actual)) / k


def recall_at_k(recommended, actual, k=config.TOP_K):
    """Of the items the customer actually bought, the fraction we recommended."""
    actual = set(actual)
    if not actual:
        return 0.0
    return len(set(recommended[:k]) & actual) / len(actual)


def train_test_split_matrix(user_item, test_frac=0.2, min_items=2, seed=42):
    """Hide a fraction of each customer's items as a test set.

    Returns (train_matrix, test_items). The held-out items are zeroed in the
    train matrix, so similarity built from it never sees them (no leakage).
    Customers with fewer than min_items are skipped (can't split).
    """
    rng = random.Random(seed)
    train = user_item.copy()
    test_items = {}
    for customer in user_item.index:
        owned = list(user_item.columns[user_item.loc[customer] > 0])
        if len(owned) < min_items:
            continue
        n_test = max(1, int(round(len(owned) * test_frac)))
        held = rng.sample(owned, n_test)
        train.loc[customer, held] = 0
        test_items[customer] = set(held)
    return train, test_items


def evaluate(user_item, k=config.TOP_K, test_frac=0.2, sample=None, seed=42):
    """Average precision@k and recall@k over customers, using a held-out split."""
    train, test_items = train_test_split_matrix(user_item, test_frac, seed=seed)
    item_sim = build_item_similarity(train)

    customers = list(test_items.keys())
    if sample:
        customers = random.Random(seed).sample(customers, min(sample, len(customers)))

    precisions, recalls = [], []
    for customer in customers:
        recs = recommend_for_customer(customer, train, item_sim, k=k)
        actual = test_items[customer]
        precisions.append(precision_at_k(recs, actual, k))
        recalls.append(recall_at_k(recs, actual, k))

    return {
        "customers_evaluated": len(customers),
        "k": k,
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
    }


if __name__ == "__main__":
    df = data_prep.clean_transactions(data_prep.load_raw())
    df = df[df["country"] == "United Kingdom"]
    df = data_prep.filter_to_products(df)

    user_item = data_prep.get_user_item_matrix(df)
    print("User-item matrix:", user_item.shape)

    item_sim = build_item_similarity(user_item)
    names = df.drop_duplicates("stock_code").set_index("stock_code")["description"]

    sample_cust = user_item.index[0]
    owned = user_item.loc[sample_cust]; owned = owned[owned > 0].index
    recs = recommend_for_customer(sample_cust, user_item, item_sim)
    print(f"\nCustomer {sample_cust} bought:", [names.get(c, c) for c in owned][:8])
    print("We recommend:", [names.get(c, c) for c in recs])

    print("\nEvaluating (held-out split, sample of 1000 customers)...")
    print(evaluate(user_item, sample=1000))