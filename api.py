"""api.py - the front door. Loads all four engines once at startup and exposes
them as JSON endpoints. Run from the project root with:

    uvicorn api:app --reload

then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import data_prep
import association_rules
import cross_sell
import offer_engine
import next_action

STATE = {}


@asynccontextmanager
async def lifespan(app):
    df = data_prep.clean_transactions(data_prep.load_raw())
    df_uk = df[df["country"] == "United Kingdom"]
    products = data_prep.filter_to_products(df_uk)

    STATE["rules"] = association_rules.mine_rules(data_prep.get_baskets(products))
    STATE["user_item"] = data_prep.get_user_item_matrix(products)
    STATE["item_sim"] = cross_sell.build_item_similarity(STATE["user_item"])
    STATE["prices"] = products.groupby("stock_code")["price"].mean()
    STATE["scored"] = offer_engine.score_rfm(data_prep.get_customer_features(df_uk))
    STATE["names"] = products.drop_duplicates("stock_code").set_index("stock_code")["description"]
    yield
    STATE.clear()


app = FastAPI(title="Recommendation & Decision Engine", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(STATE.keys())}


@app.get("/rules")
def rules(product: str, k: int = 5):
    matches = association_rules.rules_for_product(STATE["rules"], product, k=k)
    out = [{
        "antecedents": list(row["antecedents"]),
        "consequents": list(row["consequents"]),
        "support": round(float(row["support"]), 4),
        "confidence": round(float(row["confidence"]), 4),
        "lift": round(float(row["lift"]), 2),
    } for _, row in matches.iterrows()]
    return {"product": product, "rules": out}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, k: int = 5):
    if customer_id not in STATE["user_item"].index:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    recs = cross_sell.recommend_for_customer(customer_id, STATE["user_item"], STATE["item_sim"], k=k)
    names = STATE["names"]
    return {"customer_id": customer_id,
            "recommendations": [{"stock_code": c, "description": str(names.get(c, c))} for c in recs]}


@app.get("/offer/{customer_id}")
def offer(customer_id: int):
    if customer_id not in STATE["scored"].index:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    segment = str(STATE["scored"].loc[customer_id, "segment"])
    return {"customer_id": customer_id, **offer_engine.recommend_offer(segment)}


@app.get("/next-best-action/{customer_id}")
def nba(customer_id: int):
    if customer_id not in STATE["scored"].index:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    a = next_action.next_best_action(customer_id, STATE["scored"], STATE["user_item"],
                                     STATE["item_sim"], STATE["prices"])
    return {"customer_id": customer_id, "action": a.type,
            "score": round(float(a.score), 2), "reason": a.reason, "payload": a.payload}