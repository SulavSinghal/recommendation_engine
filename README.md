# Recommendation & Decision Engine

A pipeline that turns raw retail transactions into a single recommended action
per customer. It answers three questions in sequence: what product to recommend,
what offer to send, and what action to take next.

## Architecture

The four tasks are one decision pipeline, not four separate scripts:

```
            Transaction data
                  |
        +---------+----------+
        v                    v
  Association rules     Offer engine
  (Task 1)             (Task 3: segment -> offer)
        |                    |
        v                    |
  Cross-sell engine          |
  (Task 2)                   |
        |                    |
        +---------+----------+
                  v
          Next best action
          (Task 4: pick ONE)
                  |
                  v
                api.py
```

Task 1 finds products bought together. Task 2 turns that plus collaborative
filtering into per-customer product recommendations. Task 3 segments customers
and maps each segment to an offer. Task 4 scores the candidate actions from
Tasks 2 and 3 and picks the single best one. `api.py` serves everything as JSON.

## Project structure

```
recommendation-engine/
├── config.py              # paths + tunable parameters
├── data_prep.py           # shared load + clean + reshape (used by all engines)
├── association_rules.py   # Task 1
├── cross_sell.py          # Task 2
├── offer_engine.py        # Task 3
├── next_action.py         # Task 4
├── api.py                 # serves all engines over HTTP
├── requirements.txt
└── data/
    ├── raw/               # the downloaded dataset (gitignored)
    └── processed/         # optional cleaned-data cache (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download UCI "Online Retail II", place the CSV in `data/raw/`, and make sure its
name matches `config.RAW_FILE`.

## Running

Each engine runs standalone for inspection (`python association_rules.py`, etc.).
To serve everything:

```bash
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000/docs for an interactive page to call every
endpoint: `/rules`, `/recommend/{id}`, `/offer/{id}`, `/next-best-action/{id}`.

## Data preparation

All four engines share `data_prep.py`, which loads the CSV and cleans it once:
rows with no customer ID, cancelled orders (invoices starting with "C"),
non-positive quantities or prices, and blank descriptions are removed. On the
full dataset this drops roughly 24% of rows, the large majority being orders with
no customer ID. Cleaning is kept general; task-specific filtering (removing
postage, fees, and adjustment line items so they don't pollute analysis) is
applied only where it's needed, in the basket and similarity builders, because
those non-product lines may still legitimately count as revenue for segmentation.

## Task 1 — Association Rule Mining

Market basket analysis on the United Kingdom subset (~90% of transactions;
~33,000 baskets over ~5,200 products). The pipeline reshapes each invoice into a
one-hot basket, mines frequent itemsets, and ranks the resulting rules by lift.
Rules are ranked by lift rather than confidence because confidence alone is
misleading when the recommended item is simply popular — a high-confidence rule
means little if the consequent appears in most baskets anyway. Lift divides
confidence by the consequent's baseline frequency, isolating genuine association.

Algorithm: the implementation began with Apriori, which ran fine at a minimum
support of 0.02 (15 rules). Lowering support to 0.01 to surface more patterns,
Apriori raised a memory error — it tried to allocate 4.36 GiB for a candidate
array of shape (33,361 × 140,185), because it enumerates every candidate itemset
combination at once, and that set explodes as more items clear the threshold.
Switching to FP-Growth resolved it: FP-Growth mines the same frequent itemsets by
compressing the transactions into a prefix tree rather than enumerating
candidates, so memory stays bounded. A controlled comparison confirmed both
algorithms return identical frequent itemsets, so the change affects performance
only, not results.

Tuning: at support 0.02 only 15 rules survived (the highest-volume pairs); at
0.01, 272 rules emerged, including high-lift but low-volume relationships the
stricter threshold had hidden, such as the Poppy's Playhouse room sets (lift ≈
56). Support was held at 0.01 to balance coverage against noise.

The mined rules predominantly describe product collections — matching variants of
a single designed set (playhouse rooms, spotty plates and cups, the Charlotte and
Lola dolls). A smaller and more useful group are functional complements, such as
red and blue drawer knobs or back-door and shed key fobs, where the pairing
reflects a shared use case rather than a matching design.

## Task 2 — Cross-Sell Recommendation Engine

The engine recommends two ways. The rule-based strategy reuses Task 1: given the
items already in a basket, it returns their highest-lift consequents — a
within-basket signal. The collaborative-filtering strategy works at the customer
level, recommending products bought by customers with similar histories, via
cosine similarity over a binarized user-item matrix. The two are complementary
because association rules only connect items that appear in the same transaction,
whereas collaborative filtering connects items a customer's peers buy even when
those items were never in one basket. Customer 12346, a doormat-heavy buyer,
illustrates this: the engine recommended five further doormats drawn from
purchases spread across the customer's history — affinities within-basket rules
would miss.

Implementation notes: the user-item matrix is binarized before similarity is
computed, treating the data as implicit feedback (bought vs not bought) so a large
quantity of one item doesn't distort scores. Similarity is computed on the
transposed matrix to yield item-to-item rather than customer-to-customer
similarity. Non-product codes are filtered out first; left in, postage co-occurs
with almost every basket and would surface as "similar" to everything.

Evaluation: quality is measured with precision@k and recall@k under a held-out
split. For each customer, 20% of purchased items are hidden, and the
item-similarity matrix is rebuilt on the remaining training purchases only — so
held-out items never influence the recommendations used to score them, which
avoids data leakage. On a random sample of 1,000 UK customers with a fixed seed,
the engine scores precision@5 = 0.143 and recall@5 = 0.102. For context, a random
recommender scores roughly 0.001 precision@5, so 0.143 is about two orders of
magnitude above chance — a healthy result for untuned item-based collaborative
filtering on sparse retail data. Recall is lower than precision because
high-volume customers have large held-out sets that five recommendations cannot
fully recover, a structural property of recall@k rather than a model weakness.

## Task 3 — Offer Recommendation Engine

Two steps: segment each customer with RFM (Recency, Frequency, Monetary), then map
each segment to an offer via a small, explainable rules table.

Scoring handles two real-data issues. Recency is inverted, so a customer who
bought recently (few days) gets a high score, not a low one. And because most
customers in this data ordered exactly once, Frequency is full of tied values;
naively bucketing it with `qcut` fails ("bin edges must be unique"), so values are
ranked before bucketing to break the ties. Each of R, F, M is scored into four
tiers; recency and a combined frequency-monetary score then place the customer on
a 2D grid. The grid's key property is that recency splits high-value customers
into opposite groups: a recent high-value customer is a Champion, while a high-
value customer who has gone quiet is At Risk — the same value, but opposite
treatment. A flat "rank by spend" would miss this entirely.

Offers are stored as data (a segment-to-offer dictionary), not in a heavyweight
rule engine, so each rule is readable and defensible. The logic follows business
sense rather than blanket discounting: Champions get recognition (early access,
loyalty perks), not discounts, because discounting customers who would buy anyway
just burns margin; At Risk high-value customers get an aggressive win-back, since
losing them is expensive; New customers get a welcome offer to drive a repeat
purchase; Hibernating customers get a low-cost nudge.

On the full customer base the segment sizes were Hibernating 2,243, Loyal 1,016,
Champions 821, At Risk 696, Needs Attention 674, and New 428. Hibernating being
the largest group — roughly 38% — is itself a finding: the business acquires many
customers who never return, which makes the smaller Champions and At Risk groups
disproportionately valuable. The logic was validated against three hand-checked
customers: 12346 (£77,556 spend, silent 326 days) → At Risk → win-back; 12347
(recent, frequent) → Champions → loyalty perk; 12350 (one order, long lapsed) →
Hibernating → light email. The segment-boundary thresholds are a tunable design
choice; they were set to produce sensible, actionable group sizes.

## Task 4 — Next Best Action Engine

The orchestrator. For a customer it generates candidate actions — cross-sell the
top product (Task 2), send the segment offer (Task 3), or do nothing — scores each
by expected value (probability of success × value if it succeeds), and returns the
single best. Value comes from observable data: a cross-sell is worth the
recommended product's price; an offer is worth the customer's average order value
(monetary ÷ frequency). The probabilities are assumed conversion and take rates,
documented as assumptions rather than measured values; a real deployment would
calibrate them from past campaign results.

This expected-value framing produces the decisions you'd want. For customer 12346,
the lapsing whale, the win-back offer scores far above any product recommendation,
because the expected value of re-engaging a customer worth thousands per order
dwarfs a single low-priced product sale. The engine is not biased toward offers,
though: when a customer's recommended product is high-priced, the cross-sell can
win on the same expected-value scale. The "do nothing" baseline ensures the engine
can decline to act when nothing is worthwhile.

A decision tree is then fitted over the resulting decisions (RFM scores → chosen
action) to distil the policy into an interpretable, auditable set of if/then
rules. This is a surrogate over the expected-value policy: it makes the decision
logic readable for a stakeholder. With genuine outcome data (which action actually
succeeded), the tree would instead be trained to predict success directly.

## Serving

`api.py` exposes all four engines as JSON endpoints with FastAPI. Models are
loaded once at startup via a lifespan handler — the association rules,
user-item matrix, item-similarity matrix, RFM segments, and prices are all built a
single time, so requests are fast rather than rebuilding state per call. FastAPI's
automatic docs at `/docs` provide an interactive way to call every endpoint.

## Limitations & next steps

The Next Best Action conversion rates are assumptions, not measured values, and
the scoring uses gross expected value without netting out the cost of an offer
(the discount itself); both are straightforward to refine given campaign data. The
dataset has no action-outcome labels, so the decision tree is an interpretable
surrogate over the policy rather than a trained success predictor. Collaborative
filtering cannot recommend to a brand-new customer with no purchase history (a
cold-start gap), though such customers can still receive an offer via RFM.
Analysis focuses on the UK subset for tractability and cleaner signal, and product
descriptions are used as item labels for readability at a small cost in stability
versus stock codes.