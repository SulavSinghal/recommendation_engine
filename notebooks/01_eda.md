# Phase 0 - Exploratory Data Analysis (do this BEFORE building any engine)

You can't build good recommendations on data you haven't looked at. Work through
this checklist in a notebook or a scratch script. The goal isn't pretty charts --
it's to *know your data* so the cleaning steps in `data_prep.py` are justified.

## Get the data
- Download UCI "Online Retail II".
- Put the file in `data/raw/` and confirm the name matches `config.RAW_FILE`.

## Look at the shape
- [ ] How many rows and columns? What are the dtypes?
- [ ] `df.head()` -- do the columns mean what you expect?
- [ ] `df.describe()` -- anything suspicious in Quantity / Price?

## Find the problems (these justify your cleaning steps)
- [ ] How many rows have a missing Customer ID? (you'll drop these)
- [ ] How many invoices start with "C"? (cancellations -- you'll drop these)
- [ ] How many rows have Quantity <= 0 or Price <= 0? (returns / bad rows)
- [ ] Any blank or junk Descriptions?

## Understand the structure (this shapes your engines)
- [ ] Items per basket -- distribution. (drives Task 1 min_support)
- [ ] Purchases per customer -- are there enough repeat buyers for CF? (Task 2)
- [ ] Spread of total spend per customer. (Task 3 Monetary value)
- [ ] Date range of the data. (Task 3 Recency reference date)

## Write down what you learned
Jot 3-5 sentences of findings. These become the "Data" section of your README
and the reasons behind each cleaning step -- exactly what graders look for.
