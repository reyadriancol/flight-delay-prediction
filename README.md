# Flight Delay Prediction

Predicting whether a U.S. domestic flight will arrive 15+ minutes late, using
on-time performance data published by the Bureau of Transportation Statistics.

**Result:** [FILL: e.g. "ROC-AUC 0.72 on a held-out test set, identifying 61% of
delayed flights at a 30% false-positive rate."]

---

## Why this problem

Arrival delays cost U.S. airlines and passengers billions annually, and the
downstream effects compound: a late inbound aircraft delays its next departure,
crews time out, and connections are missed. A model that flags high-risk flights
before departure gives operations teams a window to act — reassigning gates,
pre-positioning crews, or proactively rebooking passengers on tight connections.

This project frames that as a binary classification problem using only
information available **before the flight departs**. Features that leak
post-departure knowledge (actual departure time, taxi-out duration, air time)
are excluded, since a model that uses them would be useless in production.

## Data

| | |
|---|---|
| **Source** | [BTS TranStats — Marketing Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) |
| **Period** | [FILL: e.g. January 2023 – December 2023] |
| **Rows** | [FILL: e.g. ~6.7M flights, sampled to 500K for modeling] |
| **Target** | `ArrDel15` — 1 if the flight arrived 15+ minutes late, 0 otherwise |
| **Class balance** | [FILL: e.g. ~19% positive class] |

Raw data is not committed to this repository. Run the download script below to
reproduce the dataset locally.

### Features used

[FILL — group them so the reader sees the reasoning, e.g.:]

- **Schedule:** month, day of week, scheduled departure hour, scheduled elapsed time
- **Route:** origin, destination, distance
- **Carrier:** operating carrier code
- **Derived:** [e.g. origin airport historical delay rate, red-eye flag]

## Approach

1. **Cleaning** — dropped cancelled and diverted flights, handled missing
   arrival times, [FILL: any other decisions worth defending in an interview]
2. **Preprocessing** — a single scikit-learn `Pipeline` with `ColumnTransformer`:
   one-hot encoding for categoricals, [FILL: scaling / imputation strategy].
   Keeping this in a pipeline prevents train/test leakage and makes the whole
   thing serializable.
3. **Models** — logistic regression as an interpretable baseline, then a random
   forest to capture non-linear interactions between route, carrier, and time of day.
4. **Validation** — [FILL: e.g. stratified train/test split, 5-fold CV on the
   training set. If you split by date rather than randomly, say so — it's a
   stronger choice and worth calling out.]
5. **Metric choice** — accuracy is misleading on an imbalanced target (predicting
   "never delayed" scores well and helps nobody). Primary metric is ROC-AUC, with
   precision/recall reported at the operating threshold.

## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class) | 0.50 | — | — | — |
| Logistic Regression | [FILL] | [FILL] | [FILL] | [FILL] |
| Random Forest | [FILL] | [FILL] | [FILL] | [FILL] |

[FILL: 2–3 sentences interpreting the table. Which model won and by how much?
Was the added complexity of the random forest worth it, or did logistic
regression get within a point or two? Saying "the simpler model was nearly as
good, so I'd ship it" is a strong answer if it's true.]

### What drove the predictions

[FILL: top features and, more importantly, what they mean. e.g. "Scheduled
departure hour was the strongest single signal — delay probability roughly
doubles between 6am and 6pm departures, consistent with delays propagating
through the day as aircraft fall behind schedule."]

## Repository structure

```
.
├── data/                  # not tracked — populated by the download script
├── notebooks/
│   ├── 01_eda.ipynb       # exploration and target definition
│   └── 02_modeling.ipynb  # pipeline construction and evaluation
├── src/
│   ├── download_bts.py    # fetches and unzips BTS monthly files
│   └── features.py        # feature engineering used by the pipeline
├── sql/
│   └── analysis.sql       # [FILL or delete if unused]
├── requirements.txt
└── README.md
```

## Running it

```bash
git clone https://github.com/[FILL: your-username]/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt

python src/download_bts.py        # downloads raw data into data/raw/
jupyter notebook notebooks/02_modeling.ipynb
```

Requires Python 3.10+. Full download is roughly [FILL] GB.

## Limitations

[FILL — this section earns more credibility than the results table. Honest
candidates. Pick the ones that are true:]

- Weather is a major driver of delays and is not included; the model captures it
  only indirectly through month and route.
- No aircraft tail-number history, so delay propagation from the prior leg is
  not modeled directly.
- Trained on a single year — performance on a different traffic environment is untested.

## Next steps

- [FILL: e.g. join NOAA weather by origin airport and departure hour]
- [FILL: e.g. calibrate probabilities and tune the threshold to an explicit
  cost ratio for false alarms vs. missed delays]

## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn · [FILL: PostgreSQL, Tableau]

---

Built as the capstone for an M.S. in Data Science (Eastern University, 2026).
