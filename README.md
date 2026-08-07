# Flight Delay Prediction

Estimating the risk that a U.S. domestic flight arrives 15+ minutes late, using
only information available at the time the flight is scheduled.

**Result:** ROC-AUC 0.699 on held-out data, catching 64% of flights that
actually arrived late — against a naive baseline that catches none while
scoring 78% accuracy.

**[Live demo →](FILL: your Streamlit Community Cloud URL)**

---

## Why this problem

Delays cost airlines and passengers time and money, and the effects compound: a
late arrival delays the next departure, crews time out, connections are missed.
A model that flags high-risk flights before departure gives operations teams and
travelers a window to plan around it.

The constraint that shaped this project: **no day-of-operation information.**
Weather, security events, and cascading delays from a connecting aircraft are
unknown when a flight is scheduled, so the model can't use them — and neither can
`CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`,
or actual departure time, all of which are only recorded after the flight
operates. Including them would inflate every metric on this page and produce a
model useless in production.

That choice caps how well the model can possibly do. It also makes the output
honest: the model identifies **where delay risk concentrates structurally** —
which routes, carriers, and departure slots carry more of it — rather than
predicting a specific flight on a specific day.

## Data

| | |
|---|---|
| **Source** | [BTS TranStats — Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) |
| **Period** | January – December 2025, assembled from twelve monthly extracts |
| **Rows** | 6,879,483 completed flights after cleaning |
| **Target** | `ARR_DEL15` — 1 if the flight arrived 15+ minutes late |
| **Class balance** | 78% on time / 22% delayed |
| **Modeling sample** | 500,000 rows, stratified |

Raw data is not committed here. The twelve monthly CSVs go in `data/` and are
combined with `glob` + `pd.concat` at the top of the notebook.

### Features

Eight predictors, all known at scheduling time:

- **Schedule** — `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR`, `CRS_ELAPSED_TIME`
- **Route** — `ORIGIN`, `DEST`
- **Carrier** — `OP_UNIQUE_CARRIER`
- **Engineered** — `HOLIDAY_PROX`, capped days to the nearest U.S. federal
  holiday, computed across a three-year window so dates near January 1 measure
  back to the previous New Year's Day rather than forward eleven months

`DISTANCE` was dropped for collinearity with `CRS_ELAPSED_TIME`. `CRS_DEP_TIME`
was dropped once `DEP_HOUR` was derived from it. `FL_DATE` was dropped after
`HOLIDAY_PROX` was built, since `MONTH` and `DAY_OF_WEEK` already carry its
generalizable signal and encoding raw dates would overfit to specific days.

## Approach

**Cleaning.** Cancelled and diverted flights were removed — they have no arrival
outcome and can't contribute to a supervised target. That accounted for 122,134
of the 122,135 missing `ARR_DEL15` values; the single remaining record was
treated as a data-quality anomaly and dropped, along with one flight carrying a
negative scheduled elapsed time.

**Preprocessing.** Two `ColumnTransformer` pipelines rather than one. Linear
models and the random forest use sparse one-hot encoding plus `StandardScaler`;
`HistGradientBoostingClassifier` rejects sparse input, so it gets a dense
preprocessor. Everything is wrapped in an sklearn `Pipeline`, so the fitted
encoders travel with the model and there's no train/test leakage.

**Sampling.** A learning curve at 100K / 300K / 500K / 1M / full moved ROC-AUC by
less than 0.01, so the final models train on a 500,000-row stratified sample —
near-identical performance at a fraction of the compute.

**Imbalance.** All models use `class_weight='balanced'`. Without it, early
models scored high accuracy by predicting "on time" for nearly everything and
identifying almost no real delays.

**Metric choice.** Accuracy is actively misleading here — the naive baseline in
the table below scores 77.69% while catching zero delays. Recall is the primary
metric, since a missed delay costs a traveler more than a false alarm, with F1
and ROC-AUC to keep precision honest. `GridSearchCV` tuned on recall, 3-fold.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **HistGradientBoosting (tuned)** | **0.6474** | **0.3443** | 0.6424 | **0.4483** | **0.6991** |
| Random Forest | 0.6118 | 0.3181 | **0.6472** | 0.4265 | 0.6749 |
| Logistic Regression | 0.6097 | 0.3124 | 0.6240 | 0.4163 | 0.6514 |
| SGD Logistic Regression | 0.6115 | 0.3115 | 0.6124 | 0.4129 | 0.6490 |
| Baseline (overall average) | 0.7769 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

Random Forest edges out HistGradientBoosting on recall by 0.5 percentage points —
close enough to be noise. HistGradientBoosting wins on F1 and ROC-AUC and was
selected as the final model on that basis: better balance between catching real
delays and not crying wolf.

The gap between the baseline's 77.69% accuracy and its 0.500 ROC-AUC is the
clearest single argument in this project for reading past accuracy on an
imbalanced target.

### Reading the scores

The deployed model outputs a **comparative risk score, not a calibrated
probability.** `class_weight='balanced'` reweights training so delays appear as
common as on-time arrivals, which pushes predicted probabilities toward 0.5 and
detaches them from real-world frequencies. A flight scoring 60% ranks riskier
than one scoring 40% — it does not mean six in ten such flights arrive late. The
app presents the number this way deliberately, and omits a hard delayed/not-delayed
verdict, since `predict()`'s fixed 0.5 cutoff would contradict the score shown
beside it.

## The app

A Streamlit application serving the trained pipeline: pick an origin and
destination (filtered to routes present in the training data), a date, a carrier,
and a departure hour, and get a risk score with a plain-language band.

Three things worth noting in the build:

- **Scheduled duration is looked up, not entered.** A traveler booking a flight
  doesn't know its `CRS_ELAPSED_TIME`, but the route implies it. A
  route → median-duration table built from the training data fills it in, and
  doubles as the guard that keeps the model from being asked to score a route it
  never saw.
- **Carrier comparison** re-scores the same route, date, and hour for all twelve
  carriers, isolating how much of the estimate comes from the airline versus the
  route and schedule.
- **Day × hour heatmap** scores all 168 weekday/departure-hour combinations in a
  single batched call, making the schedule structure the model learned visible
  directly.

## Repository structure

```
.
├── notebooks/
│   └── DelayedFlights_Project.ipynb   # EDA, cleaning, feature engineering, modeling
├── Streamlit_app/
│   ├── Home.py
│   ├── best_hgb_model.pkl             # fitted Pipeline — includes the preprocessor
│   ├── route_duration.csv             # ORIGIN/DEST → median CRS_ELAPSED_TIME
│   └── pages/
│       ├── 1_Resume.py
│       ├── 2_Portfolio.py
│       └── 3_Flight_Delay_Simulator.py
├── requirements.txt
└── README.md
```

## Running it

```bash
git clone https://github.com/FILL-your-username/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
streamlit run Streamlit_app/Home.py
```

To retrain: download the twelve 2025 monthly extracts from the BTS link above
into `data/`, then run the notebook top to bottom.

The saved pipeline is tied to the scikit-learn version that produced it — pin it
in `requirements.txt` or loading will warn or fail.

## Limitations

- **No weather.** The single largest driver of delays, captured only indirectly
  through month and route.
- **No delay propagation.** Without tail-number history, the model can't see that
  the inbound aircraft is already running late — often the real cause.
- **Precision is 0.34.** Roughly two of every three flights flagged as elevated
  risk arrive on time. That trade is deliberate given the recall priority, but
  it's the number to look at before trusting any single prediction.
- **One year of data.** Performance in a different traffic environment is untested.
- **Scores aren't calibrated** — see "Reading the scores" above.
- **Structural, not causal.** Carrier differences reflect each airline's overall
  on-time pattern in the training data. They aren't a claim about why.

## Next steps

- Join NOAA weather by origin airport and departure hour — the largest expected gain
- Calibrate probabilities (`CalibratedClassifierCV`) so the output can be read as
  a real likelihood, and tune the threshold against an explicit cost ratio
- Add a prior-leg delay feature using tail number, accepting that it moves the
  model from scheduling-time to day-of-operation

## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn · Altair · Streamlit · joblib

---

Capstone project, M.S. Data Science, Eastern University (2026).
