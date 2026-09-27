# RailPulse — Mumbai Train Delay Predictor

An ML extension of a data-based Mumbai local train delay project: supervised regression to predict delay in minutes, and unsupervised clustering to discover natural delay patterns without any labels.

## Dataset

Sourced from Kaggle. 2,880 rows covering 4 routes, 4 time slots, 7 days of week, and 7 months (October–April), with a `delay_minutes` target ranging 0–24 minutes (mean ≈ 8.2, std ≈ 6.8).

Columns: `date`, `route`, `time_slot`, `day_of_week`, `month`, `is_weekend`, `is_monsoon`, `delay_minutes`.

## Exploratory Findings

Before any modeling, groupby analysis on the raw data revealed which features actually carry signal:

- **`time_slot` is the dominant driver.** Morning Peak (~12.7 min) and Evening Peak (~13.0 min) run roughly 4x higher delay than Afternoon and Night (~3.5 min each).
- **`day_of_week` matters moderately.** Weekdays average ~9.1–9.5 min delay; Saturday and Sunday drop to ~5.3–5.4 min.
- **`route` has almost no effect.** All four routes average within 0.1 minute of each other (8.15–8.25), and the route × time_slot breakdown shows every route follows an identical time-of-day pattern.
- **`is_monsoon` was a dead column** — constant at 0 across every row, contributing zero information.

This evidence-first approach shaped every feature decision that followed, rather than including every column by default.

## Feature Engineering

- Dropped `date` (not generalizable), `is_monsoon` (zero variance), `is_weekend` (redundant with the more detailed `day_of_week`).
- One-hot encoded `route`, `time_slot`, `day_of_week`, and `month` (with `drop_first=True` to avoid the dummy variable trap).
- `month` was initially assumed constant (all October) but was found to span 7 months — encoded as a category rather than a raw number, since months wrap around a calendar and a raw numeric encoding would misrepresent distance (e.g., October=10 vs January=1 look far apart numerically but are close in time).

## Train/Test Split

Rather than a random shuffle, the data was split **by month** — training on October–January, testing on February–April. This mimics a real deployment scenario: train on past data, predict on genuinely unseen future months, rather than just unseen rows.

A side effect discovered during feature importance analysis: since `month_2`, `month_3`, and `month_4` were constant (always 0) throughout training, the model never learned to use them at all — they showed exactly 0.000 importance. This is a known trade-off of time-based splits worth flagging honestly rather than hiding.

## Supervised Learning: Regression

Three algorithms were trained and compared on identical train/test data:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3.574 | 4.467 | 0.565 |
| Random Forest | 3.927 | 5.028 | 0.449 |
| Gradient Boosting | 3.563 | 4.470 | 0.564 |

**Finding:** Linear Regression and Gradient Boosting perform essentially identically; Random Forest performs noticeably worse. This is explained by the underlying structure of the data — the exploratory analysis showed the relationship between features and delay is largely **additive** (time_slot and day_of_week each shift delay independently, with no meaningful interaction effects between them). Random Forest's strength is capturing complex interactions, which don't exist here in any strong form, so its added flexibility instead overfits to noise in a moderately-sized dataset (1,824 training rows).

**Feature importance** (from Random Forest) confirmed the exploratory findings numerically: `time_slot_Morning Peak` (0.347) and `time_slot_Evening Peak` (0.267) dominated, together accounting for over 60% of the model's reliance, while all `route_*` features sat near 0.03 — confirming route is close to irrelevant.

**Final model chosen: Linear Regression** — equal performance to Gradient Boosting, with far greater simplicity and interpretability.

The final model is saved via `joblib` as `delay_regression_model.pkl` for reuse without retraining.

## Unsupervised Learning: Clustering

The clustering task asked a different question entirely: with no target label given, can an algorithm rediscover the same behavioral patterns found through supervised analysis, purely from feature similarity?

**Method:** KMeans, after `StandardScaler` normalization (required since `delay_minutes` and one-hot columns sit on very different numeric scales).

**Choosing K:** The elbow method (inertia across K=2 to K=10) statistically suggested K=7–8. However, inspecting actual cluster contents at K=8 revealed it was dominated by `day_of_week` alone — every cluster mapped almost perfectly to a single day, ignoring the much stronger `time_slot` effect. This was traced to a subtle encoding artifact: `day_of_week` expanded into 6 dummy columns versus `time_slot`'s 3, giving it disproportionate combined weight in distance calculations even after per-column scaling.

**K=4 (after removing the unrelated `month` feature, which was separately found to create its own spurious cluster)** produced a cleaner, more interpretable result:
- Morning Peak — its own cluster
- Evening Peak — its own cluster
- Afternoon + Night — merged into one cluster (the algorithm correctly recognized these two time slots are statistically indistinguishable, ~3.5 min average each)
- **Sunday specifically** — its own cluster across all time slots, with a distinctly lower mean delay (5.29 min vs 8.68 min for everyone else)

**Validation:** Silhouette score favored K=8 numerically (0.212 vs 0.179 for K=4), but both scores are modest overall (well below the 0.5 threshold considered "strong"). Given that K=8 was shown to reflect an encoding artifact rather than genuine structure, **K=4 was chosen as the final clustering** on interpretability grounds — a case where the more meaningful result was not the one with the highest raw score.

**Visualization:** PCA was used to compress the 18 encoded features down to 2 dimensions for plotting. The resulting scatter plot showed distinct horizontal bands with genuine separation between clusters, though only ~24.8% of total variance was retained in 2D — meaning the true separation in full feature space is likely cleaner than the 2D plot alone suggests. Silhouette score (computed on full-dimensional data) is treated as the more reliable measure of cluster quality.

## Key Takeaways

- More complex algorithms (Random Forest, higher K) did not automatically perform better — both cases were traced back to genuine data structure (additive relationships, encoding weight imbalance) rather than treated as black boxes.
- Every modeling decision was backed by evidence from exploratory analysis, not assumption.
- Both the supervised and unsupervised halves of the project independently converged on the same real-world patterns: **peak hours** and **weekends (specifically Sunday)** are the true drivers of Mumbai train delays, while **route** has minimal measurable effect in this dataset.

## Tech Stack

Python, pandas, scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor, KMeans, PCA, StandardScaler), matplotlib, joblib.
