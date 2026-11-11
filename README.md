---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
- sklearn
- tabular-regression
- real-estate
- built-environment
- gradient-boosting
- melbourne
- ai-methods-1
metrics:
- r2
- mae
- mape
---

# 5-day2-assignment

# Melbourne Housing Price Prediction

**Team:** Sindhuja Davuluri, Diego Alpízar, Breno Sader, Soumya Datta  
**Course:** AI Methods I – Day 2: Supervised learning (IAAC MaAI 2026/27)  
**Model on Hugging Face:** https://huggingface.co/iaac-maai/team5-melbourne-price-prediction

## Intended use
This model estimates the **sale price of a residential property in Melbourne, Australia** from property characteristics, location variables, and sale-date features. It is a scikit-learn pipeline containing preprocessing and a tuned `HistGradientBoostingRegressor`, trained on a cleaned subset of the Melbourne Housing Market dataset.

- **Intended users and use:** students, architects, designers, buyers, and analysts who need a quick estimate for exploratory studies or want to inspect how property characteristics affect a prediction.
- **Teaching use:** this is an educational model and should not replace a professional valuation.
- **Out-of-scope uses:** lending, investment, taxation, legal decisions, listing or purchasing decisions, or any other high-stakes financial decision.
- **Geographic scope:** Melbourne-area properties represented in the training data; it should not be assumed to generalize to other cities or markets.

## Model details
| | |
|---|---|
| **Problem type** | Regression |
| **Target** | `log10(Price in AUD)` |
| **Algorithm** | `HistGradientBoostingRegressor` with gradient-boosted decision trees |
| **Preprocessing** | Numeric median imputation and scaling; nominal most-frequent imputation and one-hot encoding with `handle_unknown="ignore"` |
| **Tuning** | `GridSearchCV` over `learning_rate` = 0.05, 0.1, 0.2; `max_leaf_nodes` = 15, 31, 63; `max_iter` = 300 |
| **Saved file** | `melbourne_best_model.joblib`, the fitted scikit-learn pipeline |
| **Input shape** | 19 original columns, expanded to 215 encoded/scaled features |
| **Random state** | 42 |

## Dataset
| | |
|---|---|
| **Name** | Melbourne Housing Snapshot |
| **Source** | [Kaggle Dataset](https://www.kaggle.com/anthonypino/melbourne-housing-market) |
| **Rows used** | ~12,000 properties (after cleaning) |
| **Licence** | CC0: Public Domain |

## Input features

The pipeline expects a pandas `DataFrame` with these 19 columns. The target `Price` is not included in the input.

| Column | Type | Description |
|---|---|---|
| `Rooms`, `Distance`, `Bedroom2`, `Bathroom`, `Car` | numeric | Room, distance, bedroom, bathroom, and car-space features |
| `Landsize`, `YearBuilt`, `Latitude`, `Longitude` | numeric | Land, construction-year, and geographic features |
| `Propertycount`, `SaleYear`, `SaleMonth` | numeric | Area property count and sale-date features |
| `HouseAge`, `LandPerRoom`, `BathPerRoom` | numeric | Derived age, land-per-room, and bathroom-per-room features |
| `HasCarSpace`, `IsHouse` | numeric | Derived binary indicators |
| `Postcode`, `DistanceZone` | category | Melbourne postcode and CBD-distance zone |

Missing numeric values are imputed with the training median. Missing or unseen nominal categories are handled by the imputer and encoder.

## Output

The model returns `log10` of the predicted sale price in Australian dollars. Convert it back with:

```python
price_aud = 10 ** log_price
```

## Training data

The source is the [Melbourne Housing Market Kaggle dataset](https://www.kaggle.com/anthonypino/melbourne-housing-market). The notebook loaded 34,857 rows, retained 12,084 properties after cleaning and feature engineering, and used 9,662 rows for training and 2,416 for testing. The data covers Melbourne-area sales from February 2016 to March 2018.

## Repository structure
```
├── data/        # downloaded and cleaned data (not committed)
├── figures/     # exported plots (model comparison, error distribution, etc.)
├── models/      # model card (README.md); model.joblib lives on Hugging Face
├── notebook/    # supervised_learning.ipynb & data_cleaning_and_EDA.ipynb
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `notebook/supervised_learning.ipynb`, select the `.venv` kernel and run all cells.  

## Approach
- **Target:** House price (`Price`, log-transformed), main metric: **$R^2$ score** and **MAE** to evaluate variance explained and real-dollar error.
- **Features:** 19 initial features including numerical variables (`Rooms`, `Distance`, `Bathroom`, `Landsize`, `YearBuilt`) and categorical zones (`DistanceZone`, `Postcode`). 
- **Preprocessing:** Scikit-learn Pipelines with `ColumnTransformer`: median imputation and scaling for numerical features, and `OneHotEncoder` for nominal variables.
- **Models compared:** Dummy Regressor (Baseline), Linear Regression, Ridge, KNN, Decision Tree, Random Forest, and HistGradientBoostingRegressor.

## Results
| Model | CV score ($R^2$ mean ± std) | MAE mean ($) | Fit time (s) |
|---|---|---|---|
| Baseline (mean) | -0.00 ± 0.00 | 0.19 | 0.2 s |
| Linear regression | -71B ± 92B | 1,380,707 | 0.4 s |
| Ridge | 0.80 ± 0.01 | 0.08 | 0.3 s |
| KNN (k=10) | 0.80 ± 0.01 | 0.08 | 0.9 s |
| Decision tree | 0.77 ± 0.02 | 0.08 | 0.4 s |
| Random forest | 0.86 ± 0.01 | 0.06 | 15.5 s |
| **Gradient boosting** | **0.87 ± 0.01** | **0.06** | **6.2 s** |

**Final model:** Optimized `HistGradientBoostingRegressor` (via `GridSearchCV`).  
**Test set performance:** $R^2 = 0.89$, MAE = \$0.06 (log scale), MAPE = $0.96\%$.

![Model comparison](https://huggingface.co/iaac-maai/team5-melbourne-price-prediction/resolve/main/figures/05_regression_test_results.png)
![Feature importance](https://huggingface.co/iaac-maai/team5-melbourne-price-prediction/resolve/main/figures/06_house_feature_importance.png)

## Training procedure

- The target was transformed with `log10(Price)`.
- Data was split 80/20 with `random_state=42`; the test set was held out for final evaluation.
- Five-fold shuffled cross-validation with `random_state=42` was used for comparison and tuning.
- Models compared included a mean baseline, linear regression, Ridge, KNN, decision tree, random forest, and gradient boosting.
- The tuned gradient-boosting model was selected because it achieved the strongest cross-validation score among the tested models.

On the held-out test set, the final model achieved **R2 = 0.89**, **MAE = 0.06**, **RMSE = 0.08**, and **MAPE = 0.96%**. MAE and RMSE are measured in log10-price units.

## How to use

```bash
pip install scikit-learn pandas numpy joblib huggingface_hub
```

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="iaac-maai/team5-melbourne-price-prediction",
    filename="melbourne_best_model.joblib",
)
model = joblib.load(path)
```

Provide a DataFrame with the 19 columns listed above, then convert the prediction with `price_aud = 10 ** model.predict(example)[0]`.

**Security:** `.joblib` files use Python pickle-based serialization and can execute code when loaded. Only load model files from sources you trust.

## Reproducibility

The full analysis and training workflow is in [day2_application of supervised_learning_model.ipynb](day2_application%20of%20supervised_learning_model.ipynb). It cleans the data, creates derived features, compares models, tunes the final estimator, evaluates the test set, saves figures, and exports the model.

## Limitations
- Highly dependent on geographical coordinates (Latitude/Longitude); missing or imprecise location data significantly impacts predictions.
- Landsize and a few count columns were *capped* at their IQR bounds, and 0 m² is common for units.
- Dropping rows with a missing YearBuilt removed more than half of the priced sales, which may bias the sample.
- The data covers only 2016–2018, so conclusions may not hold for today's market.

## AI usage disclosure
| | |
|---|---|
| **Tools** | Claude |
| **Why** | To structure code pipelines efficiently, understand mathematical error behaviors (like multicollinearity in linear models), troubleshoot errors, and draft documentation. |
| **Where** | Notebook logic explanation, pipeline structure, error debugging, and README structure. |
| **How** | Asked for conceptual breakdowns and code snippets, verified outputs against local execution in Jupyter, and adapted explanations to project requirements. |
| **Lessons learned** | Using pipelines effectively prevents data leakage, and tree-based models handle complex categorical interactions much better than linear models. |

## License
Code released under the [MIT License](LICENSE). The dataset is subject to its own licence (see above).
