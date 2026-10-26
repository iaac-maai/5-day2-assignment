# 5-day2-assignment
Breno Sader, Diego Alpizar, Sindhuja Davuluri, Soumya Datta

Melbourne Housing Price Prediction
Team: Sindhuja Davuluri, Diego Alpízar, Breno Sader, Soumya Datta Course: AI Methods I – Day 2: Supervised learning (IAAC MaAI 2026/27) Model on Hugging Face: [Insert Hugging Face repository link here]
Use case
This model predicts residential real estate prices in Melbourne to assist buyers, investors, and real estate agents in estimating fair market value. Inaccurate property valuations are costly, as overestimating can lead to prolonged listing times and financial stagnation, while underestimating results in significant financial losses for sellers.
Dataset




Name
Melbourne Housing Snapshot
Source
Kaggle Dataset
Rows used
~13,500 properties (after cleaning)
Licence
CC0: Public Domain

Repository structure
├── data/        # downloaded and cleaned data (not committed)
├── figures/     # exported plots (model comparison, error distribution, etc.)
├── models/      # model card (README.md); model.joblib lives on Hugging Face
├── notebook/    # supervised_learning.ipynb & data_cleaning_and_EDA.ipynb
├── requirements.txt
└── README.md
How to run
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Open notebook/supervised_learning.ipynb, select the .venv kernel and run all cells.
(Note: Run notebook/data_cleaning_and_EDA.ipynb first if regenerating the cleaned dataset).
Approach
Target: House price (Price, log-transformed), main metric: R2 score and MAE to evaluate variance explained and real-dollar error.
Features: 19 initial features including numerical variables (Rooms, Distance, Bathroom, Landsize, YearBuilt) and categorical zones (DistanceZone, Postcode).
Preprocessing: Scikit-learn Pipelines with ColumnTransformer: median imputation and scaling for numerical features, and OneHotEncoder for nominal variables.
Models compared: Dummy Regressor (Baseline), Linear Regression, Ridge, KNN, Decision Tree, Random Forest, and HistGradientBoostingRegressor.
Results
Model
CV score (R2 mean ± std)
MAE mean ($)
Fit time (s)
Baseline (mean)
-0.00 ± 0.00
0.19
0.2 s
Linear regression
-71B ± 92B
1,380,707
0.4 s
Ridge
0.80 ± 0.01
0.08
0.3 s
KNN (k=10)
0.80 ± 0.01
0.08
0.9 s
Decision tree
0.77 ± 0.02
0.08
0.4 s
Random forest
0.86 ± 0.01
0.06
15.5 s
Gradient boosting
0.87 ± 0.01
0.06
6.2 s

Final model: Optimized HistGradientBoostingRegressor (via GridSearchCV).
Test set performance: R2=0.89, MAE = $0.06 (log scale), MAPE = $0.96\%$.
 
Limitations
Highly dependent on geographical coordinates (Latitude/Longitude); missing or imprecise location data significantly impacts predictions.
Landsize and a few count columns were capped at their IQR bounds, and 0 m² is common for units.
Dropping rows with a missing YearBuilt removed more than half of the priced sales, which may bias the sample.
The data covers only 2016–2018, so conclusions may not hold for today's market.
AI usage disclosure




Tools
Claude
Why
To structure code pipelines efficiently, understand mathematical error behaviors (like multicollinearity in linear models), troubleshoot errors, and draft documentation.
Where
Notebook logic explanation, pipeline structure, error debugging, and README structure.
How
Asked for conceptual breakdowns and code snippets, verified outputs against local execution in Jupyter, and adapted explanations to project requirements.
Lessons learned
Using pipelines effectively prevents data leakage, and tree-based models handle complex categorical interactions much better than linear models.

License
Code released under the MIT License. The dataset is subject to its own licence (see above).
