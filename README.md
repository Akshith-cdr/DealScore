# DealScore: Fair Price Prediction for Indian Used Cars

DealScore is a beginner-friendly supervised machine learning project that predicts the fair market price of an Indian used car, compares it with the listed price, generates a DealScore from 1 to 10, and classifies the listing as a Great Deal, Fair Deal, or Bad Deal.

## 1. Project Overview

Buying a used car is difficult because two cars with the same model can have very different prices depending on year, kilometers driven, fuel type, ownership, mileage, engine size, and seller type. DealScore helps a buyer estimate whether a listing is priced fairly.

This project is useful because it turns raw listing data into a simple decision-support tool:

- Regression predicts the fair market price.
- DealScore compares the predicted fair price with the seller's listed price.
- Classification converts that comparison into a readable deal label.

Regression and classification work together in a practical flow. First, the regression model estimates what the car should cost. Then the classifier and scoring logic decide whether the listed price is attractive, fair, or overpriced.

## 2. Folder Structure Explanation

```text
DealScore/
|-- data/
|   |-- raw/
|   |   `-- used_cars_india.csv
|   `-- processed/
|       `-- cars_cleaned.csv
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_fair_price_regression.ipynb
|   `-- 03_deal_quality_classification.ipynb
|-- src/
|   |-- preprocess.py
|   |-- train_regression.py
|   |-- train_classifier.py
|   `-- predict.py
|-- app/
|   `-- app.py
|-- models/
|   |-- fair_price_model.pkl
|   `-- deal_classifier.pkl
|-- requirements.txt
`-- README.md
```

- `data/raw/used_cars_india.csv`: original Kaggle dataset. Add this file manually.
- `data/processed/cars_cleaned.csv`: cleaned dataset created by `src/preprocess.py`.
- `notebooks/01_eda.ipynb`: visual analysis of prices, brands, correlations, kilometers, and fuel type.
- `notebooks/02_fair_price_regression.ipynb`: notebook version of fair price model training.
- `notebooks/03_deal_quality_classification.ipynb`: notebook version of deal category training.
- `src/preprocess.py`: cleans the raw dataset and creates model-ready features.
- `src/train_regression.py`: trains Linear Regression, Random Forest, and XGBoost regressors.
- `src/train_classifier.py`: trains Logistic Regression, Random Forest, and XGBoost classifiers.
- `src/predict.py`: reusable prediction pipeline for one car listing.
- `app/app.py`: simple Streamlit user interface.
- `models/fair_price_model.pkl`: saved best regression model.
- `models/deal_classifier.pkl`: saved best classifier model.
- `requirements.txt`: Python dependencies.

## 3. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install libraries:

```bash
pip install -r requirements.txt
```

Why each library is used:

- `pandas`: load, clean, and save tabular data.
- `numpy`: numeric calculations and score clipping.
- `scikit-learn`: preprocessing, model pipelines, metrics, and baseline models.
- `xgboost`: stronger tree-based regression and classification models.
- `matplotlib` and `seaborn`: EDA graphs.
- `streamlit`: interactive app.
- `joblib`: save and load trained models.
- `jupyter`: run notebooks.

## 4. Data Preprocessing

Place your Kaggle CSV here:

```text
data/raw/used_cars_india.csv
```

Then run:

```bash
python src/preprocess.py
```

Expected output:

```text
Cleaned dataset saved to: data/processed/cars_cleaned.csv
Rows: <number>, Columns: <number>
```

What preprocessing does:

- Standardizes common column names.
- Removes duplicate rows.
- Extracts numbers from values like `18.2 kmpl` and `1197 CC`.
- Handles missing numeric values with medians.
- Handles missing categorical values with modes.
- Creates `car_age`, `km_per_year`, and `engine_per_seat`.
- Saves the cleaned dataset.

The code expects these final columns:

- `brand`
- `model`
- `year`
- `fuel_type`
- `transmission`
- `owner_type`
- `km_driven`
- `seller_type`
- `mileage`
- `engine`
- `seats`
- `price`

## 5. Exploratory Data Analysis

Open:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Included graphs:

- Price distribution: shows whether most cars are low, medium, or high priced.
- Brand analysis: shows the most common brands in the dataset.
- Correlation heatmap: shows relationships between numeric columns.
- KM driven vs price: checks whether higher usage lowers price.
- Fuel type analysis: compares price ranges across fuel types.

## 6. Regression Model

Run:

```bash
python src/train_regression.py
```

Models trained:

- Linear Regression baseline
- Random Forest Regressor
- XGBoost Regressor

Metrics:

- RMSE: typical error size, with large errors punished more.
- MAE: average absolute price error in rupees.
- R2 Score: how much price variation the model explains.

The best model is saved as:

```text
models/fair_price_model.pkl
```

## 7. DealScore Logic

The score compares the listed price with the predicted fair price:

```text
discount_percent = ((fair_price - listed_price) / fair_price) * 100
raw_score = 5 + (discount_percent / 5)
DealScore = clipped raw_score between 1 and 10
```

Interpretation:

- Score near 10: listed price is much lower than fair price.
- Score near 5: listed price is close to fair price.
- Score near 1: listed price is much higher than fair price.

Deal category rule:

- Great Deal: listed price is at least 10% below fair price.
- Fair Deal: listed price is within 10% above or below fair price.
- Bad Deal: listed price is more than 10% above fair price.

## 8. Classification Model

Run regression first, then train the classifier:

```bash
python src/train_regression.py
python src/train_classifier.py
```

Models trained:

- Logistic Regression baseline
- Random Forest Classifier
- XGBoost Classifier

Metrics:

- Accuracy: overall correct predictions.
- Precision: how reliable each predicted category is.
- Recall: how well each real category is found.
- Confusion Matrix: where the classifier is correct or confused.

The best model is saved as:

```text
models/deal_classifier.pkl
```

## 9. Prediction Pipeline

Run:

```bash
python src/predict.py
```

Expected output:

```text
{
  'predicted_fair_price': 480000.0,
  'listed_price': 450000.0,
  'estimated_savings': 30000.0,
  'deal_score': 6,
  'deal_category': 'Fair Deal'
}
```

Actual values depend on your dataset and trained models.

## 10. Streamlit App

Run:

```bash
streamlit run app/app.py
```

The app includes:

- Input form for car details.
- Predicted fair price.
- Listed price comparison.
- Estimated savings.
- DealScore display.
- Deal category output.

## 11. Screenshots

Add screenshots after running the app:

```text
screenshots/app_home.png
screenshots/prediction_result.png
```

## 12. Final Project Flow

1. Download a Kaggle Indian used-car dataset.
2. Rename it to `used_cars_india.csv`.
3. Place it in `data/raw/`.
4. Run preprocessing to create `data/processed/cars_cleaned.csv`.
5. Run EDA notebook to understand the dataset.
6. Train regression models and save the best fair price model.
7. Train classification models and save the best deal classifier.
8. Use `src/predict.py` for reusable predictions.
9. Use `app/app.py` for an interactive Streamlit demo.

## Common Beginner Mistakes

- Forgetting to place the raw CSV at `data/raw/used_cars_india.csv`.
- Using a dataset with different column names and not checking the preprocessing error message.
- Running the classifier before the regression model.
- Forgetting to activate the virtual environment.
- Comparing model performance only with accuracy and ignoring regression error values.
- Expecting identical results across datasets. Kaggle datasets vary, so metrics will differ.

## Future Scope

- Add more data from multiple cities.
- Add car variant-level features.
- Add better outlier handling.
- Add visual screenshots to the README.
- Improve the Streamlit layout after model validation.
