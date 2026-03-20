# Energy Consumption Forecasting

Forecasting hourly energy consumption using machine learning, with a focus on XGBoost and time-series feature engineering.

## Project Structure

```
energy-forecast/
├── data/                    # Raw and processed data (not committed to git)
│   └── .gitkeep
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory data analysis
│   └── 02_feature_eng.ipynb # Feature engineering
│   └── 03_modelling.ipynb   # Model building and evaluation
├── src/
│   └── features.py          # Reusable feature engineering functions
│   └── evaluate.py          # Evaluation metrics and walk-forward validation
├── app.py                   # Streamlit demo (later)
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# or
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
- Go to: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
- Download and unzip into the `data/` folder
- You should have `energy_dataset.csv` and `weather_features.csv` in `data/`

### 4. Start exploring
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Approach

1. **EDA**: Understand the data, check for missing values, visualise patterns
2. **Feature Engineering**: Create lag features, rolling statistics, time-based features, merge weather data
3. **Baseline Models**: Naive forecast, seasonal naive, simple moving average
4. **XGBoost Model**: Gradient boosting with engineered features
5. **Evaluation**: Walk-forward validation with MAE, RMSE, MAPE
6. **Error Analysis**: Where does the model fail and why?
7. **Streamlit Demo**: Interactive visualisation of predictions vs actuals

## Dataset

Hourly energy demand, generation, prices and weather data from Spain (2015-2018).
Source: [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
