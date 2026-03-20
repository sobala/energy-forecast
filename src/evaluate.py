"""
Evaluation metrics and walk-forward validation for time-series forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    Note: breaks when y_true contains zeros.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, model_name='Model'):
    """Print all metrics for a model."""
    results = {
        'model': model_name,
        'MAE': round(mae(y_true, y_pred), 2),
        'RMSE': round(rmse(y_true, y_pred), 2),
        'MAPE': round(mape(y_true, y_pred), 2),
    }
    print(f"\n{model_name}:")
    print(f"  MAE:  {results['MAE']}")
    print(f"  RMSE: {results['RMSE']}")
    print(f"  MAPE: {results['MAPE']}%")
    return results


def walk_forward_validation(df, feature_cols, target_col, model, 
                            train_months=12, test_months=1, step_months=1):
    """
    Walk-forward validation for time-series.
    
    Instead of one train/test split, we:
    1. Train on months 1-12, test on month 13
    2. Train on months 1-13, test on month 14
    3. And so on...
    
    This gives a more realistic estimate of model performance.
    
    Args:
        df: DataFrame with a datetime index
        feature_cols: list of feature column names
        target_col: name of target column
        model: sklearn-compatible model (must have fit/predict)
        train_months: initial training window in months
        test_months: test window in months
        step_months: how far to step forward each iteration
    
    Returns:
        DataFrame with actual vs predicted values across all test windows
    """
    results = []
    
    dates = df.index.to_series()
    min_date = dates.min()
    max_date = dates.max()
    
    train_end = min_date + pd.DateOffset(months=train_months)
    
    fold = 0
    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)
        
        train = df[df.index < train_end]
        test = df[(df.index >= train_end) & (df.index < test_end)]
        
        if len(test) == 0:
            break
        
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        fold_results = pd.DataFrame({
            'actual': y_test.values,
            'predicted': preds,
            'fold': fold,
        }, index=test.index)
        
        results.append(fold_results)
        
        train_end += pd.DateOffset(months=step_months)
        fold += 1
    
    return pd.concat(results)
