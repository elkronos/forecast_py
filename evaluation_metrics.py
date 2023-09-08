from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

def mean_absolute_scaled_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    scale = mean_absolute_error(y_true[1:], y_true[:-1])
    return mae / scale

def calculate_metrics(y_true, y_pred):
    results = {}
    results['MSE'] = mean_squared_error(y_true, y_pred)
    results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['MAE'] = mean_absolute_error(y_true, y_pred)
    results['R2'] = r2_score(y_true, y_pred)
    results['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    results['sMAPE'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    results['MASE'] = mean_absolute_scaled_error(y_true, y_pred)
    return results
