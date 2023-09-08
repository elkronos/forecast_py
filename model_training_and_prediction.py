from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tbats import TBATS
from pygam import LinearGAM
from pmdarima import auto_arima

def get_best_arima_order(train_data):
    best_model = auto_arima(train_data, seasonal=True, m=365, trace=False)
    return best_model.order

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict_model(model, X_test):
    return model.predict(X_test)

def prophet_training_and_prediction(train_reset, test, train_size):
    prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)
    prophet.fit(train_reset.rename(columns={'date': 'ds', 'data': 'y'}))
    future = prophet.make_future_dataframe(periods=len(test), freq='D')
    forecast = prophet.predict(future)
    return forecast['yhat'][train_size:]

def arima_training_and_prediction(train, test, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.predict(start=1, end=len(test))
    return forecast

def ets_training_and_prediction(train, test):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

def tbats_training_and_prediction(train, test):
    estimator = TBATS(seasonal_periods=(7, 365.25))
    model = estimator.fit(train)
    forecast = model.forecast(steps=len(test))
    return forecast

def linear_gam_training_and_prediction(X_train, y_train, X_test):
    gam = LinearGAM().fit(X_train, y_train)
    forecast = gam.predict(X_test)
    return forecast

def xgboost_training_and_prediction(X_train, y_train, X_test):
    model = XGBRegressor(objective ='reg:squarederror')
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast

def random_forest_training_and_prediction(X_train, y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast

def svr_training_and_prediction(X_train, y_train, X_test):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast

def elastic_net_training_and_prediction(X_train, y_train, X_test):
    model = ElasticNet(alpha=0.1, l1_ratio=0.7)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast

def ridge_regression_training_and_prediction(X_train, y_train, X_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast