import xgboost as xgb
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib.pyplot as plt
def create_time_series_features(df):
    time_series_df = df.copy()
    time_series_df['month'] = time_series_df.index.month
    time_series_df['dayofmonth'] = time_series_df.index.day
    time_series_df['year'] = time_series_df.index.year
    return time_series_df

# Load and preprocess the data
dataset = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]
df = [pd.read_csv(f, parse_dates=True, index_col=0) for f in dataset]
data = pd.concat(df, axis=1)
data.columns = ["interest", "vacancy", "consumer price index"]
data = create_time_series_features(data)

data = data.ffill().dropna()

HTX_data = ["Metro_median_sale_price_uc_sfrcondo_sm_sa_week.csv",
            "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"]
df = [pd.read_csv(f) for f in HTX_data]
df = [pd.DataFrame(dfs.iloc[6, 5:]) for dfs in df]

for dfs in df:
    dfs.index = pd.to_datetime(dfs.index)
    dfs["month"] = dfs.index.to_period("M")

hp_data = df[0].merge(df[1], on="month")
hp_data.index = df[0].index

del hp_data["month"]
hp_data.columns = ["price", "value"]

data.index = data.index + timedelta(days=2)
hp_data = data.merge(hp_data, left_index=True, right_index=True)

hp_data["adjusted_price"] = hp_data["price"] / hp_data["consumer price index"] * 100
hp_data["adjusted_value"] = hp_data["value"] / hp_data["consumer price index"] * 100
hp_data['adjusted_price'] = pd.to_numeric(hp_data['adjusted_price'], errors='coerce')
hp_data['adjusted_value'] = pd.to_numeric(hp_data['adjusted_value'], errors='coerce')
hp_data["next_quarter"] = hp_data["adjusted_price"].shift(-13)
hp_data = create_time_series_features(hp_data)
hp_data.dropna(inplace=True)

# Define predictors and target
predictors = ["interest", "vacancy", "adjusted_price", "adjusted_value", "dayofmonth", "month", "year"]
target = "next_quarter"

# Split data into predictors and target
X, y = hp_data[predictors], hp_data[target]


# Function to add lag features
def add_lags(df, lag_values):
    for lag in lag_values:
        df.loc[:, f'lag_{lag}'] = df['next_quarter'].shift(lag)
    return df.dropna()


# Define the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)  # Set the number of splits according to your dataset size
lag_values = [7, 28, 84]
mse_scores = []  # To store MSE for each fold


def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "objective": 'reg:squarederror',

        "eval_metric": ['rmse', 'mae', ],  # Include RMSE and MAE
        "tree_method": 'hist',
        "device": 'cuda',
    }

    avg_rmse = 0.0

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train_optuna, X_val = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_optuna, y_val = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        print(f"Fold {i}:")
        print(f"  X train: {X_train_optuna}")
        print(f"  y train:  {y_train_optuna}")
        # Merge lag features with training and testing sets
        X_train_optuna['next_quarter'] = y_train_optuna  # Add target_variable to create lag features
        X_val['next_quarter'] = y_val  # Add target_variable to create lag features

        X_train_lagged = add_lags(X_train_optuna, lag_values)
        X_val_lagged = add_lags(X_val, lag_values)

        y_train_lagged = X_train_lagged.pop('next_quarter')
        y_test_lagged = X_val_lagged.pop('next_quarter')

        model = xgb.XGBRegressor(**params)  # Define your XGBoost model
        model.fit(X_train_lagged, y_train_lagged, eval_set=[(X_val_lagged, y_test_lagged)], verbose=100)
        predictions = model.predict(X_val_lagged)
        model.set_params()
        fold_rmse = mean_squared_error(y_test_lagged, predictions, squared=False)
        avg_rmse += fold_rmse

    avg_rmse /= tscv.n_splits
    print('Average RMSE:', avg_rmse)

    return avg_rmse


# optimize hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")
param_importances = optuna.visualization.plot_param_importances(study)
param_importances.show()
print('Best parameters', study.best_params)
print('Best value', study.best_value)
print('Best trial', study.best_trial)
trial = study.best_trial

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=best_params)

# Define predictors and target
predictors = ["interest", "vacancy", "adjusted_price", "adjusted_value", "dayofmonth", "month", "year"]
target = "next_quarter"

# Split data into train and test after filling missing values
train_size = int(0.8 * len(hp_data))
train, test = hp_data.iloc[:train_size], hp_data.iloc[train_size:]
X_train, X_test, y_train, y_test = train[predictors], test[predictors], train[target], test[target]

# Initialize and retrain the XGBoost model with params from optuna
xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
xgb_predictions = xgb_model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, xgb_predictions)
print('Mean Absolute Error:', mae)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, xgb_predictions) * 100
print('Mean Absolute Percentage Error:', mape)

# Calculate MSE
mse = mean_squared_error(y_test, xgb_predictions)
print('Mean Squared Error:', mse)

# Calculate NRMSE
nrmse = mean_squared_error(y_test, xgb_predictions, squared=False) / (max(y_test) - min(y_test))
print('Normalized Root Mean Squared Error:', nrmse)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, xgb_predictions))
print('Root Mean Squared Log Error:', rmsle)

# Calculate R^2
r_2 = r2_score(y_test, xgb_predictions)
print('R^2 score:', r_2)

import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(X_train, y_train, color='g')
plt.plot(X_test, xgb_predictions, color = 'r')
plt.show()

metrics = {
    'MAE': mae,
    'MAPE': mape,
    'MSE': mse,
    'NRMSE': nrmse,
    'RMSLE': rmsle,
    'R^2': r_2
}

plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Evaluation Metrics')
plt.xticks(rotation=45)
plt.show()

filename = "xgboost_regression_htx"
pickle.dump(xgb_model, open(filename, "wb"))
