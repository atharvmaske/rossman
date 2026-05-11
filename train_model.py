import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv(
    'dataset/train.csv',
    low_memory=False
)

store_df = pd.read_csv(
    'dataset/store.csv',
    low_memory=False
)

merge_df = train_df.merge(
    store_df,
    how='left',
    on='Store'
)

merge_df = merge_df[
    merge_df['Open'] == 1
].copy()

merge_df['Date'] = pd.to_datetime(
    merge_df['Date']
)

merge_df['Day'] = merge_df['Date'].dt.day
merge_df['Month'] = merge_df['Date'].dt.month
merge_df['Year'] = merge_df['Date'].dt.year

input_cols = [
    'Store',
    'DayOfWeek',
    'Promo',
    'StateHoliday',
    'StoreType',
    'Assortment',
    'Day',
    'Month',
    'Year'
]

target_col = 'Sales'

inputs = merge_df[input_cols].copy()
targets = merge_df[target_col].copy()

numeric_cols = [
    'Store',
    'Day',
    'Month',
    'Year'
]

categorical_cols = [
    'DayOfWeek',
    'Promo',
    'StateHoliday',
    'StoreType',
    'Assortment'
]

imputer = SimpleImputer(strategy='mean')

inputs[numeric_cols] = imputer.fit_transform(
    inputs[numeric_cols]
)

scaler = MinMaxScaler()

inputs[numeric_cols] = scaler.fit_transform(
    inputs[numeric_cols]
)

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded_data = encoder.fit_transform(
    inputs[categorical_cols]
)

encoded_cols = encoder.get_feature_names_out(
    categorical_cols
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_cols,
    index=inputs.index
)

X = pd.concat([
    inputs[numeric_cols],
    encoded_df
], axis=1)

y = targets

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

preds = model.predict(X)

rmse = np.sqrt(
    mean_squared_error(y, preds)
)

print(f'RMSE: {rmse}')

pickle.dump(
    model,
    open('model.pkl', 'wb')
)

pickle.dump(
    scaler,
    open('scaler.pkl', 'wb')
)

pickle.dump(
    encoder,
    open('encoder.pkl', 'wb')
)

pickle.dump(
    imputer,
    open('imputer.pkl', 'wb')
)

print('Model saved successfully!')