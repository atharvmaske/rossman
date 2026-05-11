import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

st.set_page_config(
    page_title='Rossmann Sales Prediction',
    layout='centered'
)

st.title('Rossmann Store Sales Prediction')

st.write('Predict store sales using Machine Learning')

store = st.number_input(
    'Store ID',
    min_value=1,
    value=1
)

day_of_week = st.selectbox(
    'Day Of Week',
    [1, 2, 3, 4, 5, 6, 7]
)

promo = st.selectbox(
    'Promo Running?',
    [0, 1]
)

state_holiday = st.selectbox(
    'State Holiday',
    ['0', 'a', 'b', 'c']
)

store_type = st.selectbox(
    'Store Type',
    ['a', 'b', 'c', 'd']
)

assortment = st.selectbox(
    'Assortment Type',
    ['a', 'b', 'c']
)

day = st.slider(
    'Day',
    1,
    31,
    15
)

month = st.slider(
    'Month',
    1,
    12,
    6
)

year = st.slider(
    'Year',
    2013,
    2015,
    2015
)

if st.button('Predict Sales'):

    input_df = pd.DataFrame({
        'Store': [store],
        'DayOfWeek': [day_of_week],
        'Promo': [promo],
        'StateHoliday': [state_holiday],
        'StoreType': [store_type],
        'Assortment': [assortment],
        'Day': [day],
        'Month': [month],
        'Year': [year]
    })

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

    input_df[numeric_cols] = imputer.transform(
        input_df[numeric_cols]
    )

    input_df[numeric_cols] = scaler.transform(
        input_df[numeric_cols]
    )

    encoded = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_cols = encoder.get_feature_names_out(
        categorical_cols
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols
    )

    final_df = pd.concat([
        input_df[numeric_cols].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)

    prediction = model.predict(final_df)

    st.success(
        f'Predicted Sales: ₹ {prediction[0]:,.2f}'
    )