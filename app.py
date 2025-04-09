# Updated app1.py with pipeline fitting fix

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import pickle
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data Preprocessing
@st.cache_data
def load_data():
    train = pd.read_csv( r"C:\Users\abims\Downloads\train_data_con.csv")
    test = pd.read_csv(r"C:\Users\abims\Downloads\test_data _con.csv")
    return train, test

train_df, test_df = load_data()

# Create target columns
def create_targets(df):
    if 'order' not in df.columns:
        raise KeyError("The 'order' column is missing from the DataFrame. Please verify the input data.")
    df['converted'] = df['order'].apply(lambda x: 1 if 'checkout' in str(x).lower() else 0)
    df['revenue'] = df.groupby('session_id')['price'].transform('sum')
    return df

train_df = create_targets(train_df)
test_df = create_targets(test_df)

# Aggregate session data
def aggregate_session_data(df):
    session_df = df.groupby('session_id').agg({
        'page1_main_category': 'nunique',
        'page2_clothing_model': 'count',
        'colour': 'nunique',
        'price': 'sum',
        'converted': 'max',
        'revenue': 'max',
        'country': 'first',
        'model_photography': 'first'
    }).reset_index()
    session_df.rename(columns={
        'page1_main_category': 'unique_categories',
        'page2_clothing_model': 'total_clicks',
        'colour': 'unique_colors'
    }, inplace=True)
    return session_df

train_session = aggregate_session_data(train_df)
test_session = aggregate_session_data(test_df)

# Feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['price_per_click'] = X['price'] / (X['total_clicks'] + 1)
        X['color_diversity'] = X['unique_colors'] / (X['total_clicks'] + 1)
        X['session_length'] = X['total_clicks']
        X['bounce_rate'] = (X['total_clicks'] == 1).astype(int)
        X['exit_rate'] = X['total_clicks'] / (X['unique_categories'] + 1)
        return X

# Preprocessing pipeline
numeric_features = ['unique_categories', 'total_clicks', 'unique_colors', 'price', 'price_per_click', 'color_diversity']
categorical_features = ['country', 'model_photography']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('minmax_scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

full_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

# === Fit the full_pipeline on training data to avoid 'Pipeline not fitted' error ===
full_pipeline.fit(train_session.drop(columns=['session_id', 'converted', 'revenue']))

# Streamlit Interface
st.title('Customer Conversion Analysis')

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
manual_input = st.checkbox("Or enter data manually")
input_df = pd.DataFrame()

if uploaded_file is not None:
    sample = uploaded_file.read(1024).decode('utf-8')
    uploaded_file.seek(0)
    sep = ';' if ';' in sample else ','
    input_df = pd.read_csv(uploaded_file, sep=sep, skipinitialspace=True)
    columns_to_rename = {
        'session ID': 'session_id',
        'page 1 (main category)': 'page1_main_category',
        'page 2 (clothing model)': 'page2_clothing_model',
        'model photography': 'model_photography',
        'price 2': 'price_2'
    }
    input_df.rename(columns={k: v for k, v in columns_to_rename.items() if k in input_df.columns}, inplace=True)
    st.write(input_df)

if manual_input:
    st.subheader("Manual Input")
    with st.form("manual_input_form"):
        session_id = st.text_input("Session ID")
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2008)
        month = st.number_input("Month", min_value=1, max_value=12, value=4)
        day = st.number_input("Day", min_value=1, max_value=31, value=1)
        order = st.number_input("Order", min_value=1, step=1)
        country = st.selectbox("Country", ["Australia", "Austria", "Belgium", "British Virgin Islands", "Cayman Islands", "Christmas Island", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "unidentified", "Faroe Islands", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "India", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Arab Emirates", "United Kingdom", "USA", ".biz", ".com", ".int", ".net", ".org"])
        main_category = st.selectbox("Main Category", ["trousers", "skirts", "blouses", "sale"])
        clothing_model = st.text_input("Clothing Model")
        colour = st.selectbox("Colour", ["beige", "black", "blue", "brown", "burgundy", "gray", "green", "navy blue", "of many colors", "olive", "pink", "red", "violet", "white"])
        location = st.selectbox("Location", ["top left", "top in the middle", "top right", "bottom left", "bottom in the middle", "bottom right"])
        model_photography = st.selectbox("Model Photography", ["en face", "profile"])
        price = st.number_input("Price", min_value=0.0, step=0.01)
        price_2 = st.selectbox("Is Price Higher than Average?", ["yes", "no"])
        page = st.number_input("Page", min_value=0,max_value=5, step=1)
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        
        manual_data = pd.DataFrame({
            'session_id': [session_id],
            'year': [year],
            'month': [month],
            'day': [day],
            'order': [order],
            'country': [country],
            'page1_main_category': [main_category],
            'page2_clothing_model': [clothing_model],
            'colour': [colour],
            'location': [location],
            'model_photography': [model_photography],
            'price': [price],
            'price_2': [price_2]
        })
        
        input_df = manual_data

if not input_df.empty:
    try:
        processed_df = aggregate_session_data(create_targets(input_df))
        class_model = pickle.load(open('best_classifier.pkl', 'rb'))
        reg_model = pickle.load(open('best_regressor.pkl', 'rb'))
        cluster_model = pickle.load(open('best_clusterer.pkl', 'rb'))

        conv_pred = class_model.predict(processed_df.drop(['session_id'], axis=1))
        revenue_pred = reg_model.predict(processed_df.drop(['session_id'], axis=1))
        clusters = cluster_model.predict(full_pipeline.transform(processed_df.drop(['session_id'], axis=1)))

        processed_df['Conversion Prediction'] = conv_pred
        processed_df['Revenue Estimate'] = revenue_pred
        processed_df['Customer Segment'] = clusters

        st.subheader('Predictions')
        st.dataframe(processed_df[['session_id', 'Conversion Prediction', 'Revenue Estimate', 'Customer Segment']])

        # Visualizations
        st.subheader('Customer Segments Distribution')
        fig, ax = plt.subplots()
        sns.countplot(x='Customer Segment', data=processed_df)
        st.pyplot(fig)

        st.subheader('Revenue Distribution')
        fig, ax = plt.subplots()
        sns.histplot(processed_df['Revenue Estimate'])
        st.pyplot(fig)

        st.subheader("Conversion Rate by Customer Segment")
        fig, ax = plt.subplots()
        processed_df['Customer Segment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        st.error(f"Prediction Error: {e}")
else:
    st.warning("Please upload a CSV or enter data manually.")