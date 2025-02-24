# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
                             davies_bouldin_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import davies_bouldin_score
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Data Preprocessing
@st.cache_data
def load_data():
    train = pd.read_csv( r"C:\Users\abims\Downloads\train_data (1).csv")
    test = pd.read_csv(r"C:\Users\abims\Downloads\test_data (1).csv")
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



#  Exploratory Data Analysis (EDA) 
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")

    # Time-based Analysis
    if 'SESSION START' in df.columns:
        df['hour'] = pd.to_datetime(df['SESSION START']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['SESSION START']).dt.day_name()

    # Conversion distribution
    plt.figure(figsize=(8,5))
    sns.countplot(x='converted', data=df)
    plt.title('Conversion Distribution')
    st.pyplot(plt)

    # Numerical feature distributions
    df[['unique_categories', 'total_clicks', 'unique_colors', 'price', 'revenue']].hist(figsize=(12,8))
    plt.tight_layout()
    st.pyplot(plt)

    # Correlation heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
   


perform_eda(train_session)

#  Feature Engineering 
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


#Preprocessing Pipeline
numeric_features = ['unique_categories', 'total_clicks', 'unique_colors', 'price', 'price_per_click', 'color_diversity']
categorical_features = ['country','model_photography']


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('minmax_scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

full_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

#Balancing Techniques
X = train_session.drop(['converted', 'revenue', 'session_id'], axis=1)
y_class = train_session['converted']
RandomUnderSampler()
smote = SMOTE(random_state=42)
class_weight='balanced'
if len(y_class.unique()) > 1:
    X_resampled, y_resampled = smote.fit_resample(X, y_class)
else:
    print("SMOTE skipped: Only one class present in y_class")
    X_resampled, y_resampled = X, y_class  


print("X",X.columns.tolist())

# Model Building & Evaluation 
mlflow.set_experiment("Customer_Conversion_Analysis")

# Classification
with mlflow.start_run(run_name="Classification"):
    class_pipeline = Pipeline([
        ('preprocessing', full_pipeline),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
   

    class_pipeline.fit(X_resampled, y_resampled)
    class_preds = class_pipeline.predict(X)

    mlflow.log_metrics({
        'accuracy': accuracy_score(y_class, class_preds),
        'precision': precision_score(y_class, class_preds),
        'recall': recall_score(y_class, class_preds),
        'f1_score': f1_score(y_class, class_preds),
        'roc_auc': roc_auc_score(y_class, class_preds)
    })

    mlflow.sklearn.log_model(class_pipeline, "classification_model")

# Regression
y_reg = train_session['revenue']
with mlflow.start_run(run_name="Regression"):
    reg_pipeline = Pipeline([
        ('preprocessing', full_pipeline),
        ('regressor', GradientBoostingRegressor())
    ])

    reg_pipeline.fit(X, y_reg)
    reg_preds = reg_pipeline.predict(X)

    mlflow.log_metrics({
        'rmse': np.sqrt(mean_squared_error(y_reg, reg_preds)),
        'mae': mean_absolute_error(y_reg, reg_preds),
        'r2': r2_score(y_reg, reg_preds)
    })

    mlflow.sklearn.log_model(reg_pipeline, "regression_model")

# Clustering
# Clustering
with mlflow.start_run(run_name="Clustering"):
    preprocessed_data = full_pipeline.fit_transform(X)

    # Convert sparse to dense if necessary
    if sparse.issparse(preprocessed_data):
        preprocessed_data_dense = preprocessed_data.toarray()
    else:
        preprocessed_data_dense = preprocessed_data

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(preprocessed_data_dense)

    # Calculate silhouette score
    silhouette = silhouette_score(preprocessed_data_dense, clusters)

    # Calculate davies_bouldin_score (Now using dense data)
    davies_bouldin = davies_bouldin_score(preprocessed_data_dense, clusters)

    # Log metrics
    mlflow.log_metrics({
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin
    })

    # Save the KMeans model
    mlflow.sklearn.log_model(kmeans, "clustering_model")



# Model Saving
pickle.dump(class_pipeline, open('best_classifier.pkl', 'wb'))
pickle.dump(reg_pipeline, open('best_regressor.pkl', 'wb'))
pickle.dump(kmeans, open('best_clusterer.pkl', 'wb'))

#Streamlit Application
st.title('Customer Conversion Analysis')

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
manual_input = st.checkbox("Or enter data manually", key='checked')
#st.write("manual_input",manual_input)
input_df = pd.DataFrame()
# Initialize session state for checkbox
if 'checked' not in st.session_state:
    st.session_state.checked = False

# Function to reset checkbox
def reset_checkbox():
    st.session_state.checked = False
    manual_input =False

if uploaded_file is not None:
    #reset_checkbox()
    # Read the first few bytes to detect the separator
    sample = uploaded_file.read(1024).decode('utf-8')
    uploaded_file.seek(0)  # Reset pointer after reading

    # Determine separator based on sample content
    sep = ';' if ';' in sample else ','

    # Read CSV with the appropriate separator
    input_df = pd.read_csv(uploaded_file, sep=sep, skipinitialspace=True)
    columns_to_rename = {
    'session ID': 'session_id',
    'page 1 (main category)': 'page1_main_category',
    'page 2 (clothing model)': 'page2_clothing_model',
    'model photography':'model_photography',
    'price 2':'price_2'
    }

    for col in columns_to_rename:
        if col in input_df.columns:
            input_df.rename(columns={col: columns_to_rename[col]}, inplace=True)
        else:
            print(f"Column '{col}' does not exist.")
    #input_df = input_df[0].str.split(';', expand=True)
    st.write(input_df)
    #input_df = pd.read_csv(uploaded_file)
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
    
    processed_df = aggregate_session_data(create_targets(input_df))

    # Load models
    class_model = pickle.load(open('best_classifier.pkl', 'rb'))
    reg_model = pickle.load(open('best_regressor.pkl', 'rb'))
    cluster_model = pickle.load(open('best_clusterer.pkl', 'rb'))
    
    try:
        # Predictions
        conv_pred = class_model.predict(processed_df.drop(['session_id'], axis=1))
        revenue_pred = reg_model.predict(processed_df.drop(['session_id'], axis=1))
        clusters = cluster_model.predict(full_pipeline.transform(processed_df.drop(['session_id'], axis=1)))

        # Display results
        processed_df['Conversion Prediction'] = conv_pred
        processed_df['Revenue Estimate'] = revenue_pred
        processed_df['Customer Segment'] = clusters

        st.subheader('Predictions')
        st.dataframe(processed_df[['session_id', 'Conversion Prediction', 'Revenue Estimate', 'Customer Segment']])
    except Exception as e:
        st.error(f"Prediction Error: {e}")

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

else :
    st.warning("Please upload a CSV or enter data manually.")