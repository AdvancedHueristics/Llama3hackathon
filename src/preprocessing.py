import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_missing_values(df, method="Simple"):
    logger.info(f"Handling missing values using method: {method}")

    if method == "Drop Rows":
        df = df.dropna()
    elif method == "Drop Columns":
        df = df.dropna(axis=1)
    elif method == "Simple":
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=['object']).columns] = cat_imputer.fit_transform(df.select_dtypes(include=['object']))
    elif method == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        logger.warning(f"Unknown method: {method}. No changes applied.")
    
    return df

def remove_duplicates(df):
    logger.info("Removing duplicate rows")
    return df.drop_duplicates()

def encode_categorical(df, method="Label"):
    logger.info(f"Encoding categorical features using method: {method}")
    
    if method == "Label":
        le = LabelEncoder()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = le.fit_transform(df[column].astype(str))
    elif method == "OneHot":
        df = pd.get_dummies(df, drop_first=True)
    else:
        logger.warning(f"Unknown encoding method: {method}. No encoding applied.")
    
    return df

def handle_outliers(df, method="IQR"):
    logger.info(f"Handling outliers using method: {method}")

    if method == "IQR":
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == "Z-Score":
        from scipy.stats import zscore
        df = df[(np.abs(zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
    else:
        logger.warning(f"Unknown outlier handling method: {method}. No changes applied.")
    
    return df

def scale_and_normalize(df, method="Standard"):
    logger.info(f"Scaling and normalizing data using method: {method}")

    if method == "Standard":
        scaler = StandardScaler()
    elif method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Robust":
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown scaling method: {method}. No scaling applied.")
        return df

    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df
