## Loading Data

from types import new_class
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

## Data Preprocessing
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the parameters of a scikit-learn LogisticRegression model."""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Sets the parameters of a scikit-learnLogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But the server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 3
    n_features = 34  # Number of features in dataset
    model.classes_ = np.array([i for i in range(3)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data_client1() -> Dataset:
    ## Load Dataset
    fdf = pd.read_csv('data/Train Data/Train Data Zip/frequency_domain_features_train.csv')
    hrn = pd.read_csv('data/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv')
    tdf = pd.read_csv('data/Train Data/Train Data Zip/time_domain_features_train.csv')

    train_df = pd.merge(fdf, hrn, on='uuid')
    train_df = pd.merge(train_df, tdf, on='uuid')

    ## Test Data
    tfdf = pd.read_csv('data/Test Data/Test Zip/frequency_domain_features_test.csv')
    thrn = pd.read_csv('data/Test Data/Test Zip/heart_rate_non_linear_features_test.csv')
    ttdf = pd.read_csv('data/Test Data/Test Zip/time_domain_features_test.csv')

    test_df = pd.merge(tfdf, thrn, on='uuid')
    test_df = pd.merge(test_df, ttdf, on='uuid')

    df = pd.concat([train_df, test_df])
    df = df.drop(['uuid', 'HR'], axis=1)
    #df = df[0:3000]

    ## Label Encoder
    lb = LabelEncoder()  ## Encoder that convert and store all the information
    df['condition'] = lb.fit_transform(df['condition'])

    x = df.drop('condition', axis=1)
    y = df.condition

    # Standardizing the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

def load_data() -> Dataset:
    ## Load Dataset
    fdf = pd.read_csv('data/Train Data/Train Data Zip/frequency_domain_features_train.csv')
    hrn = pd.read_csv('data/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv')
    tdf = pd.read_csv('data/Train Data/Train Data Zip/time_domain_features_train.csv')

    train_df = pd.merge(fdf, hrn, on='uuid')
    train_df = pd.merge(train_df, tdf, on='uuid')

    ## Test Data
    tfdf = pd.read_csv('data/Test Data/Test Zip/frequency_domain_features_test.csv')
    thrn = pd.read_csv('data/Test Data/Test Zip/heart_rate_non_linear_features_test.csv')
    ttdf = pd.read_csv('data/Test Data/Test Zip/time_domain_features_test.csv')

    test_df = pd.merge(tfdf, thrn, on='uuid')
    test_df = pd.merge(test_df, ttdf, on='uuid')

    df = pd.concat([train_df, test_df])
    df = df.drop(['uuid', 'HR'], axis=1)
    #df = df[0:3000]

    ## Label Encoder
    lb = LabelEncoder()  ## Encoder that convert and store all the information
    df['condition'] = lb.fit_transform(df['condition'])

    x = df.drop('condition', axis=1)
    y = df.condition

    # Standardizing the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

def load_data_client2() -> Dataset:
    ## Load Dataset
    fdf = pd.read_csv('data/Train Data/Train Data Zip/frequency_domain_features_train.csv')
    hrn = pd.read_csv('data/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv')
    tdf = pd.read_csv('data/Train Data/Train Data Zip/time_domain_features_train.csv')

    train_df = pd.merge(fdf, hrn, on='uuid')
    train_df = pd.merge(train_df, tdf, on='uuid')

    ## Test Data
    tfdf = pd.read_csv('data/Test Data/Test Zip/frequency_domain_features_test.csv')
    thrn = pd.read_csv('data/Test Data/Test Zip/heart_rate_non_linear_features_test.csv')
    ttdf = pd.read_csv('data/Test Data/Test Zip/time_domain_features_test.csv')

    test_df = pd.merge(tfdf, thrn, on='uuid')
    test_df = pd.merge(test_df, ttdf, on='uuid')

    df = pd.concat([train_df, test_df])
    df = df.drop(['uuid', 'HR'], axis=1)
   # df = df[0:3000]

    ## Label Encoder
    lb = LabelEncoder()  ## Encoder that convert and store all the information
    df['condition'] = lb.fit_transform(df['condition'])

    x = df.drop('condition', axis=1)
    y = df.condition

    # Standardizing the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    """ Select the 80% of the data as Training data and 20% as test data """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (x_train, y_train), (x_test, y_test)

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y Datasets"""
    randon_gen = np.random.default_rng()
    perm = randon_gen.permutation(len(X))
    return X[perm], y[perm]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y Datasets into a variety of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )