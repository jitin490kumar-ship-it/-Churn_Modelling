# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')


# Features (Independent Variables)
X = dataset.iloc[:, 3:-1].values
print(X)
# Target (Dependent Variable)
y = dataset.iloc[:, -1].values
print(y)


# Encoding the Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One-Hot Encoding the Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


