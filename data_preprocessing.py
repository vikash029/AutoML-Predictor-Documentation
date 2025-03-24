import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical target (if classification)
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Scale numerical features
    X = StandardScaler().fit_transform(X)
    
    return X, y
