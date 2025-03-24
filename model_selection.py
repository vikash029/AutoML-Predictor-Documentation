from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np

def recommend_model(data, target_column):
    from src.data_preprocessing import preprocess_data

    # Preprocess Data
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Detect problem type
    problem_type = 'classification' if len(set(y)) < 10 else 'regression'

    # Define models
    models = {
        "classification": {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "k-NN": KNeighborsClassifier()
        },
        "regression": {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "k-NN Regressor": KNeighborsRegressor()
        }
    }

    selected_models = models[problem_type]
    results = {}

    for name, model in selected_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy" if problem_type == "classification" else "neg_mean_squared_error")
        results[name] = np.mean(scores)
    
    best_model = max(results, key=results.get) if problem_type == "classification" else min(results, key=results.get)
    return best_model
