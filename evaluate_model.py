import joblib
from sklearn.metrics import classification_report, mean_squared_error

def evaluate_model(X_test, y_test):
    model = joblib.load("best_model.pkl")

    predictions = model.predict(X_test)
    if len(set(y_test)) < 10:  # Classification
        print(classification_report(y_test, predictions))
    else:  # Regression
        print(f"MSE: {mean_squared_error(y_test, predictions)}")
