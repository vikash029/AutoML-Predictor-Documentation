import joblib
from sklearn.model_selection import train_test_split

def train_and_save_model(data, target_column):
    from src.model_selection import recommend_model
    from src.data_preprocessing import preprocess_data

    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Get the best model
    best_model_name = recommend_model(data, target_column)
    model = eval(best_model_name)()

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "best_model.pkl")
    print(f"✅ Model saved as best_model.pkl")
