

AutoML Predictor

AutoML Predictor is a Python library which automatically identifies the optimum machine learning model for a given dataset. It is capable of both classification and regression problems while dealing with data preprocessing, model selection, as well as model evaluation.

🔧 Installation

pip install -r requirements.txt

🚀 Usage

import pandas as pd
from src.model_selection import recommend_model

# Load dataset
data = pd.read_csv("dataset.csv")

# Get best model recommendation
best_model = recommend_model(data, target_column="label")
print(f"Best Selected Model: {best_model}")

✨ Features

Automatic Model Selection (Classification & Regression)

Preprocessing (Handles missing values, encoding, scaling)

Multiple Algorithms (Random Forest, SVM, Logistic Regression, etc.)

Performance Evaluation using Cross-Validation


📂 Folder Structure

AutoML-Predictor/
│── dataset/               # Sample datasets  
│── notebooks/             # Jupyter notebooks for analysis  
│── src/                   # Main scripts  
│   ├── data_preprocessing.py  
│   ├── model_selection.py  
│   ├── train_model.py  
│   ├── evaluate_model.py  
│── requirements.txt       # Dependencies  
│── README.md              # Documentation  
│── .gitignore             # Ignored files  
│── setup.py               # Packaging (optional)

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss improvements.

📜 License

This project is open-source !!

