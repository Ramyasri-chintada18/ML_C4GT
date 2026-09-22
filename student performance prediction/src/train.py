import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_save_model():
    # Base paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "Student_Performance.csv")
    models_dir = os.path.join(project_root, "models")
    model_save_path = os.path.join(models_dir, "student_performance_model.pkl")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Encode categorical variable
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    # Define features and target
    features = [
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]
    X = df[features]
    y = df['Performance Index']

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation on Test Set:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save trained model
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to: {model_save_path}")

if __name__ == "__main__":
    train_and_save_model()
