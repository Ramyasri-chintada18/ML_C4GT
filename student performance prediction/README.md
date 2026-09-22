# Student Performance Prediction

A machine learning project to predict student academic performance based on demographic, socio-economic, and academic factors.

## Project Structure

```
student-performance-prediction/
│
├── venv/                      # Virtual environment
│
├── data/
│   └── student_data.csv       # Raw / processed dataset
│
├── notebooks/
│   └── student_prediction.ipynb # Jupyter notebook for EDA & modeling
│
├── models/                    # Saved trained model artifacts
│
├── src/                       # Reusable source code and modules
│
├── requirements.txt           # Project dependencies
│
└── README.md                  # Project overview and instructions
```

## Setup Instructions

1. **Activate Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Notebook**:
   ```bash
   jupyter notebook notebooks/student_prediction.ipynb
   ```
