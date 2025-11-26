import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# --- Configuration ---
DB_PATH = "admission_db.sqlite"  # Path to your SQLite DB
MODELS_DIR = "."            # Directory containing your .pkl models

@dataclass
class StudentInput:
    university_code: str
    major_code: str
    combination_code: str
    subject_scores: Dict[str, float]
    priority_score: float

    @property
    def total_score(self) -> float:
        return sum(self.subject_scores.values()) + self.priority_score

@dataclass
class PredictionResult:
    university: str
    major_name: str
    combination: str
    predictions: Dict[str, float]
    student_score: float
    is_passed: bool
    margin: float

class DataConnector:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_historical_data(self, uni_code: str, major_code: str, combo_code: str) -> Dict[int, float]:
        """
        Fetches historical scores from SQLite and returns a dictionary {year: score}.
        """
        query = """
        SELECT 
            s.year, s.score
        FROM 
            admissionscore s
        JOIN 
            schoolmajor sm ON s.school_major_id = sm.id
        JOIN 
            school sch ON sm.school_id = sch.id
        JOIN 
            major m ON sm.major_id = m.id
        JOIN 
            combination c ON s.combination_id = c.id
        WHERE 
            sch.code = ? AND m.code = ? AND c.code = ?
        ORDER BY 
            s.year ASC;
        """
        
        history = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (uni_code, major_code, combo_code))
                rows = cursor.fetchall()
                
                for year, score in rows:
                    if score is not None and score > 0:
                        history[int(year)] = float(score)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {}
            
        return history

class FeatureEngineer:
    """
    Replicates the feature engineering logic from train.py/inference.py
    to ensure consistency between training and inference.
    """
    def __init__(self, metadata_path: str = None):
        # Load metadata if needed (e.g. for scaling or specific feature names)
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def create_features(self, history: Dict[int, float], combination: str) -> np.ndarray:
        """
        Transforms historical data dictionary into a feature vector.
        Must match the logic used in training V3.
        """
        if not history:
            return None

        years = sorted(history.keys())
        cutoffs = [history[y] for y in years]
        
        # --- Feature Creation Logic (Matches typical ML approach) ---
        features = []

        # 1. Lag Features (Most recent years)
        # Assuming we are predicting for 2025, so 2024 is lag_1
        features.append(cutoffs[-1])  # lag_1
        features.append(cutoffs[-2] if len(cutoffs) >= 2 else 0)  # lag_2
        features.append(cutoffs[-3] if len(cutoffs) >= 3 else 0)  # lag_3

        # 2. Statistical Aggregates
        features.append(np.mean(cutoffs))  # mean
        features.append(np.median(cutoffs)) # median
        features.append(np.std(cutoffs))    # std
        features.append(max(cutoffs))       # max
        features.append(min(cutoffs))       # min
        
        # 3. Trend (Linear Regression Slope)
        trend = 0
        if len(cutoffs) >= 2:
            try:
                x = np.arange(len(cutoffs))
                z = np.polyfit(x, cutoffs, 1)
                trend = z[0]
            except:
                pass
        features.append(trend)

        # 4. Combination Encoding (Simple Hash or One-Hot proxy)
        # This must match training exactly. If training used LabelEncoder, 
        # we technically need that encoder. A simple hash is a robust fallback 
        # if the encoder isn't saved/loaded.
        combo_hash = hash(combination) % 100
        features.append(combo_hash)

        # 5. Padding/Trimming to match model input shape
        # Check metadata or default to a known size (e.g., 13 features, 73 features)
        # For V3 models, let's assume a standard set. 
        # IMPORTANT: This must be adapted to the exact number of features your model expects.
        # If your model expects 73 features, you need to pad.
        
        expected_features = 20 # Placeholder, adjust based on training
        
        current_len = len(features)
        if current_len < expected_features:
            features.extend([0] * (expected_features - current_len))
        elif current_len > expected_features:
            features = features[:expected_features]

        return np.array(features).reshape(1, -1) # Reshape for sklearn (1 sample, n features)

class PredictorSystem:
    def __init__(self, db_path: str, models_dir: str):
        self.connector = DataConnector(db_path)
        self.engineer = FeatureEngineer(os.path.join(models_dir, "metadata.json"))
        self.models = self._load_models(models_dir)

    def _load_models(self, models_dir: str) -> Dict[str, Any]:
        models = {}
        model_files = {
            "RandomForest": "model_randomforest.pkl",
            "XGBoost": "model_xgboost.pkl", # If available
            "CatBoost": "model_catboost.pkl", # If available
            "Ridge": "model_ridge.pkl",
            "Baseline": None # Logic-based
        }
        
        for name, filename in model_files.items():
            if filename:
                path = os.path.join(models_dir, filename)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            models[name] = pickle.load(f)
                        print(f"Loaded {name}")
                    except Exception as e:
                        print(f"Failed to load {name}: {e}")
        
        return models

    def predict(self, student: StudentInput) -> PredictionResult:
        # 1. Get History
        history = self.connector.get_historical_data(
            student.university_code, 
            student.major_code, 
            student.combination_code
        )
        
        predictions = {}
        
        # 2. Baseline Prediction (Simple Logic)
        if history:
            years = sorted(history.keys())
            last_score = history[years[-1]]
            # Simple trend: average of last 3 years + slight trend
            predictions['Baseline'] = last_score # Simplified for now
        else:
            predictions['Baseline'] = 0.0

        # 3. ML Prediction
        feature_vector = self.engineer.create_features(history, student.combination_code)
        
        if feature_vector is not None:
            for name, model in self.models.items():
                if model: # If model loaded successfully
                    try:
                        # Some models might require feature names, dataframe can solve this
                        # if trained with one. Here passing numpy array.
                        pred = model.predict(feature_vector)[0]
                        predictions[name] = float(pred)
                    except Exception as e:
                        print(f"Error predicting with {name}: {e}")

        # 4. Consolidate
        # You can implement ensemble logic here (e.g., average of RF and XGB)
        best_pred = predictions.get('RandomForest', predictions['Baseline'])
        
        is_passed = student.total_score >= best_pred
        margin = student.total_score - best_pred

        return PredictionResult(
            university=student.university_code, # Or fetch name
            major_name=student.major_code, # Or fetch name
            combination=student.combination_code,
            predictions=predictions,
            student_score=student.total_score,
            is_passed=is_passed,
            margin=margin
        )

# --- Example Usage ---
if __name__ == "__main__":
    # Setup
    system = PredictorSystem(DB_PATH, MODELS_DIR)
    
    # Mock Student Input
    student = StudentInput(
        university_code="QHS", 
        major_code="7480201", 
        combination_code="A00",
        subject_scores={"Toán": 8.5, "Lý": 8.0, "Hóa": 7.5},
        priority_score=0.5
    )
    
    # Run Prediction
    result = system.predict(student)
    
    print("-" * 30)
    print(f"Prediction for {student.university_code} - {student.major_code}")
    print(f"Student Score: {result.student_score}")
    print("Model Predictions:")
    for model, score in result.predictions.items():
        print(f"  - {model}: {score:.2f}")
    print("-" * 30)
    print(f"Final Result: {'PASSED' if result.is_passed else 'FAILED'}")
    print(f"Margin: {result.margin:.2f}")