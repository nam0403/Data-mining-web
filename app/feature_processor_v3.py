# =============================================================================
# CONSTANTS - Features có vấn đề cần loại bỏ
# =============================================================================

# Features có bug công thức
import pickle
import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder

BUG_FEATURES = [
    "p90_minus_cutoff_lag1",  # Công thức sai: = -cutoff_lag_1
    "p95_minus_cutoff_lag1",  # Công thức sai: = -cutoff_lag_1
    "cutoff_difficulty_ratio",  # Nhân 1e6 thay vì tính ratio
    "mean_x_cutoff_lag1",  # Luôn = 0
    "trend_alignment",  # Luôn = 0
    "competition_intensity",  # Luôn = 0
    "estimate_from_p90",  # Luôn = 1
]

# Features duplicate
DUPLICATE_FEATURES = [
    "cutoff_lag1_value",  # = cutoff_lag_1
    "estimate_naive",  # = cutoff_lag_1
    "major_avg_cutoff_all_time",  # = cutoff_avg_all
]

# Features constant hoặc không có ý nghĩa
CONSTANT_FEATURES = [
    "score_min_lag1",  # Luôn = 0
]

# Features gây multicollinearity cao (giữ cutoff_lag_1, bỏ các cái redundant)
REDUNDANT_FEATURES = [
    "cutoff_avg_all",  # corr 0.96 với cutoff_lag_1
    "cutoff_avg_5y",  # corr 0.97 với cutoff_avg_3y
    "cutoff_std_5y",  # corr cao với cutoff_std_3y
    "cutoff_lag_4",  # Quá xa, nhiều NaN
    "cutoff_lag_5",  # Quá xa, nhiều NaN
    "score_percentile_95_lag1",  # corr 0.99 với p90
    "score_percentile_99_lag1",  # corr cao với p90, p95
    "cutoff_max_history",  # Gây overfitting (importance 61%!)
]

# Identifier columns không dùng làm features
IDENTIFIER_COLS = [
    "major_key",
    "target_year",
    "diem_chuan",
    "university",
    "major_name",
    "major_code",
    "year",  # Duplicate với target_year
]

# Tổng hợp tất cả features cần loại bỏ
DROP_FEATURES = BUG_FEATURES + DUPLICATE_FEATURES + CONSTANT_FEATURES + REDUNDANT_FEATURES

# Ngưỡng NaN để loại bỏ feature
HIGH_NAN_THRESHOLD = 0.7


# =============================================================================
# DATA CLEANING
# =============================================================================


def clean_outliers(df: pd.DataFrame, min_cutoff: float = 14.0, max_cutoff: float = 30.0, verbose: bool = True) -> pd.DataFrame:
    """
    Xử lý outlier trong diem_chuan

    Điểm chuẩn hợp lệ thường trong khoảng 14-30
    (14 = điểm sàn, 30 = max thường, 30 = có ưu tiên)
    """
    df = df.copy()

    # Đếm outliers
    outlier_mask = (df["diem_chuan"] < min_cutoff) | (df["diem_chuan"] > max_cutoff)
    n_outliers = outlier_mask.sum()

    if verbose and n_outliers > 0:
        print(f"\n   ⚠ Found {n_outliers:,} outliers ({n_outliers/len(df)*100:.2f}%)")
        print(f"     diem_chuan range before: [{df['diem_chuan'].min():.2f}, {df['diem_chuan'].max():.2f}]")

        # Hiển thị một số outliers
        outliers = df[outlier_mask]["diem_chuan"].describe()
        print(f"     Outlier stats: min={outliers['min']:.2f}, max={outliers['max']:.2f}")

    # Loại bỏ outliers
    df_clean = df[~outlier_mask].reset_index(drop=True)

    if verbose:
        print(f"     diem_chuan range after:  [{df_clean['diem_chuan'].min():.2f}, {df_clean['diem_chuan'].max():.2f}]")
        print(f"     Samples remaining: {len(df_clean):,}")

    return df_clean


# =============================================================================
# FEATURE PROCESSOR (Improved)
# =============================================================================


class FeatureProcessorV3:
    """
    Feature Processor cải tiến:
    - Lọc features có vấn đề
    - Lọc features có quá nhiều NaN
    - Lọc features có variance = 0
    - Thêm feature engineering
    """

    def __init__(self, drop_features: List[str] = None, nan_threshold: float = HIGH_NAN_THRESHOLD, add_engineered_features: bool = True):

        self.drop_features = drop_features or DROP_FEATURES
        self.nan_threshold = nan_threshold
        self.add_engineered_features = add_engineered_features

        self.combination_encoder = LabelEncoder()
        self.feature_cols = []
        self.numeric_medians = {}
        self.high_nan_cols = []
        self.zero_var_cols = []
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các features mới có ý nghĩa"""
        df = df.copy()

        # 1. Momentum features (cải tiến)
        if "cutoff_lag_1" in df.columns and "cutoff_lag_2" in df.columns:
            df["momentum_1y"] = df["cutoff_lag_1"] - df["cutoff_lag_2"]

        if "cutoff_lag_2" in df.columns and "cutoff_lag_3" in df.columns:
            df["momentum_2y"] = df["cutoff_lag_2"] - df["cutoff_lag_3"]

        # 2. Acceleration
        if "momentum_1y" in df.columns and "momentum_2y" in df.columns:
            df["acceleration"] = df["momentum_1y"] - df["momentum_2y"]

        # 3. Relative position
        if "cutoff_lag_1" in df.columns and "cutoff_avg_3y" in df.columns:
            df["cutoff_vs_avg3y"] = df["cutoff_lag_1"] - df["cutoff_avg_3y"]

        # 4. Volatility-adjusted z-score
        if "cutoff_std_3y" in df.columns and "cutoff_avg_3y" in df.columns:
            df["zscore_3y"] = (df["cutoff_lag_1"] - df["cutoff_avg_3y"]) / (df["cutoff_std_3y"] + 0.1)

        # 5. Distance to historical bounds
        if "cutoff_max_history" in df.columns and "cutoff_lag_1" in df.columns:
            df["distance_to_max"] = df["cutoff_max_history"] - df["cutoff_lag_1"]

        if "cutoff_min_history" in df.columns and "cutoff_lag_1" in df.columns:
            df["distance_to_min"] = df["cutoff_lag_1"] - df["cutoff_min_history"]

        # 6. Score-based difficulty estimate (sửa công thức đúng)
        if "score_percentile_90_lag1" in df.columns and "cutoff_lag_1" in df.columns:
            df["p90_vs_cutoff"] = df["score_percentile_90_lag1"] - df["cutoff_lag_1"]

        # 7. Trend strength
        if "cutoff_trend_coef" in df.columns and "cutoff_trend_r2" in df.columns:
            df["trend_strength"] = df["cutoff_trend_coef"] * df["cutoff_trend_r2"]

        # 8. Stability indicator
        if "cutoff_std_3y" in df.columns:
            df["is_stable"] = (df["cutoff_std_3y"] < 1.0).astype(int)

        return df

    def fit(self, df: pd.DataFrame) -> "FeatureProcessorV3":
        """Fit encoder và xác định features"""
        print("\n   Feature filtering:")

        # Add engineered features
        if self.add_engineered_features:
            df = self._engineer_features(df)

        # Encode combination
        self.combination_encoder.fit(df["combination"].fillna("UNKNOWN"))

        # Lấy tất cả numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Bước 1: Loại bỏ identifier columns
        feature_cols = [c for c in numeric_cols if c not in IDENTIFIER_COLS]
        print(f"     After removing identifiers: {len(feature_cols)} features")

        # Bước 2: Loại bỏ features có vấn đề (bugs, duplicates, etc.)
        feature_cols = [c for c in feature_cols if c not in self.drop_features]
        print(f"     After removing problematic: {len(feature_cols)} features")

        # Bước 3: Loại bỏ features có quá nhiều NaN
        nan_ratios = df[feature_cols].isna().sum() / len(df)
        self.high_nan_cols = nan_ratios[nan_ratios > self.nan_threshold].index.tolist()
        feature_cols = [c for c in feature_cols if c not in self.high_nan_cols]
        print(f"     After removing high NaN (>{self.nan_threshold*100:.0f}%): {len(feature_cols)} features")

        # Bước 4: Loại bỏ features có variance = 0
        self.zero_var_cols = []
        for col in feature_cols:
            if df[col].std() == 0 or df[col].nunique() <= 1:
                self.zero_var_cols.append(col)
        feature_cols = [c for c in feature_cols if c not in self.zero_var_cols]
        print(f"     After removing zero variance: {len(feature_cols)} features")

        self.feature_cols = feature_cols

        # Thêm combination_encoded
        if "combination_encoded" not in self.feature_cols:
            self.feature_cols.append("combination_encoded")

        # Lưu median cho missing values
        for col in self.feature_cols:
            if col in df.columns and col != "combination_encoded":
                self.numeric_medians[col] = df[col].median()

        self.is_fitted = True
        print(f"     Final features: {len(self.feature_cols)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        if not self.is_fitted:
            raise ValueError("FeatureProcessor chưa được fit!")

        df = df.copy()

        # Add engineered features
        if self.add_engineered_features:
            df = self._engineer_features(df)

        # Encode combination
        df["combination_encoded"] = self.combination_encoder.transform(df["combination"].fillna("UNKNOWN"))

        # Select features (chỉ lấy những features có trong data)
        available_features = [c for c in self.feature_cols if c in df.columns]
        X = df[available_features].copy()

        # Fill missing values với median
        for col in X.columns:
            if col in self.numeric_medians:
                X[col] = X[col].fillna(self.numeric_medians[col])
            else:
                X[col] = X[col].fillna(0)

        # Handle infinity
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if col in self.numeric_medians:
                X[col] = X[col].fillna(self.numeric_medians[col])
            else:
                X[col] = X[col].fillna(0)

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, filepath: str):
        """Lưu processor"""
        data = {
            "combination_encoder": self.combination_encoder,
            "feature_cols": self.feature_cols,
            "numeric_medians": self.numeric_medians,
            "high_nan_cols": self.high_nan_cols,
            "zero_var_cols": self.zero_var_cols,
            "drop_features": self.drop_features,
            "nan_threshold": self.nan_threshold,
            "add_engineered_features": self.add_engineered_features,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "FeatureProcessorV3":
        """Load processor"""
        processor = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        processor.combination_encoder = data["combination_encoder"]
        processor.feature_cols = data["feature_cols"]
        processor.numeric_medians = data["numeric_medians"]
        processor.high_nan_cols = data.get("high_nan_cols", [])
        processor.zero_var_cols = data.get("zero_var_cols", [])
        processor.drop_features = data.get("drop_features", DROP_FEATURES)
        processor.nan_threshold = data.get("nan_threshold", HIGH_NAN_THRESHOLD)
        processor.add_engineered_features = data.get("add_engineered_features", True)
        processor.is_fitted = data["is_fitted"]

        return processor