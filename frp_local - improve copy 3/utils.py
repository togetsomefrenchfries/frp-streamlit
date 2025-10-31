# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 工具函数模块
Utility Functions for FRP Rebar Durability Prediction

包含：
- sklearn兼容性补丁
- 安全模型加载
- 通用工具函数
"""

import base64
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

def apply_sklearn_compatibility_patch():
    """Apply patches for sklearn compatibility issues"""
    try:
        from sklearn.compose import _column_transformer
        
        # Patch missing _RemainderColsList if needed
        if not hasattr(_column_transformer, '_RemainderColsList'):
            class _RemainderColsList(list):
                """Compatibility class for older sklearn versions"""
                def __init__(self, remainder_columns):
                    super().__init__(remainder_columns)
                    self.remainder_columns = remainder_columns
            
            _column_transformer._RemainderColsList = _RemainderColsList
            print("Applied _RemainderColsList compatibility patch")
        
        return True
    except Exception as e:
        print(f"Failed to apply compatibility patch: {e}")
        return False

def safe_pickle_load(pickled_data):
    """Safely load pickled model data with version compatibility handling"""
    try:
        # First try normal loading
        return pickle.loads(pickled_data)
    except AttributeError as e:
        # Handle sklearn version compatibility issues
        if "_RemainderColsList" in str(e):
            apply_sklearn_compatibility_patch()
            try:
                return pickle.loads(pickled_data)
            except Exception as retry_e:
                print(f"Retry after patch failed: {retry_e}")
                return None
        else:
            print(f"Pickle loading failed: {e}")
            return None
    except Exception as e:
        print(f"General pickle loading error: {e}")
        return None

def load_model_from_base64(base64_data):
    """Load model from base64 encoded pickle data with error handling"""
    try:
        pickled_data = base64.b64decode(base64_data)
        return safe_pickle_load(pickled_data)
    except Exception as e:
        print(f"Base64 model loading failed: {e}")
        return None

def global_clean_categorical_features(X):
    """Global function to clean categorical features - can be pickled"""
    X_clean = X.copy()
    for col in X_clean.columns:
        X_clean[col] = X_clean[col].fillna('unknown').astype(str)
    return X_clean

def create_enhanced_preprocessor(categorical_cols, numeric_cols, add_polynomial=True, polynomial_degree=2):
    """Create enhanced preprocessor with optional polynomial features"""
    
    # Enhanced categorical transformer
    categorical_transformer = Pipeline([
        ('cleaner', FunctionTransformer(global_clean_categorical_features, validate=False)),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    
    # Enhanced numeric transformer with optional polynomial features and feature selection
    if add_polynomial and len(numeric_cols) > 1:  # Only add if multiple numeric features
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=polynomial_degree, interaction_only=True, include_bias=False)),
            ('selector', SelectKBest(f_regression, k=min(50, len(numeric_cols)*2)))  # Limit features
        ])
    else:
        # Use feature selection even without polynomial features
        if len(numeric_cols) > 10:  # Only if many features
            numeric_transformer = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_regression, k=min(20, len(numeric_cols))))  # Select best features
            ])
        else:
            numeric_transformer = StandardScaler()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ] if categorical_cols else [
            ('num', numeric_transformer, numeric_cols)
        ]
    )
    
    return preprocessor

def diagnose_model_performance(y_true, y_pred, model_name="Model"):
    """Diagnose model performance and return insights"""
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Check for patterns in residuals
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    
    diagnosis = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'residual_bias': residual_mean,
        'residual_std': residual_std,
        'model_name': model_name
    }
    
    return diagnosis

def print_model_performance(diagnosis):
    """打印模型性能诊断结果"""
    print(f"\n📊 {diagnosis['model_name']} Performance Report:")
    print("=" * 50)
    print(f"R² Score:      {diagnosis['r2']:.4f}")
    print(f"RMSE:          {diagnosis['rmse']:.4f}")
    print(f"MAE:           {diagnosis['mae']:.4f}")
    print(f"MSE:           {diagnosis['mse']:.4f}")
    print(f"Residual Bias: {diagnosis['residual_bias']:.4f}")
    print(f"Residual Std:  {diagnosis['residual_std']:.4f}")
    print("=" * 50)

def validate_dataframe(df, required_columns=None, name="DataFrame"):
    """验证DataFrame的完整性"""
    if df is None:
        raise ValueError(f"{name} is None")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} missing required columns: {missing_cols}")
    
    print(f"✅ {name} validation passed: {df.shape}")
    return True

def safe_convert_to_numeric(series, default_value=0):
    """安全地将series转换为数值类型"""
    try:
        # 尝试直接转换
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # 填充NaN值
        numeric_series = numeric_series.fillna(default_value)
        
        return numeric_series
    except Exception as e:
        print(f"Warning: Failed to convert to numeric: {e}")
        return pd.Series([default_value] * len(series), index=series.index)

def clean_column_names(df):
    """清理DataFrame的列名"""
    # 移除前后空格
    df.columns = df.columns.str.strip()
    
    # 替换特殊字符
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
    
    return df

def save_model_safely(model, filepath, additional_info=None):
    """安全地保存模型"""
    import pickle
    import os
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        save_data = {
            'model': model,
            'additional_info': additional_info,
            'timestamp': pd.Timestamp.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Model saved successfully: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def load_model_safely(filepath):
    """安全地加载模型"""
    import pickle
    
    try:
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        if isinstance(save_data, dict) and 'model' in save_data:
            print(f"✅ Model loaded successfully: {filepath}")
            return save_data['model'], save_data.get('additional_info')
        else:
            # 向后兼容：直接是模型对象
            print(f"✅ Legacy model loaded: {filepath}")
            return save_data, None
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def train_validation_test_split(X, y, test_size=0.1, val_size=0.2, random_state=42):
    """
    将数据按照7:2:1的比例分割为训练集、验证集、测试集
    
    Args:
        X: 特征数据
        y: 目标变量
        test_size: 测试集比例 (默认0.1, 即10%)
        val_size: 验证集比例 (从剩余数据中的比例, 默认0.2, 即22.2%的总数据)
        random_state: 随机种子
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # 第一次分割: 分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 第二次分割: 从剩余数据中分出验证集
    # val_size需要调整为在剩余数据中的比例
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"📊 数据分割结果 (7:2:1):")
    print(f"   训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"   总计: {len(X)} 样本")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# 在模块导入时应用兼容性补丁
apply_sklearn_compatibility_patch()