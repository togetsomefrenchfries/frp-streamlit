# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 配置管理模块
Configuration Management for FRP Rebar Durability Prediction
"""

import os
from pathlib import Path

class Config:
    """配置类 - 管理所有配置参数"""
    
    # ================================
    # 基本路径配置
    # ================================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # 确保目录存在
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # ================================
    # 数据文件配置
    # ================================
    DEFAULT_DATA_FILE = r"E:\大学\intern\2025-summer-concret\database 4.xlsx"
    PROCESSED_DATA_FILE = DATA_DIR / "processed_data.csv"
    
    # ================================
    # 模型配置
    # ================================
    # 支持的模型类型
    SUPPORTED_MODELS = {
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM', 
        'random_forest': 'Random Forest',
        'voting': 'Voting Ensemble'
    }
    
    # 默认模型参数
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    # ================================
    # 特征工程配置
    # ================================
    # 核心特征列表（基于原代码的14个特征）
    CORE_FEATURES = [
        'pH_of_condition_enviroment',
        'condition_time', 
        'fiber_content',
        'Temperature',
        'diameter',
        'concrete',
        'load_value',
        'Chloride_ion',
        'Glass_or_Basalt',
        'Vinyl_ester_or_Epoxy',
        'surface_treatment',
        'max_strength'
    ]
    
    # 目标变量
    TARGET_VARIABLE = 'Tensile strength retention'
    
    # 数值特征和分类特征
    NUMERIC_FEATURES = [
        'pH_of_condition_enviroment', 'condition_time', 'fiber_content',
        'Temperature', 'diameter', 'load_value', 'max_strength'
    ]
    
    CATEGORICAL_FEATURES = [
        'concrete', 'Chloride_ion', 'Glass_or_Basalt', 
        'Vinyl_ester_or_Epoxy', 'surface_treatment'
    ]
    
    # ================================
    # 预处理配置
    # ================================
    # 缺失值处理策略
    MISSING_VALUE_STRATEGY = {
        'numeric': 'median',  # mean, median, zero
        'categorical': 'unknown'  # unknown, mode, drop
    }
    
    # 异常值处理
    OUTLIER_METHOD = 'iqr'  # iqr, zscore, none
    OUTLIER_THRESHOLD = 3
    
    # 特征缩放方法
    SCALING_METHOD = 'standard'  # standard, minmax, robust, none
    
    # ================================
    # 模型训练配置
    # ================================
    # 训练/验证/测试分割比例 (7:2:1)
    TEST_SIZE = 0.1      # 10% 测试集
    VALIDATION_SIZE = 0.2  # 20% 验证集 (从剩余90%中分出)
    TRAIN_SIZE = 0.7     # 70% 训练集
    
    # 交叉验证折数
    CV_FOLDS = 5
    
    # 超参数优化
    ENABLE_HYPERPARAMETER_TUNING = True
    TUNING_CV_FOLDS = 3
    TUNING_N_ITER = 50
    HYPERPARAMETER_SEARCH_METHOD = 'grid'  # 'grid', 'random', 'bayesian'
    
    # 超参数搜索空间
    HYPERPARAMETER_SPACES = {
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'num_leaves': [31, 50, 100]
        }
    }
    
    # ================================
    # 评估指标配置
    # ================================
    EVALUATION_METRICS = [
        'r2_score',
        'mean_squared_error', 
        'mean_absolute_error',
        'root_mean_squared_error'
    ]
    
    # ================================
    # 日志配置
    # ================================
    LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ================================
    # 绘图配置
    # ================================
    PLOT_STYLE = 'seaborn-v0_8'
    FIGURE_SIZE = (10, 6)
    DPI = 100
    
    # ================================
    # 数据库配置（可选，用于兼容原系统）
    # ================================
    DATABASE_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'frp_database')
    }
    
    # ================================
    # 材料属性配置（用于特征工程）
    # ================================
    MATERIAL_PROPERTIES = {
        'fiber_densities': {
            'Glass': 2.55,
            'Carbon': 1.84,
            'Basalt': 2.67
        },
        'matrix_densities': {
            'Vinyl ester': 1.09,
            'Epoxy': 1.1,
            'Polyester': 1.38
        }
    }
    
    @classmethod
    def get_model_params(cls, model_name):
        """获取指定模型的参数"""
        return cls.MODEL_PARAMS.get(model_name, {})
    
    @classmethod
    def get_feature_config(cls):
        """获取特征配置"""
        return {
            'core_features': cls.CORE_FEATURES,
            'numeric_features': cls.NUMERIC_FEATURES,
            'categorical_features': cls.CATEGORICAL_FEATURES,
            'target_variable': cls.TARGET_VARIABLE
        }
    
    @classmethod
    def validate_config(cls):
        """验证配置的完整性"""
        errors = []
        
        # 检查必要的目录
        if not cls.DATA_DIR.exists():
            errors.append(f"Data directory not found: {cls.DATA_DIR}")
            
        # 检查特征配置
        if not cls.CORE_FEATURES:
            errors.append("CORE_FEATURES cannot be empty")
            
        if not cls.TARGET_VARIABLE:
            errors.append("TARGET_VARIABLE must be specified")
            
        # 检查模型配置
        if not cls.SUPPORTED_MODELS:
            errors.append("SUPPORTED_MODELS cannot be empty")
            
        return errors

# 默认配置实例
config = Config()

# 配置验证
validation_errors = config.validate_config()
if validation_errors:
    print("Warning: Configuration validation errors:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("Configuration validation passed")