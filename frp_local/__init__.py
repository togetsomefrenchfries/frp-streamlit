# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 本地版本
FRP Rebar Durability Prediction - Local Version

一个完整的机器学习系统，用于预测FRP钢筋在各种环境条件下的耐久性
"""

__version__ = "1.0.0"
__author__ = "FRP Research Team"
__email__ = "frp@research.com"

# 主要模块导入
from .config import config
from .data_loader import DataLoader, load_default_data
from .preprocessor import FRPDataPreprocessor, preprocess_frp_data
from .model_trainer import ModelTrainer, train_frp_models
from .predictor import FRPPredictor, FRPPredictionPipeline, create_sample_input
from .utils import apply_sklearn_compatibility_patch

# 版本信息
__all__ = [
    'config',
    'DataLoader',
    'load_default_data', 
    'FRPDataPreprocessor',
    'preprocess_frp_data',
    'ModelTrainer',
    'train_frp_models',
    'FRPPredictor',
    'FRPPredictionPipeline',
    'create_sample_input',
    'apply_sklearn_compatibility_patch'
]