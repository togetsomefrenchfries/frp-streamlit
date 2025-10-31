# -*- coding: utf-8 -*-
"""
FRP 钢筋耐久性预测 - 预测模块
Prediction Module for FRP Rebar Durability Prediction

包含：
- 模型加载
- 特征标准化
- 预测功能
- 结果解释
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from .config import config
from .utils import load_model_safely, validate_dataframe
from .preprocessor import FRPDataPreprocessor

class FRPPredictor:
    """FRP耐久性预测器 - 用于加载模型并进行预测"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
        """
        self.model = None
        self.feature_info = None
        self.model_metadata = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        
        try:
            model_data, additional_info = load_model_safely(model_path)
            
            if model_data is None:
                print(f"❌ Failed to load model from {model_path}")
                return False
            
            # 处理不同的保存格式
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_info = model_data.get('feature_info')
                self.model_metadata = model_data.get('metrics')
            else:
                # 向后兼容：直接是模型对象
                self.model = model_data
                self.feature_info = additional_info
            
            if self.model is None:
                print(f"❌ No valid model found in {model_path}")
                return False
            
            print(f"✅ Model loaded successfully from {model_path}")
            
            # 打印模型信息
            if self.model_metadata:
                r2_score = self.model_metadata.get('r2', 'Unknown')
                print(f"   Model R² Score: {r2_score}")
            
            if self.feature_info:
                feature_count = len(self.feature_info.get('feature_names', []))
                print(f"   Expected features: {feature_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def standardize_prediction_features(self, input_data: Union[Dict, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        标准化预测输入特征，确保与训练特征完全一致
        
        Args:
            input_data: 输入数据（字典或DataFrame）
            
        Returns:
            处理后的特征数组或None
        """
        
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        try:
            # 转换为DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            # 检查模型类型
            if hasattr(self.model, 'named_steps'):
                # 这是一个Pipeline，让Pipeline自己处理预处理
                print("ℹ️ Using Pipeline model - returning DataFrame for Pipeline preprocessing")
                
                # 获取训练时的特征列
                if self.feature_info and 'feature_names' in self.feature_info:
                    training_columns = self.feature_info['feature_names']
                    
                    # 确保输入包含训练时使用的所有特征
                    missing_features = set(training_columns) - set(input_df.columns)
                    if missing_features:
                        print(f"❌ Missing required features: {list(missing_features)}")
                        return None
                    
                    # 只保留训练时使用的特征，并保持正确顺序
                    input_df = input_df[training_columns]
                    
                    return input_df  # 返回DataFrame让Pipeline处理
                else:
                    print("⚠️ No training column information found")
                    return input_df
            
            else:
                # 这是预处理过的模型，需要手动处理特征
                print("ℹ️ Using preprocessed model - applying manual feature processing")
                
                if not self.feature_info:
                    print("❌ No feature information available for preprocessing")
                    return None
                
                # 获取特征信息
                numeric_features = self.feature_info.get('numeric_features', [])
                categorical_features = self.feature_info.get('categorical_features', [])
                
                # 确保所有特征都存在
                expected_features = numeric_features + categorical_features
                missing_features = set(expected_features) - set(input_df.columns)
                
                if missing_features:
                    print(f"❌ Missing required features: {list(missing_features)}")
                    return None
                
                # 只保留需要的特征
                input_df = input_df[expected_features]
                
                print(f"🔍 Feature processing:")
                print(f"   - Numeric features: {len(numeric_features)}")
                print(f"   - Categorical features: {len(categorical_features)}")
                
                # 这里需要具体的预处理逻辑，取决于训练时使用的预处理器
                # 暂时返回数值部分
                numeric_df = input_df[numeric_features] if numeric_features else pd.DataFrame()
                
                if len(numeric_df.columns) > 0:
                    # 简单的数值处理
                    numeric_array = numeric_df.fillna(0).values
                    return numeric_array
                else:
                    print("❌ No numeric features available")
                    return None
        
        except Exception as e:
            print(f"❌ Feature standardization failed: {e}")
            return None
    
    def predict_single(self, input_data: Union[Dict, pd.DataFrame]) -> Optional[float]:
        """
        单个样本预测
        
        Args:
            input_data: 输入数据
            
        Returns:
            预测结果或None
        """
        
        if self.model is None:
            print("❌ No model loaded for prediction")
            return None
        
        try:
            # 标准化特征
            processed_data = self.standardize_prediction_features(input_data)
            
            if processed_data is None:
                return self._emergency_prediction_fallback(input_data)
            
            # 进行预测
            if isinstance(processed_data, pd.DataFrame):
                # Pipeline模型
                prediction = self.model.predict(processed_data)[0]
            else:
                # 预处理过的模型
                prediction = self.model.predict(processed_data.reshape(1, -1))[0]
            
            return float(prediction)
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return self._emergency_prediction_fallback(input_data)
    
    def predict_batch(self, input_data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        批量预测
        
        Args:
            input_data: 输入数据DataFrame
            
        Returns:
            预测结果数组或None
        """
        
        if self.model is None:
            print("❌ No model loaded for prediction")
            return None
        
        try:
            validate_dataframe(input_data, name="Input data")
            
            # 标准化特征
            processed_data = self.standardize_prediction_features(input_data)
            
            if processed_data is None:
                print("❌ Feature processing failed for batch prediction")
                return None
            
            # 进行预测
            if isinstance(processed_data, pd.DataFrame):
                # Pipeline模型
                predictions = self.model.predict(processed_data)
            else:
                # 预处理过的模型
                predictions = self.model.predict(processed_data)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            return None
    
    def _emergency_prediction_fallback(self, input_data: Union[Dict, pd.DataFrame]) -> Optional[float]:
        """
        紧急备用预测功能，当标准化失败时使用
        """
        
        try:
            print("🚨 Using emergency fallback prediction method")
            
            # 转换为DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            # 只保留数值特征
            numeric_df = input_df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) == 0:
                print("❌ No numeric features found for fallback prediction")
                return None
            
            # 简单标准化
            normalized_data = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8)
            
            # 填充NaN
            normalized_data = normalized_data.fillna(0)
            
            # 尝试预测
            prediction = self.model.predict(normalized_data.values.reshape(1, -1))[0]
            
            print("⚠️ Emergency prediction completed (results may be less accurate)")
            return float(prediction)
            
        except Exception as e:
            print(f"❌ Emergency fallback prediction also failed: {e}")
            return None
    
    def explain_prediction(self, input_data: Union[Dict, pd.DataFrame], 
                          prediction: float) -> Dict[str, Any]:
        """
        解释预测结果
        
        Args:
            input_data: 输入数据
            prediction: 预测结果
            
        Returns:
            解释信息
        """
        
        explanation = {
            'prediction': prediction,
            'input_summary': {},
            'feature_importance': None,
            'confidence_level': 'Unknown'
        }
        
        # 输入数据摘要
        if isinstance(input_data, dict):
            explanation['input_summary'] = input_data
        elif isinstance(input_data, pd.DataFrame):
            explanation['input_summary'] = input_data.iloc[0].to_dict()
        
        # 预测结果解释
        if prediction is not None:
            if prediction >= 0.9:
                explanation['durability_assessment'] = "Excellent"
                explanation['recommendation'] = "Material shows excellent durability characteristics"
            elif prediction >= 0.8:
                explanation['durability_assessment'] = "Good"
                explanation['recommendation'] = "Material has good durability, suitable for most applications"
            elif prediction >= 0.7:
                explanation['durability_assessment'] = "Fair"
                explanation['recommendation'] = "Material durability is acceptable, monitor performance"
            elif prediction >= 0.6:
                explanation['durability_assessment'] = "Poor"
                explanation['recommendation'] = "Material shows reduced durability, consider alternatives"
            else:
                explanation['durability_assessment'] = "Very Poor"
                explanation['recommendation'] = "Material durability is significantly compromised"
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        
        info = {
            'model_loaded': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_info': self.feature_info,
            'model_metadata': self.model_metadata
        }
        
        return info

class FRPPredictionPipeline:
    """完整的FRP预测管道 - 从原始数据到预测结果"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化预测管道
        
        Args:
            model_path: 模型文件路径
        """
        self.preprocessor = FRPDataPreprocessor()
        self.predictor = FRPPredictor(model_path)
    
    def predict_from_raw_data(self, raw_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        从原始数据进行完整预测
        
        Args:
            raw_data: 原始输入数据
            
        Returns:
            预测结果和解释
        """
        
        try:
            # 转换为DataFrame
            if isinstance(raw_data, dict):
                raw_df = pd.DataFrame([raw_data])
            else:
                raw_df = raw_data.copy()
            
            print("🔄 Processing raw data through preprocessing pipeline...")
            
            # 预处理
            processed_df = self.preprocessor.preprocess_data(raw_df)
            
            if processed_df is None or len(processed_df) == 0:
                return {
                    'success': False,
                    'error': 'Data preprocessing failed',
                    'prediction': None
                }
            
            # 预测
            prediction = self.predictor.predict_single(processed_df.iloc[0])
            
            if prediction is None:
                return {
                    'success': False,
                    'error': 'Prediction failed',
                    'prediction': None
                }
            
            # 解释结果
            explanation = self.predictor.explain_prediction(processed_df.iloc[0], prediction)
            
            return {
                'success': True,
                'prediction': prediction,
                'explanation': explanation,
                'processed_features': processed_df.iloc[0].to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }

# 便捷函数
def load_and_predict(model_path: str, input_data: Union[Dict, pd.DataFrame]) -> Optional[float]:
    """便捷的加载模型并预测函数"""
    
    predictor = FRPPredictor(model_path)
    if predictor.model is None:
        return None
    
    return predictor.predict_single(input_data)

def create_sample_input() -> Dict[str, Any]:
    """创建样本输入数据"""
    
    sample_input = {
        'pH_of_condition_enviroment': 7.0,
        'condition_time': 365,  # days
        'fiber_content': 60.0,  # %
        'Temperature': 25.0,  # °C
        'diameter': 12.0,  # mm
        'concrete': 0,  # 0=no concrete, 1=concrete
        'load_value': 0.3,  # relative load
        'Chloride_ion': 0,  # 0=no chloride, 1=chloride present
        'Glass_or_Basalt': 1,  # 1=Glass, 0=Basalt
        'Vinyl_ester_or_Epoxy': 1,  # 1=Vinyl ester, 0=Epoxy
        'surface_treatment': 0,  # 0=sand coated, 1=smooth
        'max_strength': 1200.0  # MPa
    }
    
    return sample_input