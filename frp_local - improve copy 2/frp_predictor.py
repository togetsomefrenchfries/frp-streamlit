#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP钢筋耐久性预测器
直接加载保存的模型进行预测，无需重新训练
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FRPPredictor:
    def __init__(self, models_dir="saved_models"):
        """初始化预测器"""
        self.models_dir = Path(models_dir)
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.models_dir}")
        
        # 加载模型信息
        self.models_info = self._load_models_info()
        
        # 加载模型
        self.models = self._load_models()
        
        # 加载预处理器
        self.preprocessor = self._load_preprocessor()
        
        print("✅ FRP预测器初始化完成")
        print(f"   📁 模型目录: {self.models_dir}")
        print(f"   🤖 可用模型: {list(self.models.keys())}")
        
    def _load_models_info(self):
        """加载模型信息"""
        info_file = self.models_dir / "models_info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"模型信息文件不存在: {info_file}")
        
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_models(self):
        """加载所有训练好的模型"""
        models = {}
        
        for model_name, info in self.models_info.items():
            if model_name == 'preprocessor':
                continue
                
            model_file = Path(info['model_file'])
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"   ✅ 加载模型: {model_name} (R²={info['test_r2']:.4f})")
            else:
                print(f"   ❌ 模型文件不存在: {model_file}")
        
        return models
    
    def _load_preprocessor(self):
        """加载预处理器"""
        preprocessor_file = Path(self.models_info['preprocessor']['preprocessor_file'])
        if not preprocessor_file.exists():
            raise FileNotFoundError(f"预处理器文件不存在: {preprocessor_file}")
        
        with open(preprocessor_file, 'rb') as f:
            preprocessor = pickle.load(f)
        
        print(f"   🔧 预处理器加载完成")
        return preprocessor
    
    def get_feature_template(self):
        """获取特征输入模板"""
        feature_names = self.models_info['preprocessor']['info']['feature_names']
        
        template = {}
        descriptions = {
            'pH_of_condition_enviroment': '环境条件pH值 (通常6-14)',
            'Chloride_ion': '氯离子存在 (0=无, 1=有)',
            'concrete': '混凝土环境 (0=无, 1=有)',
            'diameter': '纤维直径 (mm, 通常6-15)',
            'load_value': '载荷值 (可为0表示无加载)',
            'fiber_content': '纤维含量 (%, 通常0.1-2.0)',
            'initial_tensile_strength': '初始拉伸强度 (MPa, 通常800-1500)',
            'Glass_or_Basalt': '纤维类型 (1=玻璃纤维, 0=玄武岩纤维)',
            'Vinyl_ester_or_Epoxy': '树脂类型 (1=乙烯基酯, 0=环氧树脂)',
            'condition_time': '条件时间 (天/小时)',
            'Temperature': '温度 (°C, 通常20-80)',
            'Tensile_strength_retention': '拉伸强度保持率 (0-1)',
            'surface_treatment': '表面处理 (0=无, 1=有)',
            'glass_transition_temperature': '玻璃化转变温度 (°C, 通常80-150)'
        }
        
        for feature in feature_names:
            clean_name = feature.replace('feat_', '').split('_', 1)[-1]
            template[clean_name] = {
                'value': None,
                'description': descriptions.get(clean_name, '未知特征')
            }
        
        return template
    
    def predict_single(self, features, model_name='RandomForest'):
        """
        单个样本预测
        
        Parameters:
        -----------
        features : dict
            特征字典，键为特征名，值为特征值
        model_name : str
            使用的模型名称 ('RandomForest', 'XGBoost', 'LightGBM')
        
        Returns:
        --------
        dict: 预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不可用，可用模型: {list(self.models.keys())}")
        
        # 检查特征完整性
        expected_features = self.models_info['preprocessor']['info']['feature_names']
        feature_values = []
        
        for feature_name in expected_features:
            clean_name = feature_name.replace('feat_', '').split('_', 1)[-1]
            
            if clean_name in features:
                value = features[clean_name]
                if value is None:
                    # 使用缺失值填充规则
                    value = self._fill_missing_value(clean_name)
                feature_values.append(value)
            else:
                # 使用缺失值填充规则
                value = self._fill_missing_value(clean_name)
                feature_values.append(value)
        
        # 转换为数组
        X = np.array(feature_values).reshape(1, -1)
        
        # 预测
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        
        # 获取模型信息
        model_info = self.models_info[model_name]
        
        result = {
            'prediction': float(prediction),
            'model_used': model_name,
            'model_performance': {
                'test_r2': model_info['test_r2'],
                'config_id': model_info['config_id']
            },
            'input_features': dict(zip([f.replace('feat_', '').split('_', 1)[-1] for f in expected_features], feature_values)),
            'prediction_time': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, features_list, model_name='RandomForest'):
        """
        批量预测
        
        Parameters:
        -----------
        features_list : list of dict
            特征字典列表
        model_name : str
            使用的模型名称
        
        Returns:
        --------
        list: 预测结果列表
        """
        results = []
        for features in features_list:
            try:
                result = self.predict_single(features, model_name)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_features': features
                })
        
        return results
    
    def predict_from_dataframe(self, df, model_name='RandomForest'):
        """从DataFrame预测"""
        results = []
        
        for idx, row in df.iterrows():
            features = row.to_dict()
            try:
                result = self.predict_single(features, model_name)
                result['row_index'] = idx
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'row_index': idx,
                    'input_features': features
                })
        
        return results
    
    def _fill_missing_value(self, feature_name):
        """根据特征填充缺失值"""
        fill_rules = self.models_info['preprocessor']['info']['missing_value_rules']
        
        if feature_name == 'diameter':
            return 9.53  # 中位数
        elif feature_name == 'load_value':
            return 0  # 无加载
        elif feature_name == 'fiber_content':
            return 0.782  # 中位数
        elif feature_name == 'initial_tensile_strength':
            return 971.8  # 中位数
        elif feature_name == 'Temperature':
            return 25  # 室温
        elif feature_name == 'glass_transition_temperature':
            return 114.0  # 中位数
        else:
            return 0  # 默认值
    
    def get_model_info(self):
        """获取模型信息"""
        info = {}
        for model_name, model_info in self.models_info.items():
            if model_name != 'preprocessor':
                info[model_name] = {
                    'test_r2': model_info['test_r2'],
                    'train_r2': model_info['train_r2'],
                    'config_id': model_info['config_id'],
                    'n_features': model_info['n_features'],
                    'training_time': model_info['training_time']
                }
        return info

def demo_prediction():
    """演示预测功能"""
    try:
        # 初始化预测器
        predictor = FRPPredictor()
        
        print("\n🎯 获取特征模板...")
        template = predictor.get_feature_template()
        
        print("\n📋 特征模板:")
        for feature, info in template.items():
            print(f"   {feature}: {info['description']}")
        
        print("\n🧪 示例预测...")
        
        # 示例特征
        sample_features = {
            'pH_of_condition_enviroment': 7.0,
            'Chloride_ion': 1,
            'concrete': 1,
            'diameter': 10.0,
            'load_value': 50.0,
            'fiber_content': 1.0,
            'initial_tensile_strength': 1000.0,
            'Glass_or_Basalt': 1,
            'Vinyl_ester_or_Epoxy': 1,
            'condition_time': 100.0,
            'Temperature': 40.0,
            'Tensile_strength_retention': 0.8,
            'surface_treatment': 1,
            'glass_transition_temperature': 120.0
        }
        
        # 使用不同模型预测
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            if model_name in predictor.models:
                result = predictor.predict_single(sample_features, model_name)
                print(f"\n🤖 {model_name} 预测结果:")
                print(f"   预测值: {result['prediction']:.4f}")
                print(f"   模型R²: {result['model_performance']['test_r2']:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("💡 请先运行 save_best_models.py 来训练并保存模型")

if __name__ == "__main__":
    demo_prediction()