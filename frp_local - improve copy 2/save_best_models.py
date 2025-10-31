#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存最佳模型参数和预处理器
基于最新实验结果，保存可直接用于预测的模型
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import sys
import os

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 直接导入和复用现有的数据处理函数
exec(open('run_40param_experiment.py').read())

class ModelSaver:
    def __init__(self):
        self.models_dir = Path("saved_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # 从最新实验加载最佳参数
        self.best_params = self._load_best_params()
        
    def _load_best_params(self):
        """从最新实验结果加载最佳参数"""
        experiments_dir = Path("experiments")
        
        # 查找最新的详细结果文件
        json_files = list(experiments_dir.glob("*_detailed.json"))
        if not json_files:
            raise FileNotFoundError("没有找到实验结果文件")
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"📂 加载实验结果: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 提取每个模型的最佳配置
        best_params = {}
        for result in results['results']:
            model_name = result['model']
            if model_name not in best_params or result['test_r2'] > best_params[model_name]['test_r2']:
                best_params[model_name] = {
                    'config': eval(result['config']),  # 转换字符串为字典
                    'test_r2': result['test_r2'],
                    'test_rmse': result['test_rmse'],
                    'config_id': result['config_id']
                }
        
        print("✅ 最佳参数加载完成:")
        for model_name, params in best_params.items():
            print(f"   {model_name}: R²={params['test_r2']:.4f}, RMSE={params['test_rmse']:.4f}")
        
        return best_params
    
    def train_and_save_models(self):
        """训练并保存最佳模型"""
        print("🚀 开始训练并保存最佳模型...")
        
        # 使用现有的数据加载逻辑
        loader = ValidDataLoader()
        data = loader.load_valid_data()
        
        # 导入必要的函数和类
        import sys
        import importlib.util
        
        # 动态导入run_40param_experiment模块
        spec = importlib.util.spec_from_file_location("run_40param_experiment", "run_40param_experiment.py")
        exp_module = importlib.util.module_from_spec(spec)
        sys.modules["run_40param_experiment"] = exp_module
        spec.loader.exec_module(exp_module)
        
        # 使用现有的特征提取逻辑  
        print("🔧 开始基于真实结构的特征提取...")
        # 直接使用loader的方法
        result = loader.prepare_features_target(data, generate_plots=False)
        if result is None or len(result) != 4:
            print("❌ 特征提取失败")
            return None
        X, y, feature_names, sample_weights = result
        
        print(f"✅ 数据加载完成: {len(X)} 样本, {len(feature_names)} 特征")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, shuffle=True
        )
        
        models_info = {}
        
        # 训练并保存每个模型
        for model_name, params_info in self.best_params.items():
            print(f"\n🔬 训练模型: {model_name}")
            config = params_info['config']
            
            # 创建模型
            if model_name == 'RandomForest':
                model = RandomForestRegressor(**config)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(**config)
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(**config)
            else:
                continue
            
            # 训练模型
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 评估模型
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            print(f"   ✅ 训练完成: 训练R²={train_score:.4f}, 测试R²={test_score:.4f}")
            print(f"   ⏱️ 训练时间: {training_time:.2f}s")
            
            # 保存模型
            model_file = self.models_dir / f"{model_name.lower()}_best_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存模型信息
            models_info[model_name] = {
                'model_file': str(model_file),
                'config': config,
                'train_r2': train_score,
                'test_r2': test_score,
                'training_time': training_time,
                'config_id': params_info['config_id'],
                'feature_names': feature_names,
                'n_features': len(feature_names),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
        
        # 保存预处理器和特征信息
        preprocessor_data = {
            'feature_names': feature_names,
            'data_filter_rules': {
                'target_parameter': 'Tensile',
                'retention1_required': True,
                'condition_time_required': True,
                'min_features_required': 7
            },
            'missing_value_rules': {
                'diameter': 'median',
                'load_value': 0,
                'fiber_content': 'median',
                'initial_tensile_strength': 'median',
                'Temperature': 25,
                'glass_transition_temperature': 'median'
            }
        }
        
        preprocessor_file = self.models_dir / "preprocessor.pkl"
        with open(preprocessor_file, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        # 添加预处理器信息到models_info
        models_info['preprocessor'] = {
            'preprocessor_file': str(preprocessor_file),
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }
        
        preprocessor_info = {
            'feature_names': feature_names,
            'data_filter_rules': {
                'target_parameter': 'Tensile',
                'retention1_required': True,
                'condition_time_required': True,
                'min_features_required': 7
            },
            'missing_value_rules': {
                'diameter': 'median',
                'load_value': 0,
                'fiber_content': 'median',
                'initial_tensile_strength': 'median',
                'Temperature': 25,
                'glass_transition_temperature': 'median'
            }
        }
        
        # 保存完整的模型信息
        models_info['preprocessor'] = {
            'info': preprocessor_info
        }
        
        info_file = self.models_dir / "models_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(models_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 模型保存完成:")
        print(f"   📁 保存目录: {self.models_dir}")
        print(f"   📋 模型信息: {info_file}")
        
        for model_name in models_info:
            if model_name != 'preprocessor':
                print(f"   🤖 {model_name}: {models_info[model_name]['model_file']}")
        
        return models_info

if __name__ == "__main__":
    saver = ModelSaver()
    models_info = saver.train_and_save_models()
    print("\n🎉 所有模型保存完成！现在可以直接加载进行预测了。")