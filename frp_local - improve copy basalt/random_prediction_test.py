"""
随机抽取数据库数据进行预测测试
从数据库中随机选取1000条数据，筛选有效数据并使用训练好的模型进行预测
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import time
from datetime import datetime

class RandomPredictionTester:
    def __init__(self):
        self.models = {}
        self.models_info = {}
        self.feature_names = []
        
    def load_saved_models(self):
        """加载保存的模型"""
        print("🔄 加载已保存的模型...")
        
        # 加载模型信息
        models_info_path = Path("saved_models/models_info.json")
        if not models_info_path.exists():
            raise FileNotFoundError("❌ 找不到模型信息文件: saved_models/models_info.json")
            
        with open(models_info_path, 'r', encoding='utf-8') as f:
            self.models_info = json.load(f)
            
        # 加载每个模型
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            model_file = self.models_info[model_name]['model_file']
            print(f"  📂 加载 {model_name}: {model_file}")
            
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
                
        # 获取特征名称
        self.feature_names = self.models_info['RandomForest']['feature_names']
        print(f"✅ 模型加载完成，特征数: {len(self.feature_names)}")
        
    def load_random_data(self, n_samples=1000):
        """从数据库随机加载数据"""
        print(f"🎲 从数据库随机选取 {n_samples} 条数据...")
        
        # 查找数据文件
        database_file = Path("E:/大学/intern/2025-summer-concret/database 4.xlsx")
        if not database_file.exists():
            # 尝试其他可能的路径
            alt_paths = [
                Path("../database 4.xlsx"),
                Path("../../database 4.xlsx"),
                Path("database 4.xlsx")
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    database_file = alt_path
                    break
            else:
                raise FileNotFoundError("❌ 找不到数据库文件 database 4.xlsx")
                
        print(f"📁 找到数据文件: {database_file}")
        
        # 读取Excel文件
        try:
            print("🔄 读取Excel文件...")
            data = pd.read_excel(database_file)
            print(f"✅ 原始数据加载成功: {data.shape}")
            
            # 基本过滤（第一列非0）
            valid_mask = data.iloc[:, 0] != 0
            data_filtered = data[valid_mask].copy()
            print(f"📊 第一列非0的数据: {len(data_filtered)} 行")
            
            # 随机抽样
            if len(data_filtered) > n_samples:
                data_sample = data_filtered.sample(n=n_samples, random_state=42)
                print(f"🎯 随机抽取: {len(data_sample)} 条数据")
            else:
                data_sample = data_filtered.copy()
                print(f"⚠️ 可用数据少于{n_samples}条，使用全部{len(data_sample)}条")
                
            return data_sample
            
        except Exception as e:
            print(f"❌ 读取数据文件失败: {e}")
            raise
            
    def extract_features(self, data):
        """提取特征（复用之前的特征提取逻辑）"""
        print("🔧 开始特征提取...")
        
        # 定义特征提取规则（与训练时保持一致）
        feature_extraction_rules = {
            'pH_of_condition_enviroment': [54, 59, 60],
            'Chloride_ion': [61, 64, 77],
            'concrete': [53, 56, 57],
            'diameter': [18],
            'load_value': [90],
            'fiber_content': [15],
            'initial_tensile_strength': [34, 37, 40],
            'Glass_or_Basalt': [8],
            'Vinyl_ester_or_Epoxy': [10],
            'condition_time': [51],
            'Temperature': [49],
            'Tensile_strength_retention': [100],
            'surface_treatment': [22],
            'glass_transition_temperature': [12, 114]
        }
        
        # 提取特征
        features_data = {}
        for feature_name, positions in feature_extraction_rules.items():
            # 取第一个有效位置的数据
            for pos in positions:
                if pos < len(data.columns):
                    col_data = data.iloc[:, pos].copy()
                    features_data[feature_name] = col_data
                    break
            else:
                # 如果所有位置都无效，填充NaN
                features_data[feature_name] = pd.Series([np.nan] * len(data), index=data.index)
                
        # 创建特征DataFrame
        X = pd.DataFrame(features_data)
        print(f"✅ 特征提取完成: {X.shape}")
        
        return X
        
    def filter_valid_data(self, data, X):
        """筛选有效数据（与训练时保持一致的严格检查）"""
        print("🔍 执行严格的数据有效性检查...")
        
        # 获取原始数据的关键列
        target_param_col = 30 if 30 < len(data.columns) else None  # AE列 = 30 (Target_parameter)
        retention1_col = 100 if 100 < len(data.columns) else None  # 第101列(索引100)是真正的retention1数据
        condition_time_col = 51 if 51 < len(data.columns) else None
        
        # 初始化有效性mask
        valid_mask = pd.Series([True] * len(data), index=data.index)
        
        # 检查Target_parameter (AE列)
        if target_param_col is not None:
            target_col_data = data.iloc[:, target_param_col].astype(str).str.lower()
            # 检查是否包含'tensile'
            target_check = target_col_data.str.contains('tensile', na=False)
            valid_mask = valid_mask & target_check
            print(f"  Target_parameter检查: {target_check.sum()}行包含'tensile'")
            # 显示一些示例值用于调试
            unique_values = target_col_data.value_counts().head(10)
            print(f"  Target_parameter示例值: {dict(unique_values)}")
        else:
            print("  ⚠️ 跳过Target_parameter检查（列不存在）")
            
        # 检查retention1有数值
        if retention1_col is not None:
            retention_data = pd.to_numeric(data.iloc[:, retention1_col], errors='coerce')
            retention_check = pd.notna(retention_data)
            valid_mask = valid_mask & retention_check
            print(f"  retention1检查: {retention_check.sum()}行有数值 (列{retention1_col}: {data.columns[retention1_col]})")
            if retention_check.sum() > 0:
                retention_valid = retention_data[retention_check]
                print(f"    retention范围: {retention_valid.min():.3f} - {retention_valid.max():.3f}, 均值: {retention_valid.mean():.3f}")
        else:
            print("  ⚠️ 跳过retention1检查（列不存在）")
            
        # 检查condition_time有值
        if condition_time_col is not None:
            condition_check = pd.notna(data.iloc[:, condition_time_col])
            valid_mask = valid_mask & condition_check
            print(f"  condition_time检查: {condition_check.sum()}行有值")
        else:
            print("  ⚠️ 跳过condition_time检查（列不存在）")
            
        # 应用筛选
        valid_data = data[valid_mask].copy()
        valid_X = X[valid_mask].copy()
        
        print(f"📈 严格数据质量检查结果:")
        print(f"  原始抽样数据: {len(data)} 行")
        print(f"  所有条件都满足: {len(valid_data)} 行")
        print(f"  数据保留率: {len(valid_data)/len(data)*100:.1f}%")
        
        return valid_data, valid_X
        
    def fill_missing_values(self, X):
        """填充缺失值（与训练时保持一致）"""
        print("🔧 按照材料科学规则处理缺失值...")
        
        X_filled = X.copy()
        
        # 直径缺失值用中位数填充
        if 'diameter' in X_filled.columns:
            missing_count = X_filled['diameter'].isna().sum()
            if missing_count > 0:
                median_diameter = X_filled['diameter'].median()
                X_filled['diameter'].fillna(median_diameter, inplace=True)
                print(f"  处理特征 diameter 的 {missing_count} 个缺失值，用中位数填充: {median_diameter:.3f}mm")
                
        # 载荷水平缺失值填充为0
        if 'load_value' in X_filled.columns:
            missing_count = X_filled['load_value'].isna().sum()
            if missing_count > 0:
                X_filled['load_value'].fillna(0, inplace=True)
                print(f"  处理特征 load_value 的 {missing_count} 个缺失值，填充为0 (无加载)")
                
        # 纤维含量缺失值用中位数填充
        if 'fiber_content' in X_filled.columns:
            missing_count = X_filled['fiber_content'].isna().sum()
            if missing_count > 0:
                median_fiber = X_filled['fiber_content'].median()
                X_filled['fiber_content'].fillna(median_fiber, inplace=True)
                print(f"  处理特征 fiber_content 的 {missing_count} 个缺失值，用中位数填充: {median_fiber:.3f}%")
                
        # 初始拉伸强度缺失值用中位数填充
        if 'initial_tensile_strength' in X_filled.columns:
            missing_count = X_filled['initial_tensile_strength'].isna().sum()
            if missing_count > 0:
                median_strength = X_filled['initial_tensile_strength'].median()
                X_filled['initial_tensile_strength'].fillna(median_strength, inplace=True)
                print(f"  处理特征 initial_tensile_strength 的 {missing_count} 个缺失值，用中位数填充: {median_strength:.3f}MPa")
                
        # 温度缺失值填充为25°C（室温）
        if 'Temperature' in X_filled.columns:
            missing_count = X_filled['Temperature'].isna().sum()
            if missing_count > 0:
                X_filled['Temperature'].fillna(25, inplace=True)
                print(f"  处理特征 Temperature 的 {missing_count} 个缺失值，填充为25°C (室温)")
                
        # 玻璃化转变温度缺失值用中位数填充
        if 'glass_transition_temperature' in X_filled.columns:
            missing_count = X_filled['glass_transition_temperature'].isna().sum()
            if missing_count > 0:
                median_glass = X_filled['glass_transition_temperature'].median()
                X_filled['glass_transition_temperature'].fillna(median_glass, inplace=True)
                print(f"  处理特征 glass_transition_temperature 的 {missing_count} 个缺失值，用中位数填充: {median_glass:.3f}°C")
                
        print(f"✅ 缺失值填充完成，最终数据形状: {X_filled.shape}")
        
        # 确保所有数据都是数值型
        print("🔄 转换数据类型为数值型...")
        for col in X_filled.columns:
            X_filled[col] = pd.to_numeric(X_filled[col], errors='coerce')
            
        # 再次填充转换失败产生的NaN
        X_filled = X_filled.fillna(0)
        print("✅ 数据类型转换完成")
        
        return X_filled
        
    def prepare_features_for_prediction(self, X):
        """准备预测用的特征（确保特征顺序和命名与训练时一致）"""
        print("📋 准备预测特征...")
        
        # 重新排列特征顺序以匹配训练时的特征名称
        prediction_features = []
        for i, feature_name in enumerate(self.feature_names):
            # 从特征名称中提取原始特征名（去掉前缀）
            original_name = feature_name.split('_', 2)[-1]  # 例如 'feat_0_pH_of_condition_enviroment' -> 'pH_of_condition_enviroment'
            
            if original_name in X.columns:
                prediction_features.append(X[original_name])
            else:
                # 如果特征不存在，填充0
                prediction_features.append(pd.Series([0] * len(X), index=X.index))
                print(f"  ⚠️ 特征 {original_name} 不存在，用0填充")
                
        # 创建用于预测的DataFrame
        X_pred = pd.DataFrame(prediction_features).T
        X_pred.columns = self.feature_names
        
        print(f"✅ 预测特征准备完成: {X_pred.shape}")
        return X_pred
        
    def predict_with_all_models(self, X_pred):
        """使用所有模型进行预测"""
        print("🔮 开始模型预测...")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            print(f"  🤖 使用 {model_name} 模型预测...")
            
            try:
                pred = model.predict(X_pred)
                predictions[model_name] = pred
                
                # 基本统计
                print(f"    预测范围: [{pred.min():.6f}, {pred.max():.6f}]")
                print(f"    预测均值: {pred.mean():.6f}±{pred.std():.6f}")
                
            except Exception as e:
                print(f"    ❌ {model_name} 预测失败: {e}")
                predictions[model_name] = None
                
        return predictions
        
    def save_results(self, valid_data, X_pred, predictions):
        """保存预测结果"""
        print("💾 保存预测结果...")
        
        # 创建结果DataFrame
        results = pd.DataFrame()
        
        # 添加原始数据索引
        results['original_index'] = valid_data.index
        
        # 添加实际的retention值 (从第101列，索引100)
        if len(valid_data.columns) > 100:
            retention_col_name = valid_data.columns[100]  # 索引100 = 第101列
            retention_values = pd.to_numeric(valid_data.iloc[:, 100], errors='coerce')
            results['actual_retention'] = retention_values.values
            print(f"  添加了实际retention值 (列: {retention_col_name})")
            # 显示一些统计信息
            valid_retention = retention_values.dropna()
            if len(valid_retention) > 0:
                print(f"    retention统计: 范围{valid_retention.min():.3f}-{valid_retention.max():.3f}, 均值{valid_retention.mean():.3f}")
        else:
            print("  ⚠️ 未找到retention列")
        
        # 添加一些关键的原始特征（如果存在）
        key_features = ['diameter', 'fiber_content', 'Temperature', 'condition_time']
        for feature in key_features:
            if feature in X_pred.columns:
                original_name = feature
                for feat_name in X_pred.columns:
                    if feat_name.endswith(original_name):
                        results[f'input_{feature}'] = X_pred[feat_name].values
                        break
                        
        # 添加预测结果
        for model_name, pred in predictions.items():
            if pred is not None:
                results[f'prediction_{model_name}'] = pred
                
        # 计算模型间的一致性
        if len([p for p in predictions.values() if p is not None]) > 1:
            pred_cols = [f'prediction_{name}' for name, pred in predictions.items() if pred is not None]
            if len(pred_cols) > 1:
                results['prediction_std'] = results[pred_cols].std(axis=1)
                results['prediction_mean'] = results[pred_cols].mean(axis=1)
                
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"random_prediction_results_{timestamp}.csv"
        results.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ 结果已保存到: {output_file}")
        print(f"📊 预测结果统计:")
        print(f"  有效样本数: {len(results)}")
        
        for model_name in predictions.keys():
            if predictions[model_name] is not None:
                col_name = f'prediction_{model_name}'
                print(f"  {model_name}: 均值={results[col_name].mean():.6f}, 标准差={results[col_name].std():.6f}")
                
        if 'prediction_std' in results.columns:
            print(f"  模型间预测标准差: 均值={results['prediction_std'].mean():.6f}")
            
        return results
        
    def run_test(self, n_samples=1000):
        """运行完整的随机预测测试"""
        print("🚀 开始随机预测测试")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 加载模型
            self.load_saved_models()
            
            # 2. 随机加载数据
            data = self.load_random_data(n_samples)
            
            # 3. 特征提取
            X = self.extract_features(data)
            
            # 4. 筛选有效数据
            valid_data, valid_X = self.filter_valid_data(data, X)
            
            if len(valid_data) == 0:
                print("❌ 没有找到有效数据，测试结束")
                return None
                
            # 5. 填充缺失值
            X_filled = self.fill_missing_values(valid_X)
            
            # 6. 准备预测特征
            X_pred = self.prepare_features_for_prediction(X_filled)
            
            # 7. 模型预测
            predictions = self.predict_with_all_models(X_pred)
            
            # 8. 保存结果
            results = self.save_results(valid_data, X_pred, predictions)
            
            # 总结
            total_time = time.time() - start_time
            print("\n🎉 随机预测测试完成!")
            print(f"⏱️ 总用时: {total_time:.2f}秒")
            print(f"📈 成功预测: {len(results)} 个有效样本")
            
            return results
            
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("🎲 FRP材料性能随机预测测试")
    print("=" * 60)
    
    tester = RandomPredictionTester()
    results = tester.run_test(n_samples=1000)
    
    if results is not None:
        print("\n✅ 测试成功完成!")
        print("可以查看生成的CSV文件获取详细预测结果")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()