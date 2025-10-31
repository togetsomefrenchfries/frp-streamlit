import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="FRP钢筋耐久性预测系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5f7a;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e3d59;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">🔬 FRP钢筋耐久性预测系统</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">基于机器学习的纤维增强塑料钢筋性能预测平台</p>', unsafe_allow_html=True)

# 侧边栏
st.sidebar.markdown("## 📊 系统功能")
page = st.sidebar.selectbox(
    "选择功能模块",
    ["🎯 耐久性预测", "📈 模型训练", "📋 数据分析", "ℹ️ 系统信息"]
)

# 模拟数据生成函数
@st.cache_data
def generate_sample_data():
    """生成模拟的FRP数据"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'pH_environment': np.random.uniform(6.0, 12.0, n_samples),
        'chloride_ion': np.random.uniform(0, 5.0, n_samples),
        'concrete_strength': np.random.uniform(20, 60, n_samples),
        'diameter': np.random.uniform(6, 25, n_samples),
        'load_value': np.random.uniform(0, 1000, n_samples),
        'fiber_content': np.random.uniform(50, 90, n_samples),
        'tensile_strength': np.random.uniform(800, 1500, n_samples),
        'fiber_type': np.random.choice(['Glass', 'Basalt', 'Carbon'], n_samples),
        'resin_type': np.random.choice(['Vinyl_ester', 'Epoxy'], n_samples),
        'condition_time': np.random.uniform(0, 365, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'glass_transition_temp': np.random.uniform(80, 150, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量（耐久性保持率）
    df['retention_rate'] = (
        0.8 + 0.1 * (df['pH_environment'] - 7) / 5 +
        -0.15 * df['chloride_ion'] / 5 +
        0.1 * (df['concrete_strength'] - 40) / 20 +
        0.05 * (df['fiber_content'] - 70) / 20 +
        -0.1 * df['condition_time'] / 365 +
        -0.05 * (df['temperature'] - 25) / 10 +
        np.random.normal(0, 0.05, n_samples)
    )
    df['retention_rate'] = np.clip(df['retention_rate'], 0.3, 1.0)
    
    return df

# 预处理函数
def preprocess_data(df):
    """数据预处理"""
    df_processed = df.copy()
    
    # 编码分类变量
    df_processed['fiber_type_encoded'] = df_processed['fiber_type'].map({
        'Glass': 0, 'Basalt': 1, 'Carbon': 2
    })
    df_processed['resin_type_encoded'] = df_processed['resin_type'].map({
        'Vinyl_ester': 1, 'Epoxy': 0
    })
    
    # 选择数值特征
    features = [
        'pH_environment', 'chloride_ion', 'concrete_strength', 'diameter',
        'load_value', 'fiber_content', 'tensile_strength', 'condition_time',
        'temperature', 'glass_transition_temp', 'fiber_type_encoded', 'resin_type_encoded'
    ]
    
    return df_processed[features], df_processed['retention_rate']

# 训练模型函数
@st.cache_data
def train_model():
    """训练预测模型"""
    df = generate_sample_data()
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, scaler, r2, mse, X.columns.tolist()

# 页面内容
if page == "🎯 耐久性预测":
    st.markdown('<h2 class="sub-header">🎯 FRP钢筋耐久性预测</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 输入参数")
        
        # 环境参数
        with st.expander("🌍 环境参数", expanded=True):
            pH = st.slider("pH值", 6.0, 12.0, 7.5, 0.1)
            chloride = st.slider("氯离子浓度 (%)", 0.0, 5.0, 1.0, 0.1)
            temperature = st.slider("温度 (°C)", 15.0, 35.0, 25.0, 1.0)
            condition_time = st.slider("暴露时间 (天)", 0, 365, 180, 1)
        
        # 材料参数
        with st.expander("🧪 材料参数", expanded=True):
            concrete_strength = st.slider("混凝土强度 (MPa)", 20.0, 60.0, 40.0, 1.0)
            diameter = st.slider("钢筋直径 (mm)", 6.0, 25.0, 12.0, 1.0)
            fiber_content = st.slider("纤维含量 (%)", 50.0, 90.0, 70.0, 1.0)
            tensile_strength = st.slider("拉伸强度 (MPa)", 800.0, 1500.0, 1000.0, 10.0)
            glass_transition = st.slider("玻璃化转变温度 (°C)", 80.0, 150.0, 120.0, 1.0)
        
        # 类型选择
        with st.expander("🔧 材料类型", expanded=True):
            fiber_type = st.selectbox("纤维类型", ["Glass", "Basalt", "Carbon"])
            resin_type = st.selectbox("树脂类型", ["Vinyl_ester", "Epoxy"])
            load_value = st.slider("荷载值 (N)", 0.0, 1000.0, 500.0, 10.0)
    
    with col2:
        st.markdown("### 🎯 预测结果")
        
        if st.button("🚀 开始预测", type="primary"):
            # 准备输入数据
            input_data = pd.DataFrame({
                'pH_environment': [pH],
                'chloride_ion': [chloride],
                'concrete_strength': [concrete_strength],
                'diameter': [diameter],
                'load_value': [load_value],
                'fiber_content': [fiber_content],
                'tensile_strength': [tensile_strength],
                'condition_time': [condition_time],
                'temperature': [temperature],
                'glass_transition_temp': [glass_transition],
                'fiber_type_encoded': [{'Glass': 0, 'Basalt': 1, 'Carbon': 2}[fiber_type]],
                'resin_type_encoded': [{'Vinyl_ester': 1, 'Epoxy': 0}[resin_type]]
            })
            
            # 加载模型并预测
            model, scaler, r2, mse, feature_names = train_model()
            
            with st.spinner("正在进行预测..."):
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                
                # 显示预测结果
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 预测结果</h2>
                    <h1>{prediction:.1%}</h1>
                    <p>耐久性保持率</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 评估结果
                col_r2, col_mse = st.columns(2)
                with col_r2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>模型R²得分</h4>
                        <h3>{r2:.3f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_mse:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>均方误差</h4>
                        <h3>{mse:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 影响因素分析
                st.markdown("### 📊 关键影响因素")
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': importance
                }).sort_values('重要性', ascending=False)
                
                fig = px.bar(
                    importance_df.head(8), 
                    x='重要性', 
                    y='特征',
                    orientation='h',
                    title="特征重要性排序"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 模型训练":
    st.markdown('<h2 class="sub-header">📈 模型训练与评估</h2>', unsafe_allow_html=True)
    
    if st.button("🔄 重新训练模型"):
        with st.spinner("正在训练模型..."):
            model, scaler, r2, mse, feature_names = train_model()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² 得分", f"{r2:.3f}")
            with col2:
                st.metric("均方误差", f"{mse:.4f}")
            with col3:
                st.metric("特征数量", len(feature_names))
            
            st.success("✅ 模型训练完成！")
    
    # 显示训练数据统计
    df = generate_sample_data()
    st.markdown("### 📊 训练数据概览")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "📋 数据分析":
    st.markdown('<h2 class="sub-header">📋 数据分析与可视化</h2>', unsafe_allow_html=True)
    
    df = generate_sample_data()
    
    # 数据分布
    st.markdown("### 📈 数据分布")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='retention_rate', title='耐久性保持率分布')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='condition_time', y='retention_rate', 
                        color='fiber_type', title='时间 vs 保持率')
        st.plotly_chart(fig, use_container_width=True)
    
    # 相关性分析
    st.markdown("### 🔗 特征相关性")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="特征相关性热图")
    st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ 系统信息":
    st.markdown('<h2 class="sub-header">ℹ️ 系统信息</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 系统功能
        - **耐久性预测**: 基于输入参数预测FRP钢筋的耐久性
        - **模型训练**: 使用随机森林算法训练预测模型
        - **数据分析**: 提供数据可视化和统计分析
        - **实时预测**: 支持参数调整和实时结果更新
        
        ### 🔬 技术特点
        - 机器学习算法: Random Forest
        - 特征工程: 标准化预处理
        - 交互界面: Streamlit框架
        - 数据可视化: Plotly图表
        """)
    
    with col2:
        st.markdown("""
        ### 📊 模型参数
        - **训练样本**: 1000个模拟样本
        - **特征数量**: 12个关键特征
        - **算法**: 随机森林回归
        - **评估指标**: R²得分和均方误差
        
        ### 🎯 应用场景
        - 材料性能评估
        - 工程设计优化
        - 科研数据分析
        - 质量控制预测
        """)
    
    # 系统状态
    st.markdown("### 💻 系统状态")
    st.success("✅ 系统运行正常")
    st.info(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p>🔬 FRP钢筋耐久性预测系统 | 基于机器学习的材料性能预测平台</p>
        <p style='font-size: 0.9em;'>Powered by Streamlit • Built for Materials Science Research</p>
    </div>
    """, 
    unsafe_allow_html=True
)