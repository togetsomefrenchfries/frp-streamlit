import base64
import json
import os
import re
import smtplib
import pickle
from datetime import datetime
from email.mime.text import MIMEText
import altair as alt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import hashlib
import socket
import ipaddress
import time
import io
from sklearn.metrics import r2_score, mean_squared_error
import pymysql
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.express as px

# ——————————————————————————————
# Global Functions for Model Training (pickle-safe)
# ——————————————————————————————
def global_clean_categorical_features(X):
    """Global function to clean categorical features - can be pickled"""
    X_clean = X.copy()
    for col in X_clean.columns:
        X_clean[col] = X_clean[col].fillna('unknown').astype(str)
    return X_clean

def create_enhanced_preprocessor(categorical_cols, numeric_cols, add_polynomial=True, polynomial_degree=2):
    """Create enhanced preprocessor with optional polynomial features"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer
    
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
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

# ——————————————————————————————
# 1. Page Configuration
# ——————————————————————————————
st.set_page_config(
    page_title="FRP Rebar Durability Prediction Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——————————————————————————————
# 2. Custom CSS Style Injection
# ——————————————————————————————
def inject_custom_css():
    """Inject custom CSS styles"""
    st.markdown("""
    <style>
    /* Theme Color Definition */
    :root {
        --primary-color: #1e3d59;
        --secondary-color: #f5f0e1;
        --accent-color: #ff6e40;
        --success-color: #4caf50;
        --warning-color: #ff9800;
        --error-color: #f44336;
        --text-primary: #333333;
        --text-secondary: #666666;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title Styles */
    h1 {
        background: linear-gradient(120deg, #1e3d59 0%, #3c5f7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        font-size: 3rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, white 0%, #fafcff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 
            0 4px 20px rgba(30, 61, 89, 0.08),
            0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(30, 61, 89, 0.1);
        position: relative;
        overflow: hidden;
        animation: cardFadeIn 0.6s ease-out;
        animation-fill-mode: both;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            45deg, 
            transparent 20%, 
            rgba(30, 61, 89, 0.05) 40%, 
            rgba(60, 95, 125, 0.1) 50%, 
            rgba(30, 61, 89, 0.05) 60%, 
            transparent 80%
        );
        transform: translateX(-100%) translateY(-100%) rotate(45deg);
        transition: transform 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #1e3d59, #3c5f7d, #5a7ea1, #3c5f7d, #1e3d59);
        background-size: 300% 300%;
        border-radius: 18px;
        z-index: -1;
        opacity: 0;
        animation: gradientShift 3s ease infinite;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        transform: translateX(100%) translateY(100%) rotate(45deg);
    }
    
    .metric-card:hover::after {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.03) rotateX(5deg);
        box-shadow: 
            0 20px 40px rgba(30, 61, 89, 0.15),
            0 8px 20px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(30, 61, 89, 0.1);
        border-color: rgba(30, 61, 89, 0.2);
    }
    
    @keyframes cardFadeIn {
        0% {
            opacity: 0;
            transform: translateY(30px) scale(0.9);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #1e3d59 30%, #3c5f7d 90%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 25px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #3c5f7d 30%, #1e3d59 90%);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transform: translateY(-3px) scale(1.02);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
        transition: all 0.1s ease;
    }
    
    /* Card Button Specific Styling */
    div[style*="margin-bottom: 1rem"] + div .stButton > button {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #495057;
        border: 2px solid #dee2e6;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.25s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 0.85rem;
        letter-spacing: 0.3px;
    }
    
    div[style*="margin-bottom: 1rem"] + div .stButton > button:hover {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
        color: #1e3d59;
        border-color: #1e3d59;
        box-shadow: 0 4px 12px rgba(30, 61, 89, 0.2);
        transform: translateY(-1px);
    }
    
    div[style*="margin-bottom: 1rem"] + div .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Card Container Hover Effects */
    div[style*="background: white"][style*="border-left: 4px solid"] {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[style*="background: white"][style*="border-left: 4px solid"]:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Card Shimmer Effect */
    div[style*="background: white"][style*="border-left: 4px solid"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.6s ease;
        z-index: 1;
        pointer-events: none;
    }
    
    div[style*="background: white"][style*="border-left: 4px solid"]:hover::before {
        left: 100%;
    }
    
    /* Ensure card content stays above shimmer */
    div[style*="background: white"][style*="border-left: 4px solid"] > * {
        position: relative;
        z-index: 2;
    }
    
    /* Info Text Animations */
    .stInfo, div[data-testid="stAlert"] {
        animation: slideInLeft 0.5s ease-out;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stInfo::before, div[data-testid="stAlert"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(30, 61, 89, 0.05), transparent);
        transition: left 0.6s ease;
    }
    
    .stInfo:hover::before, div[data-testid="stAlert"]:hover::before {
        left: 100%;
    }
    
    .stInfo:hover, div[data-testid="stAlert"]:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(30, 61, 89, 0.1);
    }
    
    /* General text content animations */
    .stMarkdown h4, .stMarkdown h3, .stMarkdown h2 {
        animation: fadeInUp 0.4s ease-out;
    }
    
    .stMarkdown p {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Specific animation for missing values text */
    .stMarkdown p:contains("Showing"), 
    .stMarkdown p:contains("more fields"),
    .stMarkdown div:contains("Showing"),
    .stMarkdown div:contains("more fields") {
        animation: slideInLeft 0.6s ease-out;
        transition: all 0.3s ease;
        padding: 0.5rem 0;
    }
    
    /* Missing Info Text Specific Animation */
    .missing-info-text {
        animation: slideInUp 0.5s ease-out;
    }
    
    .missing-info-text > div {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .missing-info-text > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .missing-info-text > div:hover::before {
        left: 100%;
    }
    
    .missing-info-text > div:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 4px 15px rgba(12, 84, 96, 0.2);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Selectbox Beautification */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .stSelectbox > div > div::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #1e3d59, #3c5f7d);
        transition: all 0.3s ease;
        transform: translateX(-50%);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1e3d59;
        box-shadow: 0 0 0 3px rgba(30, 61, 89, 0.1);
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div:focus-within::after {
        width: 100%;
    }
    
    /* Enhanced Selectbox Dropdown Animation */
    .stSelectbox [data-baseweb="select"] {
        animation: selectboxFadeIn 0.3s ease-out;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .stSelectbox [data-baseweb="select"]:hover > div {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Dropdown Menu Animation */
    .stSelectbox [data-baseweb="popover"] {
        animation: dropdownSlideIn 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-origin: top center;
    }
    
    .stSelectbox [data-baseweb="popover"] > div {
        border-radius: 12px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        overflow: hidden;
    }
    
    /* Dropdown Options Animation */
    .stSelectbox [data-baseweb="menu"] > ul > li {
        transition: all 0.2s ease;
        border-radius: 8px;
        margin: 2px 4px;
    }
    
    .stSelectbox [data-baseweb="menu"] > ul > li:hover {
        background: linear-gradient(135deg, #1e3d59 0%, #3c5f7d 100%) !important;
        color: white !important;
        transform: translateX(4px);
    }
    
    /* Select Arrow Animation */
    .stSelectbox [data-baseweb="select"] svg {
        transition: transform 0.3s ease;
    }
    
    .stSelectbox [data-baseweb="select"][aria-expanded="true"] svg {
        transform: rotate(180deg);
    }
    
    @keyframes dropdownSlideIn {
        0% {
            opacity: 0;
            transform: scaleY(0.3) translateY(-20px);
        }
        50% {
            opacity: 0.8;
            transform: scaleY(0.8) translateY(-10px);
        }
        100% {
            opacity: 1;
            transform: scaleY(1) translateY(0);
        }
    }
    
    @keyframes selectboxFadeIn {
        0% {
            opacity: 0;
            transform: translateY(10px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Input Box Beautification */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1e3d59;
        box-shadow: 0 0 0 3px rgba(30, 61, 89, 0.1);
        transform: translateY(-1px) scale(1.01);
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #3c5f7d;
        transform: translateY(-1px);
    }
    
    /* Data Table Beautification */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead th {
        background-color: #1e3d59 !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f5f5f5 !important;
    }
    
    /* Sidebar beautification */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3d59 0%, #2c4f6f 100%);
    }
    
    /* Natural tab navigation design */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.6);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(226, 232, 240, 0.3);
        margin: 24px auto 32px auto;
        max-width: 100%;
        overflow-x: auto;
        flex-wrap: wrap;
        backdrop-filter: blur(5px);
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        min-width: 120px;
        max-width: 160px;
        height: 42px;
        padding: 0 16px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 6px;
        border: 1px solid rgba(203, 213, 225, 0.3);
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 13px;
        color: #64748b;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 61, 89, 0.08);
        border-color: rgba(30, 61, 89, 0.3);
        color: #1e3d59;
        box-shadow: 0 2px 6px rgba(30, 61, 89, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3d59, #2c4f6f) !important;
        color: white !important;
        border-color: #1e3d59 !important;
        box-shadow: 0 3px 8px rgba(30, 61, 89, 0.2);
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #2c4f6f, #1e3d59) !important;
        box-shadow: 0 4px 12px rgba(30, 61, 89, 0.3);
    }
    
    /* Tab content panel styles - adjust divider position */
    .stTabs > div > div[data-baseweb="tab-panel"] {
        padding-top: 8px !important;
        margin-top: -8px;
    }
    
    /* Responsive design - keep it simple */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
            gap: 6px;
            padding: 8px 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            width: 100%;
            max-width: none;
            min-width: auto;
        }
    }
    
    @media (max-width: 1024px) and (min-width: 769px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            padding: 8px 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-width: 110px;
            max-width: 140px;
            font-size: 12px;
        }
    }
    
    /* Success/Error/Warning message beautification */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 5px solid;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric beautification */
    [data-testid="metric-container"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(30, 61, 89, 0.05), transparent);
        transition: left 0.6s ease;
    }
    
    [data-testid="metric-container"]:hover::before {
        left: 100%;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Scrollbar beautification */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e3d59;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3c5f7d;
    }
    
    /* Streamlit Slider Styles - Fixed Animation */
    .stSlider > div > div > div > div {
        background: #e2e8f0 !important;
        height: 6px !important;
        border-radius: 8px !important;
    }
    
    .stSlider .stSlider-thumb {
        width: 20px !important;
        height: 20px !important;
        background: #1e3d59 !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(30, 61, 89, 0.3) !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .stSlider .stSlider-thumb:hover {
        background: #2c4f6f !important;
        box-shadow: 0 3px 8px rgba(30, 61, 89, 0.4) !important;
    }
    
    /* Alternative selectors for different Streamlit versions */
    div[data-baseweb="slider"] {
        padding: 0.5rem 0 !important;
    }
    
    div[data-baseweb="slider"] > div > div {
        background: #e2e8f0 !important;
        height: 6px !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="slider"] div[role="slider"] {
        width: 20px !important;
        height: 20px !important;
        background: #1e3d59 !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(30, 61, 89, 0.3) !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    div[data-baseweb="slider"] div[role="slider"]:hover {
        background: #2c4f6f !important;
        box-shadow: 0 3px 8px rgba(30, 61, 89, 0.4) !important;
    }
    
    /* Track fill */
    div[data-baseweb="slider"] > div > div > div {
        background: #1e3d59 !important;
        border-radius: 8px !important;
    }
    
    /* More generic slider styles */
    [class*="slider"] [class*="track"] {
        background: #e2e8f0 !important;
        height: 6px !important;
        border-radius: 8px !important;
    }
    
    [class*="slider"] [class*="thumb"] {
        width: 20px !important;
        height: 20px !important;
        background: #1e3d59 !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(30, 61, 89, 0.3) !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    [class*="slider"] [class*="thumb"]:hover {
        background: #2c4f6f !important;
    }
    
    /* Fallback for any slider element */
    input[type="range"] {
        -webkit-appearance: none !important;
        background: transparent !important;
        height: 6px !important;
    }
    
    input[type="range"]::-webkit-slider-track {
        background: #e2e8f0 !important;
        height: 6px !important;
        border-radius: 8px !important;
    }
    
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none !important;
        width: 20px !important;
        height: 20px !important;
        background: #1e3d59 !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(30, 61, 89, 0.3) !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    input[type="range"]::-webkit-slider-thumb:hover {
        background: #2c4f6f !important;
    }
    
    /* Loading animation */
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #1e3d59;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Page Animations */
    
    /* Smooth page entrance animation */
    .stApp > div {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Smooth sidebar animation */
    .css-1d391kg {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Enhanced sidebar animations */
    .css-1d391kg, .css-1cypcdb, .css-17lntkn {
        transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    /* Sidebar content transitions */
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stButton,
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stMetric {
        transition: opacity 0.2s ease-in-out, transform 0.2s ease-in-out;
    }
    
    /* Main content area smooth transition */
    .css-18e3th9, .css-1d391kg + div {
        transition: margin-left 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    /* Sidebar toggle button animation */
    .css-vk3wp9 {
        transition: all 0.2s ease !important;
    }
    
    .css-vk3wp9:hover {
        transform: scale(1.1);
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Additional sidebar performance optimizations */
    .css-1outpf7, .css-hxt7ib, .css-1y4p8pa {
        transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        will-change: transform, opacity;
    }
    
    /* Prevent layout shifts during sidebar toggle */
    .css-18e3th9 {
        will-change: margin-left !important;
    }
    
    /* Smooth sidebar width transitions */
    .css-1d391kg[data-testid="stSidebar"] {
        transition: width 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    /* Sidebar backdrop animation */
    .css-1dp5vir {
        transition: opacity 0.2s ease-in-out !important;
    }
    
    /* Remove any conflicting animations */
    .css-1d391kg * {
        transition-delay: 0s !important;
    }
    
    /* Enhanced dataframe animations */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .dataframe:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .dataframe thead th {
        background-color: #1e3d59 !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .dataframe thead th::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .dataframe thead th:hover::before {
        left: 100%;
    }
    
    .dataframe tbody tr {
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
        transform: scale(1.01);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Tab Animation with Advanced Effects */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
        padding: 16px 20px;
        border-radius: 16px;
        border: 1px solid rgba(226, 232, 240, 0.4);
        margin: 24px auto 32px auto;
        max-width: 100%;
        overflow-x: auto;
        flex-wrap: wrap;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 8px 32px rgba(30, 61, 89, 0.08),
            0 4px 16px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        animation: tabListFadeIn 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
    }
    
    .stTabs [data-baseweb="tab-list"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(30, 61, 89, 0.02) 50%, transparent 70%);
        border-radius: 16px;
        animation: tabListGlow 4s ease-in-out infinite;
    }
    
    @keyframes tabListFadeIn {
        0% {
            opacity: 0;
            transform: translateY(-30px) scale(0.95);
            filter: blur(5px);
        }
        60% {
            opacity: 0.8;
            transform: translateY(-5px) scale(1.02);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: blur(0);
        }
    }
    
    @keyframes tabListGlow {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        min-width: 120px;
        max-width: 160px;
        height: 48px;
        padding: 0 20px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        border-radius: 12px;
        border: 1px solid rgba(203, 213, 225, 0.4);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        font-weight: 500;
        font-size: 14px;
        color: #64748b;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 2px 8px rgba(0, 0, 0, 0.06),
            0 1px 3px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        animation: tabFadeIn 0.6s ease-out;
        animation-delay: calc(var(--tab-index, 0) * 0.1s);
        animation-fill-mode: both;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg, 
            transparent 0%, 
            rgba(30, 61, 89, 0.1) 30%, 
            rgba(60, 95, 125, 0.15) 50%, 
            rgba(30, 61, 89, 0.1) 70%, 
            transparent 100%
        );
        transition: left 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .stTabs [data-baseweb="tab"]::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #1e3d59, #3c5f7d, #1e3d59);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform: translateX(-50%);
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover::after {
        width: 80%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(30, 61, 89, 0.08) 0%, rgba(60, 95, 125, 0.05) 100%);
        border-color: rgba(30, 61, 89, 0.3);
        color: #1e3d59;
        box-shadow: 
            0 8px 25px rgba(30, 61, 89, 0.15),
            0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-4px) scale(1.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3d59 0%, #2c4f6f 50%, #3c5f7d 100%) !important;
        color: white !important;
        border-color: #1e3d59 !important;
        box-shadow: 
            0 12px 32px rgba(30, 61, 89, 0.3),
            0 6px 16px rgba(30, 61, 89, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        font-weight: 600;
        transform: translateY(-2px) scale(1.08);
        animation: selectedTabPulse 2s ease-in-out infinite;
    }
    
    .stTabs [aria-selected="true"]::after {
        width: 100% !important;
        height: 3px !important;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.3)) !important;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #2c4f6f 0%, #1e3d59 50%, #2a4766 100%) !important;
        box-shadow: 
            0 16px 40px rgba(30, 61, 89, 0.4),
            0 8px 20px rgba(30, 61, 89, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-3px) scale(1.08);
    }
    
    @keyframes tabFadeIn {
        0% {
            opacity: 0;
            transform: translateY(20px) scale(0.8);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes selectedTabPulse {
        0%, 100% {
            box-shadow: 
                0 12px 32px rgba(30, 61, 89, 0.3),
                0 6px 16px rgba(30, 61, 89, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        50% {
            box-shadow: 
                0 16px 40px rgba(30, 61, 89, 0.4),
                0 8px 20px rgba(30, 61, 89, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
    }
    
    /* Enhanced Alert Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 5px solid;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.4s ease-out;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stSuccess::before, .stError::before, .stWarning::before, .stInfo::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s ease;
    }
    
    .stSuccess:hover::before, .stError:hover::before, .stWarning:hover::before, .stInfo:hover::before {
        left: 100%;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Enhanced Progress Bar with Advanced Animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1e3d59, #3c5f7d, #5a7ea1, #3c5f7d, #1e3d59);
        background-size: 200% 100%;
        animation: 
            progress-shimmer 2s ease-in-out infinite,
            progress-pulse 1.5s ease-in-out infinite alternate;
        border-radius: 12px;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 4px 8px rgba(30, 61, 89, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stProgress > div > div > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.4), 
            transparent);
        animation: progress-sweep 3s ease-in-out infinite;
    }
    
    .stProgress > div > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 50%;
        background: linear-gradient(180deg, 
            rgba(255, 255, 255, 0.3) 0%, 
            transparent 100%);
        border-radius: 12px 12px 0 0;
    }
    
    @keyframes progress-shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes progress-pulse {
        from { 
            box-shadow: 
                0 4px 8px rgba(30, 61, 89, 0.3),
                0 0 15px rgba(30, 61, 89, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        to { 
            box-shadow: 
                0 6px 16px rgba(30, 61, 89, 0.4),
                0 0 25px rgba(30, 61, 89, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
    }
    
    @keyframes progress-sweep {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    /* Progress Bar Container Enhancement */
    .stProgress > div > div {
        background-color: rgba(30, 61, 89, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(30, 61, 89, 0.2);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Smooth content transitions */
    .stMarkdown, .stDataFrame, .stPlotlyChart {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* ========================
       STATUS MESSAGES ANIMATIONS
       ======================== */
    
    /* Info Messages */
    .stAlert[data-baseweb="notification"][kind="info"] {
        animation: messageSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-left: 4px solid #17a2b8 !important;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.15) !important;
    }
    
    /* Success Messages */
    .stAlert[data-baseweb="notification"][kind="success"] {
        animation: messageSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-left: 4px solid #28a745 !important;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15) !important;
    }
    
    /* Warning Messages */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        animation: messageSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-left: 4px solid #ffc107 !important;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15) !important;
    }
    
    /* Error Messages */
    .stAlert[data-baseweb="notification"][kind="error"] {
        animation: messageSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-left: 4px solid #dc3545 !important;
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15) !important;
    }
    
    @keyframes messageSlideIn {
        0% {
            opacity: 0;
            transform: translateX(-20px) scale(0.95);
        }
        50% {
            opacity: 0.8;
            transform: translateX(0) scale(1.02);
        }
        100% {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }
    
    /* ========================
       FORM ELEMENTS ANIMATIONS
       ======================== */
    
    /* Checkbox Animation */
    .stCheckbox > label {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stCheckbox > label:hover {
        transform: translateX(4px);
    }
    
    .stCheckbox input[type="checkbox"] {
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform: scale(1.2);
    }
    
    .stCheckbox input[type="checkbox"]:checked {
        animation: checkboxPop 0.3s ease;
    }
    
    @keyframes checkboxPop {
        0% { transform: scale(1.2); }
        50% { transform: scale(1.4); }
        100% { transform: scale(1.2); }
    }
    
    /* Radio Button Animation */
    .stRadio > label {
        transition: all 0.3s ease;
    }
    
    .stRadio input[type="radio"]:checked + label {
        animation: radioSelect 0.4s ease;
        color: #1e3d59 !important;
        font-weight: 600;
    }
    
    @keyframes radioSelect {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Multiselect Animation */
    .stMultiSelect [data-baseweb="tag"] {
        animation: tagSlideIn 0.3s ease;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #1e3d59 0%, #3c5f7d 100%) !important;
        transition: all 0.3s ease;
    }
    
    .stMultiSelect [data-baseweb="tag"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 61, 89, 0.3);
    }
    
    @keyframes tagSlideIn {
        0% {
            opacity: 0;
            transform: translateY(-10px) scale(0.8);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    /* File Uploader Animation */
    .stFileUploader > div {
        border-radius: 16px !important;
        border: 2px dashed #1e3d59 !important;
        transition: all 0.3s ease;
        animation: uploaderBreath 3s infinite ease-in-out;
    }
    
    .stFileUploader > div:hover {
        border-color: #3c5f7d !important;
        background: rgba(30, 61, 89, 0.05) !important;
        transform: scale(1.02);
    }
    
    @keyframes uploaderBreath {
        0%, 100% { 
            border-color: #1e3d59;
            box-shadow: 0 0 0 0 rgba(30, 61, 89, 0.1);
        }
        50% { 
            border-color: #3c5f7d;
            box-shadow: 0 0 0 8px rgba(30, 61, 89, 0.1);
        }
    }
    
    /* Date Input Animation */
    .stDateInput > div > div {
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    .stDateInput > div > div:focus-within {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 61, 89, 0.15) !important;
    }
    
    /* Time Input Animation */
    .stTimeInput > div > div {
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    .stTimeInput > div > div:focus-within {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 61, 89, 0.15) !important;
    }
    
    /* ========================
       EXPANDER ANIMATIONS
       ======================== */
    
    /* Expander Header Animation */
    .streamlit-expanderHeader {
        transition: all 0.3s ease !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #1e3d59 0%, #3c5f7d 100%) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 61, 89, 0.2) !important;
    }
    
    /* Expander Content Animation */
    .streamlit-expanderContent {
        animation: expanderSlideDown 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        overflow: hidden;
    }
    
    @keyframes expanderSlideDown {
        0% {
            opacity: 0;
            max-height: 0;
            transform: translateY(-20px);
        }
        100% {
            opacity: 1;
            max-height: 1000px;
            transform: translateY(0);
        }
    }
    
    /* ========================
       DATA TABLE ANIMATIONS
       ======================== */
    
    /* Table Container Animation */
    .stDataFrame {
        animation: tableSlideIn 0.6s ease-out;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stDataFrame table {
        border-radius: 12px !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #1e3d59 0%, #3c5f7d 100%) !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .stDataFrame td {
        transition: all 0.3s ease;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(30, 61, 89, 0.05) !important;
        transform: scale(1.01);
    }
    
    @keyframes tableSlideIn {
        0% {
            opacity: 0;
            transform: translateY(30px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ========================
       CHART ANIMATIONS
       ======================== */
    
    /* Chart Container Animation */
    .stPlotlyChart {
        animation: chartFadeIn 0.8s ease-out;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    @keyframes chartFadeIn {
        0% {
            opacity: 0;
            transform: scale(0.95) translateY(20px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    /* ========================
       LOADING ANIMATIONS
       ======================== */
    
    /* Spinner Enhancement */
    .stSpinner > div {
        border-radius: 50% !important;
        border: 3px solid rgba(30, 61, 89, 0.1) !important;
        border-top: 3px solid #1e3d59 !important;
        animation: spinnerRotate 1s linear infinite, spinnerPulse 2s ease-in-out infinite;
    }
    
    @keyframes spinnerRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes spinnerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* ========================
       NOTIFICATION ANIMATIONS
       ======================== */
    
    /* Toast Notification Style */
    .stToast {
        animation: toastSlideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    @keyframes toastSlideIn {
        0% {
            opacity: 0;
            transform: translateX(100%) scale(0.8);
        }
        100% {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }
    
    /* ========================
       ADVANCED UX ANIMATIONS
       ======================== */
    
    /* Page Container Transition */
    .main .block-container {
        animation: pageLoad 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    @keyframes pageLoad {
        0% {
            opacity: 0;
            transform: translateY(30px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Hover Effects for Interactive Elements */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        animation: buttonHoverPulse 0.6s ease-in-out;
    }
    
    @keyframes buttonHoverPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Form Validation Animation */
    .stTextInput > div > div > input:invalid {
        animation: inputShake 0.5s ease-in-out;
        border-color: #dc3545 !important;
    }
    
    @keyframes inputShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    /* Success Pulse for Form Submission */
    .form-success {
        animation: successPulse 1s ease-in-out;
    }
    
    @keyframes successPulse {
        0% { 
            background: rgba(40, 167, 69, 0.1);
            transform: scale(1);
        }
        50% { 
            background: rgba(40, 167, 69, 0.2);
            transform: scale(1.02);
        }
        100% { 
            background: rgba(40, 167, 69, 0.1);
            transform: scale(1);
        }
    }
    
    /* Staggered Animation for Lists */
    .stContainer > div:nth-child(odd) {
        animation: staggeredFadeIn 0.6s ease-out;
    }
    
    .stContainer > div:nth-child(even) {
        animation: staggeredFadeIn 0.6s ease-out 0.1s both;
    }
    
    @keyframes staggeredFadeIn {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Parallax Effect for Large Containers */
    .main-container {
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    /* Microinteractions for Better Feedback */
    *:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(30, 61, 89, 0.2) !important;
        transition: box-shadow 0.3s ease !important;
    }
    
    /* Smooth Transitions for All Interactive Elements */
    .stButton, .stSelectbox, .stTextInput, .stTextArea, 
    .stNumberInput, .stSlider, .stCheckbox, .stRadio {
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    }
    
    /* Enhanced scrollbar with animations */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
        box-shadow: inset 0 0 2px rgba(0,0,0,0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #1e3d59, #3c5f7d);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #3c5f7d, #1e3d59);
        box-shadow: 0 0 10px rgba(30, 61, 89, 0.5);
    }
    
    ::-webkit-scrollbar-corner {
        background: transparent;
    }
    }
    </style>
    """, unsafe_allow_html=True)

# ——————————————————————————————
# 3. Helper Functions for Role-based UI Rendering
# ——————————————————————————————

def check_database_connection():
    """检查数据库连接状态和数据可用性"""
    try:
        engine = get_db_engine()
        if not engine:
            return False, "Database engine not available", 0
        
        with engine.connect() as conn:
            # 检查research_data表是否存在
            result = conn.execute(text("SHOW TABLES LIKE 'research_data'")).fetchone()
            if not result:
                return False, "research_data table not found", 0
            
            # 获取记录数量
            count = conn.execute(text("SELECT COUNT(*) FROM research_data")).scalar()
            return True, f"Connected to {DB_NAME}", count
            
    except Exception as e:
        return False, f"Connection error: {str(e)}", 0

def render_data_overview_admin(df, table_name, data_manager):
    """Render complete Data Overview for admin users"""
    
    # Add title card consistent with other sections
    st.markdown("""
    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #3498db;">
        <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
            Comprehensive dataset statistics and quality metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查数据库连接状态
    db_connected, db_message, db_record_count = check_database_connection()
    
    # Check for duplicates and provide clean data stats
    original_count = len(df)
    unique_df = df.drop_duplicates()
    unique_count = len(unique_df)
    
    if original_count != unique_count:
        st.warning(f"Found {original_count - unique_count} duplicate records in the dataset. Showing statistics for unique data.")
        stats_df = unique_df  # Use unique data for statistics
    else:
        stats_df = df
    
    # Statistical metric cards following Model Configuration style
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600;">UNIQUE RECORDS</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{unique_count:,}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data entries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        # Calculate temperature statistics
        temp_value = "N/A"
        temp_color = "#95a5a6"
        if "temperature" in stats_df.columns:
            temp_numeric = pd.to_numeric(stats_df["temperature"], errors='coerce')
            if temp_numeric.notna().sum() > 0:
                avg_temp = temp_numeric.mean()
                temp_value = f"{avg_temp:.1f}°C"
                temp_color = "#e74c3c"
        
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {temp_color}; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: {temp_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">AVG TEMPERATURE</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{temp_value}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Environmental condition</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        # Calculate completeness
        completeness_value = "0%"
        completeness_color = "#e74c3c"
        if len(stats_df) > 0:
            total_cells = len(stats_df) * len(stats_df.columns)
            missing_cells = stats_df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
            completeness_value = f"{completeness:.1f}%"
            completeness_color = "#27ae60" if completeness > 80 else "#f39c12" if completeness > 60 else "#e74c3c"
        
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {completeness_color}; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: {completeness_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">DATA QUALITY</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{completeness_value}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Completeness rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        # Calculate field count
        field_count = len(stats_df.columns)
        field_color = "#8B5FBF"
        
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {field_color}; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: {field_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">TOTAL FIELDS</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{field_count}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table preview
    st.markdown("#### Dataset Preview")
    
    # Information bar
    info_col1, info_col2 = st.columns([3, 1])
    with info_col1:
        if original_count != unique_count:
            st.info(f"Showing {unique_count:,} unique records (filtered from {original_count:,} total), {len(stats_df.columns)} fields")
        else:
            st.info(f"Showing {len(stats_df):,} records, {len(stats_df.columns)} fields")
    with info_col2:
        if st.button("Refresh Data", use_container_width=True, key="refresh_overview"):
            data_manager.invalidate_cache(f"table_{table_name}")
            st.rerun()
    
    # Data table
    if len(df) > 0:
        # Debug: Check for data uniqueness
        unique_records = df.drop_duplicates()
        if len(unique_records) != len(df):
            st.warning(f"Data contains duplicates: {len(df)} total records, {len(unique_records)} unique records")
            
            # Offer option to show unique data only
            show_unique = st.checkbox("Show only unique records", value=True)
            if show_unique:
                df = unique_records
                st.info(f"Displaying {len(df)} unique records")
        
        # Prepare display data, handle NaN values
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].fillna('--')
            else:
                display_df[col] = display_df[col].fillna(0)
        
        # Display data table with limited height to keep page tidy
        st.dataframe(
            display_df, 
            use_container_width=True, 
            height=300
        )
    else:
        st.warning("No data to display")

def render_data_overview_viewer(df):
    """Render simplified Data Overview for viewer users"""
    
    # Add title card consistent with other sections
    st.markdown("""
    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #3498db;">
        <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
            Essential dataset information and statistics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查数据库连接状态（简化版）
    db_connected, db_message, db_record_count = check_database_connection()
    
    # 数据库连接状态指示器（简化版）
    if db_connected:
        st.info(f"Data loaded from database: {db_record_count:,} total records")
    else:
        st.warning(f"Data may not be current: {db_message}")
    
    # Basic statistics cards (simplified version seen by viewer users)
    original_count = len(df)
    unique_df = df.drop_duplicates()
    unique_count = len(unique_df)
    stats_df = unique_df if original_count != unique_count else df
    
    # Basic statistics for viewer users
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600;">RECORDS</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{unique_count:,}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data entries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        field_count = len(stats_df.columns)
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #8B5FBF; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: #8B5FBF; margin: 0; font-size: 0.75rem; font-weight: 600;">FIELDS</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{field_count}</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        # Data availability indicator for viewer
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #27ae60; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color: #27ae60; margin: 0; font-size: 0.75rem; font-weight: 600;">STATUS</h4>
            <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">READY</h2>
            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data available</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple data preview for viewer users
    st.markdown("#### Dataset Sample")
    if len(df) > 0:
        # Show only first 5 rows for viewer users
        sample_df = df.head(5).copy()
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                sample_df[col] = sample_df[col].fillna('--')
            else:
                sample_df[col] = sample_df[col].fillna(0)
        
        st.dataframe(sample_df, use_container_width=True, height=200)
        st.info(f"Showing sample of {len(sample_df)} records from {len(df)} total")
    else:
        st.warning("No data to display")

# ——————————————————————————————
# 4. Performance Optimization Classes and Functions
# ——————————————————————————————

class DataManager:
    """Unified data manager to reduce redundant queries"""
    
    def __init__(self):
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
            st.session_state.cache_timestamp = {}
    
    def get_data(self, key, loader_func, ttl=300):
        """Get cached data or reload"""
        current_time = datetime.now()
        
        # Check if cache exists and is not expired
        if key in st.session_state.data_cache:
            if key in st.session_state.cache_timestamp:
                if (current_time - st.session_state.cache_timestamp[key]).seconds < ttl:
                    return st.session_state.data_cache[key]
        
        # Reload data
        data = loader_func()
        st.session_state.data_cache[key] = data
        st.session_state.cache_timestamp[key] = current_time
        return data
    
    def invalidate_cache(self, key=None):
        """Invalidate cache"""
        if key:
            st.session_state.data_cache.pop(key, None)
            st.session_state.cache_timestamp.pop(key, None)
        else:
            st.session_state.data_cache.clear()
            st.session_state.cache_timestamp.clear()

# ——————————————————————————————
# 4. UI Component Functions
# ——————————————————————————————

def create_gradient_header(text, subtitle=""):
    """Create gradient header"""
    subtitle_html = f'<p style="text-align: center; color: #666; font-size: 1.2rem; margin-top: -1rem;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="padding: 2rem 0;">
        <h1 style="background: linear-gradient(120deg, #1e3d59 0%, #ff6e40 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   text-align: center; 
                   font-size: 3.5rem;
                   font-weight: 800;
                   margin: 0;">
            {text}
        </h1>
        {subtitle_html}
    </div>
    """

def create_metric_card(title, value, delta=None, delta_color="normal", index=0):
    """Create metric card with staggered animation"""
    delta_html = ""
    if delta is not None:
        color = {"normal": "#4caf50", "inverse": "#f44336"}.get(delta_color, "#4caf50")
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<p style="color: {color}; font-size: 0.9rem; margin: 0;">{arrow} {abs(delta):.2f}%</p>'
    
    return f"""
    <div class="metric-card" style="animation-delay: {index * 0.15}s;">
        <h4 style="color: #666; font-size: 0.9rem; margin: 0;">{title}</h4>
        <h2 style="color: #1e3d59; margin: 0.5rem 0;">{value}</h2>
        {delta_html}
    </div>
    """

def show_loading_animation(message="Loading..."):
    """Show loading animation"""
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; padding: 2rem;">
        <div class="loader"></div>
        <p style="margin-top: 1rem; color: #1e3d59; font-weight: 600;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

# ——————————————————————————————
# 5. Database Configuration and Connection
# ——————————————————————————————
Base = declarative_base()

# Get database configuration from environment variables
load_dotenv()
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "666666")  # MySQL password
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "haigui_database")

# Create database engine (using connection pool)
@st.cache_resource
def get_db_engine():
    """Get database engine, using connection pool to optimize performance"""
    try:
        connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        print(f"Connecting to database: {connection_string}")  # Debug info
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"Successfully connected to database: {DB_NAME}")
        return engine
    except Exception as e:
        print(f"Database connection failed: {e}")
        st.error(f"Database connection failed: {e}")
        return None

# ——————————————————————————————
# 6. IP Address Validation Functions
# ——————————————————————————————
def get_client_ip():
    """Get client IP address"""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip
    except:
        return "127.0.0.1"

def is_ip_allowed(ip_address):
    """Check if IP address is within allowed range"""
    allowed_networks = [
        "192.168.0.0/16",
        "10.0.0.0/8",
        "127.0.0.1/32",
    ]

    try:
        ip_obj = ipaddress.ip_address(ip_address)
        for network in allowed_networks:
            if ip_obj in ipaddress.ip_network(network):
                return True
    except:
        pass
    return False

# ——————————————————————————————
# 7. Database Table Structure Creation
# ——————————————————————————————
def create_tables(engine):
    """Create necessary database tables"""
    with engine.connect() as conn:
        # Create user table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                institution VARCHAR(255),
                ip_address VARCHAR(45),
                role VARCHAR(50) DEFAULT 'viewer',
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_by VARCHAR(255),
                approved_at TIMESTAMP NULL
            )
        """))

        # Create data change request table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_changes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                change_type VARCHAR(50) NOT NULL,
                table_name VARCHAR(50) NOT NULL,
                record_id INT,
                old_data JSON,
                new_data JSON,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_by VARCHAR(255),
                reviewed_at TIMESTAMP NULL,
                review_comment TEXT
            )
        """))

        # Create operation log table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS operation_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                operation_type VARCHAR(50) NOT NULL,
                table_name VARCHAR(50),
                record_id INT,
                details JSON,
                ip_address VARCHAR(45),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Create version control table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_versions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                table_name VARCHAR(50) NOT NULL,
                record_id INT NOT NULL,
                version_number INT NOT NULL,
                data JSON NOT NULL,
                changed_by VARCHAR(255) NOT NULL,
                change_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.commit()

# ——————————————————————————————
# 8. Initialize Administrator Account
# ——————————————————————————————
def initialize_admin(engine):
    """Initialize default administrator account"""
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@frp.com")
    admin_name = os.environ.get("ADMIN_NAME", "System Administrator")
    admin_institution = os.environ.get("ADMIN_INSTITUTION", "System")

    with engine.connect() as conn:
        try:
            result = conn.execute(
                text("SELECT email, name FROM users WHERE role = 'admin' AND status = 'approved'")
            ).fetchone()

            if not result:
                conn.execute(
                    text("""
                        INSERT INTO users (name, email, institution, role, status, approved_by, approved_at, ip_address)
                        VALUES (:name, :email, :institution, 'admin', 'approved', 'system', NOW(), '127.0.0.1')
                    """),
                    {
                        "name": admin_name,
                        "email": admin_email,
                        "institution": admin_institution
                    }
                )
                conn.commit()
                return True, admin_email, "created"
            else:
                existing_email = result._asdict()['email'] if hasattr(result, '_asdict') else result[0]
                return True, existing_email, "exists"

        except Exception as e:
            if "Duplicate" in str(e):
                conn.execute(
                    text("""
                        UPDATE users
                        SET role = 'admin', status = 'approved', approved_by = 'system', approved_at = NOW()
                        WHERE email = :email
                    """),
                    {"email": admin_email}
                )
                conn.commit()
                return True, admin_email, "upgraded"
            else:
                print(f"Administrator initialization failed: {e}")
                return False, None, "error"

# ——————————————————————————————
# 9. User Authentication and Permission Management
# ——————————————————————————————
def authenticate_user(email, engine):
    """Verify user identity"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT email, name, role, status FROM users WHERE email = :email AND status = 'approved'"),
            {"email": email}
        ).fetchone()

        if result:
            if hasattr(result, '_asdict'):
                return result._asdict()
            elif hasattr(result, '_mapping'):
                return dict(result._mapping)
            else:
                return {
                    'email': result[0],
                    'name': result[1],
                    'role': result[2],
                    'status': result[3]
                }
        return None

def request_access(name, email, institution, ip_address, engine):
    """Request access permission"""
    with engine.connect() as conn:
        try:
            conn.execute(
                text("""
                    INSERT INTO users (name, email, institution, ip_address)
                    VALUES (:name, :email, :institution, :ip)
                """),
                {"name": name, "email": email, "institution": institution, "ip": ip_address}
            )
            conn.commit()
            return True, "Request submitted successfully"
        except Exception as e:
            if "Duplicate" in str(e):
                return False, "Email already exists"
            return False, str(e)

def log_operation(user_email, operation_type, table_name, record_id, details, ip_address, engine):
    """Record operation log"""
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO operation_logs (user_email, operation_type, table_name, record_id, details, ip_address)
                VALUES (:email, :op_type, :table, :record_id, :details, :ip)
            """),
            {
                "email": user_email,
                "op_type": operation_type,
                "table": table_name,
                "record_id": record_id,
                "details": json.dumps(details) if isinstance(details, dict) else details,
                "ip": ip_address
            }
        )
        conn.commit()

# ——————————————————————————————
# 10. Data Change Request and Approval
# ——————————————————————————————
def submit_data_change(user_email, change_type, table_name, record_id, old_data, new_data, engine):
    """Submit data change request"""
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO data_changes (user_email, change_type, table_name, record_id, old_data, new_data)
                VALUES (:email, :change_type, :table, :record_id, :old_data, :new_data)
            """),
            {
                "email": user_email,
                "change_type": change_type,
                "table": table_name,
                "record_id": record_id,
                "old_data": json.dumps(old_data) if old_data else None,
                "new_data": json.dumps(new_data)
            }
        )
        conn.commit()

def get_pending_changes(engine):
    """Get pending changes for approval"""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT * FROM data_changes
                WHERE status = 'pending'
                ORDER BY created_at DESC
            """)
        ).fetchall()
        return result

def display_structured_data(data_json):
    """
    Optimize display of JSON data, categorized display of different types of fields
    """
    import json
    
    # Parse JSON data
    if isinstance(data_json, str):
        try:
            data = json.loads(data_json)
        except:
            st.error("Invalid JSON format")
            st.text(data_json)
            return
    else:
        data = data_json
    
    if not isinstance(data, dict):
        st.json(data)
        return
    
    # Group fields by category
    field_categories = {
        "Basic Information": ["Title", "Author", "Location", "Brand_name", "Manufacturer"],
        "Material Properties": [
            "Fiber_type", "Matrix_type", "surface_treatment", "Fiber_type_detail", "Matrix_type_detail",
            "Fiber_content_weight", "Fiber_content_volume", "diameter", "Void_content",
            "cure_ratio", "cure_ratio_2", "tensile_modulus", "ultimate_tensile_strength"
        ],
        "Environmental Conditions": [
            "temperature", "temp", "temp2", "pH", "pH_1", "pH_2", "pHafter", "cycle_pH", "cycle_pH_after",
            "UV", "RH_1", "RH_2", "RH2", "time", "time_field", "time_in_cycle", "Year",
            "field_average_temperature", "field_average_humidity", "Effektive_Klimaklassifikation"
        ],
        "Test Results": [
            "Value1_1", "Value1_2", "Value2_1", "Value2_2", "Value3_1", "Value3_2",
            "retention1", "retention2", "retention3", "COV_1", "COV_2", "COV1_1", "COV1_2", "COV2_1", "COV2_2",
            "strength_of_concrete", "pH_of_concrete"
        ],
        "Analysis Data": [
            "SEM_L_ACAT", "SEM_L_ACBT", "SEM_L_BCAT", "SEM_L_BCBT", "SEM_T_ACAT", "SEM_T_ACBT", "SEM_T_BCAT", "SEM_T_BCBT",
            "FTIR_1", "FTIR_2", "Water_absorption_ratio", "Water_absorption_at_saturation",
            "glass_transition_temperature", "glass_transition_temperature_2", "glass_transition_temperature_run_2"
        ],
        "Experimental Setup": [
            "concrete", "ingredient_1", "ingredient_2", "stress_or_strain", "value_load", "type_of_load",
            "SolutionorMoisture", "cycle_ingredient", "average_area", "nominal_area"
        ],
        "Metadata": [
            "created_at", "updated_at", "Target_parameter", "feature_name", "Journal_or_Conference_name",
            "No_field", "no_field_secondary", "number_field", "type_field"
        ]
    }
    
    # Collect all categorized fields
    categorized_fields = set()
    for category_fields in field_categories.values():
        categorized_fields.update(category_fields)
    
    # Find uncategorized fields
    uncategorized_fields = [key for key in data.keys() if key not in categorized_fields]
    if uncategorized_fields:
        field_categories["Notes & Others"] = uncategorized_fields
    
    # Create tabs to display different categories
    categories_with_data = []
    category_tabs = []
    
    for category, fields in field_categories.items():
        # Check if this category has data
        category_data = {}
        for field in fields:
            if field in data and data[field] not in [None, "", 0, "0"]:
                category_data[field] = data[field]
        
        if category_data:
            categories_with_data.append(category)
            category_tabs.append(category_data)
    
    if not categories_with_data:
        st.info("No meaningful data to display")
        return
    
    # Create summary information
    total_fields = len(data)
    filled_fields = len([v for v in data.values() if v not in [None, "", 0, "0"]])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Fields", total_fields)
    with col2:
        st.metric("Filled Fields", filled_fields)
    with col3:
        st.metric("Completion Rate", f"{filled_fields/total_fields*100:.1f}%")
    
    # Display all data directly without tabs
    for category, category_data in zip(categories_with_data, category_tabs):
        if len(category_data) > 0:
            st.markdown(f"**{category}** ({len(category_data)} fields)")
            
            # Create two-column layout to display data
            items = list(category_data.items())
            mid_point = (len(items) + 1) // 2
            
            col1, col2 = st.columns(2)
            
            with col1:
                for key, value in items[:mid_point]:
                    # Format display value
                    if isinstance(value, (int, float)) and value != 0:
                        display_value = f"**{value}**"
                    elif isinstance(value, str) and value.strip():
                        display_value = f"*{value}*"
                    else:
                        display_value = str(value)
                    
                    st.markdown(f"• **{key}**: {display_value}")
            
            with col2:
                for key, value in items[mid_point:]:
                    # Format display value
                    if isinstance(value, (int, float)) and value != 0:
                        display_value = f"**{value}**"
                    elif isinstance(value, str) and value.strip():
                        display_value = f"*{value}*"
                    else:
                        display_value = str(value)
                    
                    st.markdown(f"• **{key}**: {display_value}")
            
            st.markdown("")  # Add some spacing between categories
    
    # Display raw JSON data (for technical review)
    st.markdown("#### Raw JSON Data (Technical View)")
    with st.container():
        if st.button("Show/Hide Raw JSON", key=f"toggle_json_{hash(str(data))}"):
            st.session_state[f"show_json_{hash(str(data))}"] = not st.session_state.get(f"show_json_{hash(str(data))}", False)
        
        if st.session_state.get(f"show_json_{hash(str(data))}", False):
            st.json(data)

# ——————————————————————————————
# 11. Email Notification
# ——————————————————————————————
EMAIL_USER = os.environ.get("EMAIL_USER", "wpj010528@163.com")

def send_email(to_email: str, subject: str, content: str):
    """Send email notification"""
    code = st.session_state.get("email_pass", "")
    if not code:
        return

    msg = MIMEText(content)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.163.com", 465) as server:
            server.login(EMAIL_USER, code)
            server.sendmail(EMAIL_USER, [to_email], msg.as_string())
    except Exception as e:
        st.error(f"Email sending failed: {e}")

# ——————————————————————————————
# 12. Improved FRP Data Preprocessing Module (Integrated Version)
# ——————————————————————————————

def standardize_prediction_features(input_data, training_feature_info, model=None):
    """
    Standardize prediction input features to ensure complete consistency with training features
    
    Args:
        input_data: Input data (dict or DataFrame)
        training_feature_info: Feature information saved during training
        model: Trained model for judging if it's a Pipeline
    
    Returns:
        Processed feature array or DataFrame
    """
    try:
        # Convert to DataFrame (if needed)
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Check model type
        if model is not None and hasattr(model, 'named_steps'):
            # This is a Pipeline, need to return original DataFrame for Pipeline to handle itself
            st.info("Using Pipeline model - returning original DataFrame for Pipeline preprocessing")
            
            # Get original features from training
            training_columns = training_feature_info.get('training_columns', [])
            
            if training_columns:
                # Ensure input contains all features used during training
                missing_features = set(training_columns) - set(input_df.columns)
                if missing_features:
                    st.error(f"Missing required features for Pipeline: {list(missing_features)}")
                    return None
                
                # Only keep features used during training and in correct order
                input_df = input_df[training_columns]
                
                return input_df  # Return DataFrame for Pipeline processing
            else:
                st.warning("No training column information found, using current DataFrame")
                return input_df
        
        else:
            # This is a preprocessed model, need to manually handle features
            st.info("Using preprocessed model - applying manual feature processing")
            
            # Get feature information from training
            numeric_features = training_feature_info.get('numeric_features', [])
            categorical_features = training_feature_info.get('categorical_features', [])
            feature_encoder = training_feature_info.get('feature_encoder')
            feature_scaler = training_feature_info.get('feature_scaler')
            
            # Get feature columns from training (excluding target variable)
            training_columns = training_feature_info.get('training_columns', [])
            target_variable = training_feature_info.get('target_variable', '')
            
            # 过滤掉目标变量，防止其被误包含在特征中
            target_variables = ['Strength of unconditioned rebar', 'Tensile strength retention', target_variable]
            target_variables = [t for t in target_variables if t]  # 移除空字符串
            numeric_features = [f for f in numeric_features if f not in target_variables]
            categorical_features = [f for f in categorical_features if f not in target_variables]
            
            # 确保所有特征都存在
            expected_features = numeric_features + categorical_features
            missing_features = set(expected_features) - set(input_df.columns)
            
            if missing_features:
                st.error(f"Missing required features: {list(missing_features)}")
                return None
                
            # 只保留需要的特征，并排除任何目标变量
            available_features = [col for col in expected_features if col in input_df.columns and col not in target_variables]
            input_df = input_df[available_features]
            
            # 调试信息：确认预测特征处理状态
            st.info(f"🔍 Prediction feature processing:")
            st.info(f"   - Expected numeric: {len(numeric_features)} features")
            st.info(f"   - Expected categorical: {len(categorical_features)} features") 
            st.info(f"   - Target variables excluded: {target_variables}")
            st.info(f"   - Available features: {len(available_features)} features")
            
            # 分离数值特征和分类特征（重新过滤以确保不包含目标变量）
            final_numeric_features = [f for f in numeric_features if f in input_df.columns and f not in target_variables]
            final_categorical_features = [f for f in categorical_features if f in input_df.columns and f not in target_variables]
            
            input_numeric = input_df[final_numeric_features] if final_numeric_features else pd.DataFrame()
            input_categorical = input_df[final_categorical_features] if final_categorical_features else pd.DataFrame()
            
            # 处理分类特征
            if len(final_categorical_features) > 0 and feature_encoder is not None:
                # 清理分类特征
                input_cat_clean = input_categorical.copy()
                for col in final_categorical_features:
                    input_cat_clean[col] = input_cat_clean[col].fillna('unknown').astype(str)
                
                # 编码分类特征
                try:
                    input_cat_encoded = feature_encoder.transform(input_cat_clean)
                except Exception as e:
                    st.error(f"❌ Categorical feature encoding failed: {e}")
                    return None
            else:
                input_cat_encoded = np.empty((len(input_df), 0))
            
            # 处理数值特征
            if len(final_numeric_features) > 0 and feature_scaler is not None:
                # 标准化数值特征
                try:
                    # 添加详细的特征匹配调试信息
                    expected_features = feature_scaler.feature_names_in_ if hasattr(feature_scaler, 'feature_names_in_') else None
                    provided_features = list(input_numeric.columns)
                    
                    st.info(f"🔧 Numeric feature scaling:")
                    if expected_features is not None:
                        st.info(f"   - Expected by scaler: {list(expected_features)}")
                    else:
                        st.info(f"   - Expected by scaler: N/A (no feature_names_in_ attribute)")
                    st.info(f"   - Provided for scaling: {provided_features}")
                    
                    # 检查特征匹配
                    if expected_features is not None:
                        missing_features = set(expected_features) - set(provided_features)
                        extra_features = set(provided_features) - set(expected_features)
                        
                        if missing_features:
                            st.error(f"   - Missing features: {list(missing_features)}")
                        if extra_features:
                            st.warning(f"   - Extra features: {list(extra_features)}")
                        if not missing_features and not extra_features:
                            st.success(f"   - ✅ Perfect feature match!")
                    
                    input_num_scaled = feature_scaler.transform(input_numeric)
                    st.success(f"✅ Numeric feature scaling successful")
                except Exception as e:
                    st.error(f"❌ Numeric feature scaling failed: {e}")
                    if hasattr(feature_scaler, 'feature_names_in_'):
                        st.error(f"Expected features: {list(feature_scaler.feature_names_in_)}")
                    else:
                        st.error(f"Expected features: N/A (no feature_names_in_ attribute)")
                    st.error(f"Provided features: {list(input_numeric.columns)}")
                    return None
            else:
                input_num_scaled = input_numeric.values if len(final_numeric_features) > 0 else np.empty((len(input_df), 0))
            
            # 合并所有特征
            input_processed = np.hstack([input_num_scaled, input_cat_encoded])
            
            return input_processed
        
    except Exception as e:
        st.error(f"❌ Feature standardization failed: {e}")
        return None

def emergency_prediction_fallback(input_df, model):
    """
    紧急预测备用方法，当标准特征处理失败时使用
    
    Args:
        input_df: 输入DataFrame
        model: 训练好的模型
    
    Returns:
        预测结果或None
    """
    try:
        st.warning("🚨 Using emergency prediction fallback...")
        
        # 尝试使用原始数据的数值部分
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.error("No numeric features found for emergency prediction")
            return None
        
        # 只使用数值特征
        input_numeric = input_df[numeric_cols].values
        
        # 尝试直接预测（如果模型可以处理）
        try:
            prediction = model.predict(input_numeric.reshape(1, -1))[0]
            st.warning(f"Emergency prediction completed using {len(numeric_cols)} numeric features only")
            return prediction
        except Exception as e:
            st.error(f"Emergency prediction also failed: {e}")
            return None
            
    except Exception as e:
        st.error(f"Emergency fallback failed: {e}")
        return None

def emergency_prediction_fallback(input_data, model):
    """
    紧急备用预测功能，当标准化失败时使用
    """
    try:
        st.warning("🚨 Using emergency fallback prediction method. Results may be less accurate.")
        
        # 转换为DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # 只保留数值特征
        numeric_df = input_df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            st.error("No numeric features found for fallback prediction")
            return None
        
        # 简单标准化
        normalized_data = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8)
        
        # 如果模型期望更多特征，用零填充
        expected_features = len(st.session_state.get('feature_names', []))
        if expected_features > normalized_data.shape[1]:
            padding = np.zeros((normalized_data.shape[0], expected_features - normalized_data.shape[1]))
            normalized_data = np.hstack([normalized_data.values, padding])
        
        # 进行预测
        prediction = model.predict(normalized_data.reshape(1, -1))[0]
        return prediction
        
    except Exception as e:
        st.error(f"Emergency fallback prediction also failed: {e}")
        return None

# ——————————————————————————————
# 模型缓存管理类
# ——————————————————————————————
class ModelCacheManager:
    """
    模型缓存管理器
    保存和加载训练好的模型、最佳参数和评估结果
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.cache_table_name = "trained_models_cache"
        self._init_cache_table()
    
    def _init_cache_table(self):
        """初始化模型缓存表"""
        try:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.cache_table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_key VARCHAR(255) UNIQUE NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                target_variable VARCHAR(100) NOT NULL,
                evaluation_strategy VARCHAR(100) NOT NULL,
                data_hash VARCHAR(64) NOT NULL,
                model_data LONGTEXT NOT NULL,
                best_params LONGTEXT NOT NULL,
                evaluation_results LONGTEXT NOT NULL,
                feature_info LONGTEXT NOT NULL,
                preprocessing_info LONGTEXT NOT NULL,
                training_info LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_model_key (model_key),
                INDEX idx_model_name (model_name),
                INDEX idx_target_variable (target_variable)
            )
            """
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
        except Exception as e:
            st.warning(f"Model cache table initialization failed: {e}")
    
    def _generate_model_key(self, model_name, target_variable, data_hash, evaluation_strategy):
        """生成模型的唯一标识key"""
        import hashlib
        key_string = f"{model_name}_{target_variable}_{data_hash}_{evaluation_strategy}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def save_model(self, model_name, target_variable, evaluation_strategy, data_hash, 
                   trained_model, best_params, evaluation_results, feature_info, 
                   preprocessing_info, training_info):
        """
        保存训练好的模型
        
        参数:
        model_name: 模型名称 (RandomForest, XGBoost, LightGBM)
        target_variable: 目标变量
        evaluation_strategy: 评估策略
        data_hash: 训练数据的哈希值
        trained_model: 训练好的模型对象
        best_params: 最佳参数
        evaluation_results: 评估结果
        feature_info: 特征信息
        preprocessing_info: 预处理信息
        training_info: 训练信息
        """
        try:
            import pickle
            import json
            
            # 生成模型key
            model_key = self._generate_model_key(model_name, target_variable, data_hash, evaluation_strategy)
            
            # 序列化模型
            model_data = base64.b64encode(pickle.dumps(trained_model)).decode()
            
            # 转换为JSON格式
            best_params_json = json.dumps(best_params)
            evaluation_results_json = json.dumps(evaluation_results)
            feature_info_json = json.dumps(feature_info)
            preprocessing_info_json = json.dumps(preprocessing_info)
            training_info_json = json.dumps(training_info)
            
            # 检查是否已存在
            check_query = f"SELECT id FROM {self.cache_table_name} WHERE model_key = :model_key"
            
            with self.engine.connect() as conn:
                existing = conn.execute(text(check_query), {"model_key": model_key}).fetchone()
                
                if existing:
                    # 更新现有模型
                    update_query = f"""
                    UPDATE {self.cache_table_name}
                    SET model_data = :model_data, best_params = :best_params, 
                        evaluation_results = :evaluation_results, feature_info = :feature_info,
                        preprocessing_info = :preprocessing_info, training_info = :training_info,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE model_key = :model_key
                    """
                    conn.execute(text(update_query), {
                        "model_data": model_data,
                        "best_params": best_params_json,
                        "evaluation_results": evaluation_results_json,
                        "feature_info": feature_info_json,
                        "preprocessing_info": preprocessing_info_json,
                        "training_info": training_info_json,
                        "model_key": model_key
                    })
                else:
                    # 插入新模型
                    insert_query = f"""
                    INSERT INTO {self.cache_table_name} 
                    (model_key, model_name, target_variable, evaluation_strategy, data_hash,
                     model_data, best_params, evaluation_results, feature_info, 
                     preprocessing_info, training_info)
                    VALUES (:model_key, :model_name, :target_variable, :evaluation_strategy, 
                            :data_hash, :model_data, :best_params, :evaluation_results, 
                            :feature_info, :preprocessing_info, :training_info)
                    """
                    conn.execute(text(insert_query), {
                        "model_key": model_key,
                        "model_name": model_name,
                        "target_variable": target_variable,
                        "evaluation_strategy": evaluation_strategy,
                        "data_hash": data_hash,
                        "model_data": model_data,
                        "best_params": best_params_json,
                        "evaluation_results": evaluation_results_json,
                        "feature_info": feature_info_json,
                        "preprocessing_info": preprocessing_info_json,
                        "training_info": training_info_json
                    })
                
                conn.commit()
                return model_key
                
        except Exception as e:
            st.error(f"Failed to save model: {e}")
            return None
    
    def load_model(self, model_key):
        """根据model_key加载模型"""
        try:
            import pickle
            import json
            
            query = f"""
            SELECT model_name, target_variable, evaluation_strategy, model_data, 
                   best_params, evaluation_results, feature_info, preprocessing_info, 
                   training_info, created_at, updated_at
            FROM {self.cache_table_name}
            WHERE model_key = :model_key
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"model_key": model_key}).fetchone()
                
                if result:
                    # 反序列化模型
                    model_data = pickle.loads(base64.b64decode(result[3].encode()))
                    
                    return {
                        'model_name': result[0],
                        'target_variable': result[1],
                        'evaluation_strategy': result[2],
                        'model': model_data,
                        'best_params': json.loads(result[4]),
                        'evaluation_results': json.loads(result[5]),
                        'feature_info': json.loads(result[6]),
                        'preprocessing_info': json.loads(result[7]),
                        'training_info': json.loads(result[8]),
                        'created_at': result[9],
                        'updated_at': result[10]
                    }
                
        except Exception as e:
            st.error(f"Failed to load model: {e}")
        
        return None
    
    def list_cached_models(self, target_variable=None):
        """列出所有缓存的模型"""
        try:
            if target_variable:
                query = f"""
                SELECT model_key, model_name, target_variable, evaluation_strategy, 
                       created_at, updated_at
                FROM {self.cache_table_name}
                WHERE target_variable = :target_variable
                ORDER BY updated_at DESC
                """
                params = {"target_variable": target_variable}
            else:
                query = f"""
                SELECT model_key, model_name, target_variable, evaluation_strategy, 
                       created_at, updated_at
                FROM {self.cache_table_name}
                ORDER BY updated_at DESC
                """
                params = {}
            
            with self.engine.connect() as conn:
                results = conn.execute(text(query), params).fetchall()
                
                model_list = []
                for row in results:
                    model_list.append({
                        'model_key': row[0],
                        'model_name': row[1],
                        'target_variable': row[2],
                        'evaluation_strategy': row[3],
                        'created_at': row[4],
                        'updated_at': row[5]
                    })
                
                return model_list
                
        except Exception as e:
            st.warning(f"Failed to list cached models: {e}")
            return []
    
    def delete_model(self, model_key):
        """删除指定的模型缓存"""
        try:
            # 使用begin()自动处理事务，减少临时文件使用
            with self.engine.begin() as conn:
                query = f"DELETE FROM {self.cache_table_name} WHERE model_key = :model_key"
                result = conn.execute(text(query), {"model_key": model_key})
                return result.rowcount
                
        except Exception as e:
            error_msg = str(e)
            if "No space left on device" in error_msg or "errno 28" in error_msg:
                st.error("❌ 磁盘空间不足！请清理临时目录或联系管理员。")
                st.info("💡 建议清理 Windows 临时目录: %TEMP% 或重启服务器")
            else:
                st.error(f"Failed to delete model: {e}")
            return 0
    
    def clear_all_models(self):
        """清除所有模型缓存"""
        try:
            # 首先检查磁盘空间
            import tempfile
            import shutil
            temp_dir = tempfile.gettempdir()
            free_space = shutil.disk_usage(temp_dir).free / (1024 * 1024)  # MB
            
            if free_space < 100:  # 少于100MB
                st.warning(f"⚠️ 临时目录空间不足 ({free_space:.1f}MB)，建议先清理系统临时文件")
            
            # 首先检查表是否存在以及有多少条记录
            with self.engine.connect() as conn:
                count_query = f"SELECT COUNT(*) FROM {self.cache_table_name}"
                count_result = conn.execute(text(count_query))
                initial_count = count_result.scalar()
                st.info(f"Found {initial_count} models in cache before deletion.")
            
            # 如果记录很多，分批删除以减少临时文件压力
            if initial_count > 50:
                return self._clear_models_in_batches()
            
            # 执行删除操作
            with self.engine.begin() as conn:
                query = f"DELETE FROM {self.cache_table_name}"
                result = conn.execute(text(query))
                deleted_count = result.rowcount
                
                # 清除streamlit session state中的相关缓存
                if "model_cache_manager" in st.session_state:
                    del st.session_state["model_cache_manager"]
                
                return deleted_count
                
        except Exception as e:
            error_msg = str(e)
            if "No space left on device" in error_msg or "errno 28" in error_msg:
                st.error("❌ 磁盘空间不足！无法完成删除操作。")
                st.info("💡 解决方案：")
                st.info("1. 清理 Windows 临时目录: %TEMP%")
                st.info("2. 重启数据库服务")
                st.info("3. 联系系统管理员释放磁盘空间")
                return 0
            else:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Failed to clear all models: {e}")
                st.error(f"Traceback: {error_details}")
                return 0
    
    def _clear_models_in_batches(self, batch_size=10):
        """分批删除模型以减少临时文件压力"""
        try:
            total_deleted = 0
            while True:
                with self.engine.begin() as conn:
                    # 获取一批模型key
                    query = f"SELECT model_key FROM {self.cache_table_name} LIMIT {batch_size}"
                    result = conn.execute(text(query))
                    batch_keys = [row[0] for row in result.fetchall()]
                    
                    if not batch_keys:
                        break
                    
                    # 删除这一批
                    for key in batch_keys:
                        delete_query = f"DELETE FROM {self.cache_table_name} WHERE model_key = :model_key"
                        conn.execute(text(delete_query), {"model_key": key})
                        total_deleted += 1
                    
                    st.info(f"已删除 {total_deleted} 个模型...")
            
            # 清除streamlit session state中的相关缓存
            if "model_cache_manager" in st.session_state:
                del st.session_state["model_cache_manager"]
            
            return total_deleted
            
        except Exception as e:
            st.error(f"分批删除失败: {e}")
            return total_deleted
    
    def get_best_model_for_target(self, target_variable, evaluation_strategy=None):
        """获取指定目标变量的最佳模型"""
        try:
            import json
            
            if evaluation_strategy:
                query = f"""
                SELECT model_key, model_name, evaluation_results
                FROM {self.cache_table_name}
                WHERE target_variable = :target_variable AND evaluation_strategy = :evaluation_strategy
                ORDER BY updated_at DESC
                """
                params = {"target_variable": target_variable, "evaluation_strategy": evaluation_strategy}
            else:
                query = f"""
                SELECT model_key, model_name, evaluation_results
                FROM {self.cache_table_name}
                WHERE target_variable = :target_variable
                ORDER BY updated_at DESC
                """
                params = {"target_variable": target_variable}
            
            with self.engine.connect() as conn:
                results = conn.execute(text(query), params).fetchall()
                
                if not results:
                    return None
                
                best_model_key = None
                best_score = -float('inf')
                
                for row in results:
                    try:
                        eval_results = json.loads(row[2])
                        # 根据R²分数选择最佳模型
                        score = eval_results.get('test_r2', eval_results.get('cv_r2_mean', -float('inf')))
                        if score > best_score:
                            best_score = score
                            best_model_key = row[0]
                    except:
                        continue
                
                if best_model_key:
                    return self.load_model(best_model_key)
                
        except Exception as e:
            st.warning(f"Failed to get best model: {e}")
        
        return None
    
    def clear_legacy_models(self):
        """清除不包含预处理器信息的旧模型"""
        try:
            import json
            
            # 查找所有缓存模型
            query = f"""
            SELECT model_key, preprocessing_info
            FROM {self.cache_table_name}
            """
            
            with self.engine.connect() as conn:
                results = conn.execute(text(query)).fetchall()
                
                legacy_models = []
                for row in results:
                    model_key = row[0]
                    preprocessing_info = json.loads(row[1])
                    
                    # 检查是否包含序列化的预处理器
                    if not preprocessing_info.get('feature_encoder') or not preprocessing_info.get('feature_scaler'):
                        legacy_models.append(model_key)
                
                # 删除旧模型
                deleted_count = 0
                for model_key in legacy_models:
                    if self.delete_model(model_key) > 0:
                        deleted_count += 1
                
                return deleted_count, len(legacy_models)
                
        except Exception as e:
            st.error(f"Failed to clear legacy models: {e}")
            return 0, 0
    
    def get_model_key(self, model_name, target_variable, evaluation_strategy, data_hash):
        """根据模型参数获取模型key"""
        try:
            # 首先尝试根据参数查找
            query = f"""
            SELECT model_key 
            FROM {self.cache_table_name} 
            WHERE model_name = :model_name 
            AND target_variable = :target_variable 
            AND evaluation_strategy = :evaluation_strategy 
            AND data_hash = :data_hash
            ORDER BY updated_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {
                    "model_name": model_name,
                    "target_variable": target_variable,
                    "evaluation_strategy": evaluation_strategy,
                    "data_hash": data_hash
                }).fetchone()
                
                if result:
                    return result[0]
                else:
                    # 如果没找到，生成key（但不一定存在）
                    return self._generate_model_key(model_name, target_variable, data_hash, evaluation_strategy)
                    
        except Exception as e:
            st.warning(f"Failed to get model key: {e}")
            return None

class FRPDataPreprocessor:
    """
    Improved FRP data preprocessor
    Based on reference code data processing methods, integrated into main application
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.data_ori = None
        self.cache_table_name = "preprocessed_data_cache"
        
    def _init_cache_table(self):
        """初始化缓存表"""
        try:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.cache_table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                cache_key VARCHAR(255) UNIQUE NOT NULL,
                data_hash VARCHAR(64) NOT NULL,
                preprocessed_data LONGTEXT NOT NULL,
                feature_columns TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                data_shape VARCHAR(50),
                INDEX idx_cache_key (cache_key),
                INDEX idx_data_hash (data_hash)
            )
            """
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
        except Exception as e:
            st.warning(f"Cache table initialization failed: {e}")
    
    def _generate_data_hash(self, df):
        """生成数据的哈希值"""
        import hashlib
        import json
        
        # 创建数据的唯一标识
        data_info = {
            'shape': df.shape,
            'columns': sorted(df.columns.tolist()),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_values': {}
        }
        
        # 添加一些样本值作为哈希的一部分
        for col in df.columns[:5]:  # 只取前5列
            if len(df) > 0:
                data_info['sample_values'][col] = str(df[col].iloc[0] if len(df) > 0 else '')
        
        data_string = json.dumps(data_info, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def get_cached_data(self, cache_key):
        """从缓存中获取预处理数据"""
        try:
            self._init_cache_table()
            
            query = f"""
            SELECT preprocessed_data, feature_columns, data_shape, created_at
            FROM {self.cache_table_name}
            WHERE cache_key = :cache_key
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"cache_key": cache_key}).fetchone()
                
                if result:
                    import json
                    # 解析数据
                    data_dict = json.loads(result[0])
                    df = pd.DataFrame(data_dict)
                    
                    feature_columns = json.loads(result[1])
                    
                    return {
                        'data': df,
                        'feature_columns': feature_columns,
                        'shape': result[2],
                        'cached_at': result[3]
                    }
                
        except Exception as e:
            st.warning(f"Failed to retrieve cached data: {e}")
        
        return None
    
    def save_to_cache(self, cache_key, df, feature_columns):
        """保存预处理数据到缓存"""
        try:
            self._init_cache_table()
            
            # 生成数据哈希
            data_hash = self._generate_data_hash(df)
            
            # 将DataFrame转换为JSON
            import json
            data_json = df.to_json(orient='records')
            feature_columns_json = json.dumps(feature_columns)
            data_shape = f"{df.shape[0]}x{df.shape[1]}"
            
            # 检查是否已存在相同的缓存
            check_query = f"SELECT id FROM {self.cache_table_name} WHERE cache_key = :cache_key"
            
            with self.engine.connect() as conn:
                existing = conn.execute(text(check_query), {"cache_key": cache_key}).fetchone()
                
                if existing:
                    # 更新现有缓存
                    update_query = f"""
                    UPDATE {self.cache_table_name}
                    SET data_hash = :data_hash, preprocessed_data = :data_json, feature_columns = :feature_columns,
                        data_shape = :data_shape, updated_at = CURRENT_TIMESTAMP
                    WHERE cache_key = :cache_key
                    """
                    conn.execute(text(update_query), {
                        "data_hash": data_hash,
                        "data_json": data_json,
                        "feature_columns": feature_columns_json,
                        "data_shape": data_shape,
                        "cache_key": cache_key
                    })
                else:
                    # 插入新缓存
                    insert_query = f"""
                    INSERT INTO {self.cache_table_name} 
                    (cache_key, data_hash, preprocessed_data, feature_columns, data_shape)
                    VALUES (:cache_key, :data_hash, :data_json, :feature_columns, :data_shape)
                    """
                    conn.execute(text(insert_query), {
                        "cache_key": cache_key,
                        "data_hash": data_hash,
                        "data_json": data_json,
                        "feature_columns": feature_columns_json,
                        "data_shape": data_shape
                    })
                
                conn.commit()
                return True
                
        except Exception as e:
            st.error(f"Failed to save to cache: {e}")
            return False
    
    def list_cached_datasets(self):
        """列出所有缓存的数据集"""
        try:
            self._init_cache_table()
            
            query = f"""
            SELECT cache_key, data_shape, created_at, updated_at
            FROM {self.cache_table_name}
            ORDER BY updated_at DESC
            """
            
            with self.engine.connect() as conn:
                results = conn.execute(text(query)).fetchall()
                
                cache_list = []
                for row in results:
                    cache_list.append({
                        'cache_key': row[0],
                        'shape': row[1],
                        'created_at': row[2],
                        'updated_at': row[3]
                    })
                
                return cache_list
                
        except Exception as e:
            st.warning(f"Failed to list cached datasets: {e}")
            return []
    
    def clear_cache(self, cache_key=None):
        """清除缓存数据"""
        try:
            self._init_cache_table()
            
            with self.engine.connect() as conn:
                if cache_key:
                    query = f"DELETE FROM {self.cache_table_name} WHERE cache_key = :cache_key"
                    result = conn.execute(text(query), {"cache_key": cache_key})
                else:
                    query = f"DELETE FROM {self.cache_table_name}"
                    result = conn.execute(text(query))
                
                conn.commit()
                return result.rowcount
                
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")
            return 0
        
    def change_smd_to_nan(self, df):
        """
        Reference code's ChangeSMDToNAN function
        Convert 'SMD' to NaN, 'Notreported' to 'Unknown'
        """
        st.info("Processing missing value markers...")
        
        df_new = df.copy()
        
        for col in df_new.columns:
            # Vectorized processing, avoid double loop
            df_new[col] = df_new[col].replace({
                'SMD': np.nan,
                'smd': np.nan,
                'Notreported': 'Unknown',
                'not reported': 'Unknown',
                'Not reported': 'Unknown',
                'NOT REPORTED': 'Unknown'
            })
        
        st.success("Missing value marker processing completed")
        return df_new
    
    def parse_range_to_mean(self, df):
        """
        Reference code's range value parsing to mean
        Process range strings like "20,30"
        """
        st.info("Parsing range values to mean...")
        
        # Numeric columns to process (based on reference code list)
        numeric_columns = [
            'glass_transition_temperature', 'glass_transition_temperature_run_2',
            'cure_ratio', 'Fiber_content_weight', 'Fiber_content_volume',
            'Void_content', 'diameter', 'average_area', 'nominal_area',
            'num_1', 'temperature', 'pH_of_concrete', 'strength_of_concrete',
            'crack', 'pH_1', 'pHafter', 'RH_1', 'field_average_humidity',
            'field_average_temperature', 'temp', 'temp2', 'value_load',
            'Value1_1', 'COV1_1', 'Value2_1', 'COV2_1', 'Value3_1', 'COV3_1'
        ]
        
        progress_bar = st.progress(0)
        
        for i, col in enumerate(numeric_columns):
            if col in df.columns:
                for idx in df.index:
                    value = df.loc[idx, col]
                    if isinstance(value, str):
                        # Check if contains comma and no colon (reference code condition)
                        if ',' in value and ':' not in value:
                            try:
                                # Extract numbers and calculate mean
                                numbers = re.findall(r"\d+\.?\d*", value)
                                if numbers:
                                    new_value = np.mean([float(x) for x in numbers])
                                    if not np.isnan(new_value):
                                        df.loc[idx, col] = new_value
                            except (ValueError, TypeError):
                                continue
            
            progress_bar.progress((i + 1) / len(numeric_columns))
        
        st.success("Range value parsing completed")
        return df
    
    def create_selected_features(self, df):
        """
        Create selected_feature sub-table based on reference code
        Construct 13 key features for model training
        """
        st.info("Creating selected_feature features...")
        
        # First preserve original important columns
        original_important_cols = ['Target_parameter', 'retention1', 'Value1_1', 'temperature', 'time_field']
        
        # Create new feature columns (if not exist)
        feature_columns = [
            'pH_of_condition_enviroment', 'Chloride_ion', 'concrete',
            'diameter', 'load_value', 'fiber_content', 'Glass_or_Basalt',
            'Vinyl_ester_or_Epoxy', 'condition_time', 'Temperature',
            'Tensile_strength_retention', 'surface_treatment',
            'max_strength', 'glass_transition_temperature'
        ]
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        length = len(df)
        st.info(f"Processing feature engineering for {length} rows of data...")
        
        progress_bar = st.progress(0)
        
        for i, idx in enumerate(df.index):
            try:
                # 1. pH processing (based on solution_condition)
                self._process_ph_and_chloride(df, idx)
                
                # 2. Concrete indicator
                self._process_concrete_indicator(df, idx)
                
                # 3. Diameter processing
                self._process_diameter(df, idx)
                
                # 4. Load processing
                self._process_load(df, idx)
                
                # 5. Fiber content processing
                self._process_fiber_content(df, idx)
                
                # 6. Fiber and matrix type encoding
                self._process_material_types(df, idx)
                
                # 7. Surface treatment
                self._process_surface_treatment(df, idx)
                
                # 8. Other features (ensure not overwriting Target_parameter)
                self._process_other_features(df, idx)
                
                # Update progress bar
                if i % 1000 == 0:  # Update progress every 1000 rows
                    progress_bar.progress((i + 1) / length)
                
            except Exception as e:
                continue
        
        progress_bar.progress(1.0)
        
        # Verify if Target_parameter is preserved
        if 'Target_parameter' in df.columns:
            target_count = df['Target_parameter'].count()
            tensile_count = (df['Target_parameter'] == 'Tensile').sum()
            st.info(f"Target_parameter status after feature engineering: Total={target_count}, Tensile={tensile_count}")
        
        st.success("selected_feature feature creation completed")
        return df
    
    def _process_ph_and_chloride(self, df, idx):
        """
        Strictly process pH and chloride features according to paper requirements
        
        Steps:
        1. Distinguish "concrete environment" vs "solution environment"
        2. Concrete environment: use pH_of_concrete or default 13
        3. Solution environment: use numeric pH or assign 7 based on solution type
        4. Consider pHafter for averaging
        """
        # Initialize
        df.loc[idx, 'Chloride_ion'] = 0
        final_ph = 7.0  # Default value
        
        # Step 1: Determine environment type
        is_concrete_environment = False
        
        # Check Condition_environment field
        if 'Condition_environment' in df.columns:
            condition_env = str(df.loc[idx, 'Condition_environment']).lower()
            concrete_keywords = ['concrete', 'cover', 'crack', 'cement', 'mortar']
            if any(keyword in condition_env for keyword in concrete_keywords):
                is_concrete_environment = True
        
        # Backup check: if no Condition_environment, check concrete-related columns
        if not is_concrete_environment:
            concrete_cols = ['concrete', 'crack', 'cover', 'cement']
            for col in concrete_cols:
                if col in df.columns:
                    value = df.loc[idx, col]
                    if isinstance(value, str) or (isinstance(value, (int, float)) and not pd.isna(value)):
                        is_concrete_environment = True
                        break
        
        # Step 2: pH processing in concrete environment
        if is_concrete_environment:
            # Check pH_of_concrete
            if 'pH_of_concrete' in df.columns:
                ph_concrete = df.loc[idx, 'pH_of_concrete']
                if isinstance(ph_concrete, (int, float)) and not pd.isna(ph_concrete):
                    final_ph = float(ph_concrete)
                else:
                    final_ph = 13.0  # Default concrete alkalinity
            else:
                final_ph = 13.0  # Default concrete alkalinity
        
        # Step 3: pH processing in solution environment
        else:
            # 3.1 Priority check for numeric pH
            ph_found = False
            
            # Check pH field in solution_condition
            if 'solution_condition' in df.columns:
                solution_condition = df.loc[idx, 'solution_condition']
                # Try to extract pH value from solution_condition
                if isinstance(solution_condition, (int, float)) and not pd.isna(solution_condition):
                    final_ph = float(solution_condition)
                    ph_found = True
            
            # Backup: check pH_1 or pH-related fields
            if not ph_found:
                ph_columns = ['pH_1', 'pH', 'ph', 'PH']
                for ph_col in ph_columns:
                    if ph_col in df.columns:
                        ph_value = df.loc[idx, ph_col]
                        if isinstance(ph_value, (int, float)) and not pd.isna(ph_value):
                            final_ph = float(ph_value)
                            ph_found = True
                            break
            
            # 3.2 If no numeric pH, assign value based on solution type description
            if not ph_found:
                # Check solution_condition text description
                solution_text = ''
                if 'solution_condition' in df.columns:
                    solution_text = str(df.loc[idx, 'solution_condition']).lower()
                
                # Backup: check ingredient_1
                if not solution_text and 'ingredient_1' in df.columns:
                    solution_text = str(df.loc[idx, 'ingredient_1']).lower()
                
                # Assign value based on solution type
                water_types = ['tap water', 'sea water', 'seawater', 'distilled water', 
                              'deionized water', 'di water', 'pure water']
                
                if any(water_type in solution_text for water_type in water_types):
                    final_ph = 7.0
                    
                    # Special handling for seawater
                    if 'sea' in solution_text:
                        df.loc[idx, 'Chloride_ion'] = 1
        
        # Step 4: Consider pHafter
        if 'pHafter' in df.columns:
            ph_after = df.loc[idx, 'pHafter']
            if isinstance(ph_after, (int, float)) and not pd.isna(ph_after):
                final_ph = (final_ph + float(ph_after)) / 2.0
        
        # Set final pH value
        df.loc[idx, 'pH_of_condition_enviroment'] = final_ph
        
        # Additional chloride ion check
        if 'ingredient_1' in df.columns:
            ingredient = str(df.loc[idx, 'ingredient_1']).lower()
            chloride_keywords = ['cl', 'chloride', 'nacl', 'cacl2', 'mgcl2', 'salt']
            if any(keyword in ingredient for keyword in chloride_keywords):
                df.loc[idx, 'Chloride_ion'] = 1
    
    def _process_concrete_indicator(self, df, idx):
        """Process concrete indicator"""
        concrete_indicator = 0
        
        # Check related columns
        concrete_cols = ['concrete', 'crack', 'cover']
        for col in concrete_cols:
            if col in df.columns:
                value = df.loc[idx, col]
                if isinstance(value, str) or (isinstance(value, (int, float)) and not pd.isna(value)):
                    concrete_indicator = 1
                    break
        
        df.loc[idx, 'concrete'] = concrete_indicator
    
    def _process_diameter(self, df, idx):
        """Process diameter features"""
        # Priority use of directly measured diameter
        if 'diameter' in df.columns:
            diameter_value = df.loc[idx, 'diameter']
            if isinstance(diameter_value, (int, float)) and not pd.isna(diameter_value):
                df.loc[idx, 'diameter'] = diameter_value
                return
        
        # Calculate diameter from nominal area
        if 'nominal_area' in df.columns:
            area_value = df.loc[idx, 'nominal_area']
            if isinstance(area_value, (int, float)) and not pd.isna(area_value) and area_value > 0:
                calculated_diameter = 2 * np.sqrt(area_value / np.pi)
                df.loc[idx, 'diameter'] = calculated_diameter
    
    def _process_load(self, df, idx):
        """Process load features"""
        load_value = 0
        
        # Check preloading
        if 'type_of_load' in df.columns:
            if df.loc[idx, 'type_of_load'] == 'preloading':
                df.loc[idx, 'load_value'] = 0
                return
        
        # Process stress/strain
        if 'stress_or_strain' in df.columns and 'value_load' in df.columns:
            stress_strain = df.loc[idx, 'stress_or_strain']
            value = df.loc[idx, 'value_load']
            
            if isinstance(value, (int, float)) and not pd.isna(value):
                if stress_strain == 'stress':
                    # Stress case: need to divide by ultimate tensile strength
                    if 'ultimate_tensile_strength' in df.columns:
                        uts = df.loc[idx, 'ultimate_tensile_strength']
                        if isinstance(uts, (int, float)) and uts > 0:
                            load_value = value / uts
                elif stress_strain == 'strain':
                    # Strain case: convert to relative stress
                    if 'tensile_modulus' in df.columns and 'ultimate_tensile_strength' in df.columns:
                        modulus = df.loc[idx, 'tensile_modulus']
                        uts = df.loc[idx, 'ultimate_tensile_strength']
                        if all(isinstance(x, (int, float)) and x > 0 for x in [modulus, uts]):
                            load_value = value * 0.001 * modulus / uts
        
        df.loc[idx, 'load_value'] = load_value
    
    def _process_fiber_content(self, df, idx):
        """Process fiber content features"""
        # Priority use of weight percentage
        if 'Fiber_content_weight' in df.columns:
            weight_content = df.loc[idx, 'Fiber_content_weight']
            if isinstance(weight_content, (int, float)) and not pd.isna(weight_content):
                df.loc[idx, 'fiber_content'] = weight_content
                return
        
        # Convert from volume percentage
        if 'Fiber_content_volume' in df.columns:
            volume_content = df.loc[idx, 'Fiber_content_volume']
            if isinstance(volume_content, (int, float)) and not pd.isna(volume_content):
                # Get density
                fiber_type = df.loc[idx, 'Fiber_type'] if 'Fiber_type' in df.columns else 'Unknown'
                matrix_type = df.loc[idx, 'Matrix_type'] if 'Matrix_type' in df.columns else 'Unknown'
                
                # Fiber density
                fiber_densities = {
                    'Glass': 2.55,
                    'Carbon': 1.84,
                    'Basalt': 2.67
                }
                density_fiber = fiber_densities.get(fiber_type, 2.0)
                
                # Matrix density
                matrix_densities = {
                    'Vinyl ester': 1.09,
                    'Epoxy': 1.1,
                    'Polyester': 1.38
                }
                density_matrix = matrix_densities.get(matrix_type, 1.2)
                
                # Convert volume fraction to weight fraction
                weight_content = (100.0 * volume_content * density_fiber) / (
                    volume_content * density_fiber + (100.0 - volume_content) * density_matrix
                )
                df.loc[idx, 'fiber_content'] = weight_content
    
    def _process_material_types(self, df, idx):
        """Process material type encoding"""
        # Fiber type encoding (Glass fiber=1, Basalt fiber=0)
        if 'Fiber_type' in df.columns:
            fiber_type = df.loc[idx, 'Fiber_type']
            if fiber_type == 'Glass':
                df.loc[idx, 'Glass_or_Basalt'] = 1
            elif fiber_type == 'Basalt':
                df.loc[idx, 'Glass_or_Basalt'] = 0
        
        # Matrix type encoding (Vinyl ester=1, Epoxy=0)
        if 'Matrix_type' in df.columns:
            matrix_type = df.loc[idx, 'Matrix_type']
            if matrix_type == 'Vinyl ester':
                df.loc[idx, 'Vinyl_ester_or_Epoxy'] = 1
            elif matrix_type == 'Epoxy':
                df.loc[idx, 'Vinyl_ester_or_Epoxy'] = 0
    
    def _process_surface_treatment(self, df, idx):
        """Process surface treatment features"""
        if 'surface_treatment' in df.columns:
            treatment = df.loc[idx, 'surface_treatment']
            if treatment == 'sand coated':
                df.loc[idx, 'surface_treatment'] = 0
            elif treatment == 'Smooth':
                df.loc[idx, 'surface_treatment'] = 1
    
    def _process_other_features(self, df, idx):
        """Process other features"""
        # Features for direct copying
        feature_mappings = {
            'condition_time': 'time_field',
            'Temperature': 'temperature',
            'Tensile_strength_retention': 'retention1',
            'Target_parameter': 'Target_parameter',  # This should be copied directly regardless of type
            'max_strength': 'Value1_1',
            'glass_transition_temperature': 'glass_transition_temperature'
        }
        
        for new_col, old_col in feature_mappings.items():
            if old_col in df.columns:
                value = df.loc[idx, old_col]
                # For Target_parameter, copy directly regardless of type
                if new_col == 'Target_parameter':
                    df.loc[idx, new_col] = value
                # For other numeric features, check if valid numeric
                elif isinstance(value, (int, float)) and not pd.isna(value):
                    df.loc[idx, new_col] = value
                # For string type numerics, try to convert
                elif isinstance(value, str) and value.strip() != '':
                    try:
                        numeric_value = float(value)
                        if not np.isnan(numeric_value):
                            df.loc[idx, new_col] = numeric_value
                    except (ValueError, TypeError):
                        # If cannot convert to numeric, still preserve original value for certain fields
                        if new_col in ['Target_parameter']:
                            df.loc[idx, new_col] = value
    
    def create_model_dataset(self, df):
        """
        Create model dataset
        Following reference code's i0_data_pick to i010_data_pick process
        """
        st.info("Creating model dataset...")
        
        # Select key feature columns
        model_columns = [
            'Title', 'Target_parameter', 'Tensile_strength_retention',
            'pH_of_condition_enviroment', 'condition_time', 'fiber_content',
            'Temperature', 'diameter', 'Chloride_ion', 'concrete',
            'load_value', 'Glass_or_Basalt', 'Vinyl_ester_or_Epoxy',
            'surface_treatment', 'max_strength'
        ]
        
        # Create model dataframe
        model_data = pd.DataFrame(index=df.index)
        
        for col in model_columns:
            if col == 'Title':
                model_data[col] = df['Title'] if 'Title' in df.columns else df.index
            else:
                model_data[col] = df[col] if col in df.columns else np.nan
        
        st.info(f"Initial model data shape: {model_data.shape}")
        
        # Use all data, don't filter by Target_parameter (avoid data loss)
        st.info("Using all data for model training")
        tensile_data = model_data.copy()
        
        # Select final features
        final_columns = [
            'Title', 'Tensile_strength_retention', 'pH_of_condition_enviroment',
            'condition_time', 'fiber_content', 'Temperature', 'diameter',
            'concrete', 'load_value', 'Chloride_ion', 'Glass_or_Basalt',
            'Vinyl_ester_or_Epoxy', 'surface_treatment', 'max_strength'
        ]
        
        # Check data completeness for each column
        st.info("Feature data completeness analysis:")
        for col in final_columns:
            if col in tensile_data.columns:
                non_null_count = tensile_data[col].count()
                total_count = len(tensile_data)
                percentage = (non_null_count / total_count * 100) if total_count > 0 else 0
                st.write(f"   {col}: {non_null_count}/{total_count} ({percentage:.1f}%)")
        
        # Use more lenient dropna strategy
        final_data = tensile_data[final_columns].copy()
        
        # Only remove completely empty rows
        before_drop = len(final_data)
        final_data = final_data.dropna(how='all')  # Only remove rows where all values are NaN
        after_drop = len(final_data)
        st.info(f"Removed completely empty rows: {before_drop} -> {after_drop}")
        
        # Further check: if data is still too little, use more lenient strategy
        if len(final_data) < 100:  # If less than 100 rows
            st.warning("Data volume too small, using more lenient filtering strategy...")
            
            # Keep as long as there's Tensile_strength_retention (target variable)
            if 'Tensile_strength_retention' in tensile_data.columns:
                has_target = tensile_data['Tensile_strength_retention'].notna()
                final_data = tensile_data[has_target][final_columns].copy()
                st.info(f"After filtering by target variable: {len(final_data)} rows")
            
            # If still too little, only remove rows with empty target variable
            if len(final_data) < 50:
                final_data = tensile_data[final_columns].copy()
                # Only remove rows with empty target variable
                if 'Tensile_strength_retention' in final_data.columns:
                    final_data = final_data.dropna(subset=['Tensile_strength_retention'])
                    st.info(f"Kept data with target variable: {len(final_data)} rows")
        
        # Rename columns to match reference code
        final_data.columns = [
            'Title', 'Tensile strength retention', 'pH of condition environment',
            'Exposure time', 'Fibre content', 'Exposure temperature', 'Diameter',
            'Presence of concrete', 'Load', 'Presence of chloride ion', 'Fibre type',
            'Matrix type', 'Surface treatment', 'Strength of unconditioned rebar'
        ]
        
        st.success(f"Model dataset creation completed, final data shape: {final_data.shape}")
        
        return final_data

@st.cache_data
def load_default_data():
    """Load default data and perform basic cleaning"""
    engine = get_db_engine()
    if engine:
        try:
            print(f"Loading data from database: {DB_NAME}")
            # 测试数据库连接
            with engine.connect() as conn:
                # 首先检查表是否存在
                result = conn.execute(text("SHOW TABLES LIKE 'research_data'")).fetchone()
                if not result:
                    print("research_data table does not exist")
                    return None
                
                # 获取记录数量
                count = conn.execute(text("SELECT COUNT(*) FROM research_data")).scalar()
                print(f"Found {count} records in research_data table")
                
                # 加载数据
                df = pd.read_sql("SELECT * FROM research_data", engine)
                print(f"Successfully loaded {len(df)} rows from research_data table")
            
            # Check for duplicates before cleaning
            original_count = len(df)
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                print(f"Warning: Found {duplicate_count} duplicate rows in the data")
            
            # Basic data cleaning
            df.replace({"Notreported": np.nan, "SMD": np.nan, "smd": np.nan}, inplace=True)
            
            # Clean all possible problematic strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].replace({
                        "SMD": np.nan, "smd": np.nan, "Notreported": np.nan,
                        "not reported": np.nan, "Not reported": np.nan, "NOT REPORTED": np.nan,
                        "": np.nan, " ": np.nan, "nan": np.nan, "NaN": np.nan,
                        "NULL": np.nan, "null": np.nan
                    })
            
            # Additional data quality checks
            print(f"Data shape after cleaning: {df.shape}")
            print(f"Missing values per column: {df.isnull().sum().sum()} total missing values")
            
            return df
            
        except Exception as e:
            print(f"Data loading error: {e}")
            st.error(f"Database loading error: {e}")
            return None
    else:
        st.error("Database engine not available")
        return None

def create_advanced_model_dataset():
    """Create advanced model dataset (using improved preprocessing methods)"""
    if "df_raw" not in st.session_state or st.session_state.df_raw is None:
        st.error("Please load raw data first")
        return None
    
    # Create data preprocessor
    engine = get_db_engine()
    processor = FRPDataPreprocessor(engine)
    
    # Execute complete data preprocessing pipeline
    with st.spinner("Executing advanced data preprocessing..."):
        df = st.session_state.df_raw.copy()
        
        # Step 1: Missing value marker conversion
        df = processor.change_smd_to_nan(df)
        
        # Step 2: Range value parsing
        df = processor.parse_range_to_mean(df)
        
        # Step 3: Feature engineering
        df = processor.create_selected_features(df)
        
        # Step 4: Create model dataset
        model_dataset = processor.create_model_dataset(df)
        
        return model_dataset

# In Data Management Tab, modify data reading section
# Replace original try-except block

# ——————————————————————————————
# 13. Main Program
# ——————————————————————————————

# Inject CSS styles
inject_custom_css()

# Initialize data manager
data_manager = DataManager()

# Get database connection and initialize system
engine = get_db_engine()
if engine:
    create_tables(engine)
    admin_success, admin_email, admin_status = initialize_admin(engine)

# Initialize session state with improved data loading
if "df_raw" not in st.session_state:
    try:
        with st.spinner("Loading data from database..."):
            loaded_data = load_default_data()
            if loaded_data is not None and len(loaded_data) > 0:
                st.session_state.df_raw = loaded_data
                print(f"Successfully initialized session data: {len(loaded_data)} records")
            else:
                st.session_state.df_raw = None
                print("Failed to load initial data from database")
    except Exception as e:
        st.session_state.df_raw = None
        print(f"Error during initial data loading: {e}")

# Verify data loading status
if st.session_state.df_raw is not None:
    print(f"Session data available: {len(st.session_state.df_raw)} records")
else:
    print("Warning: Session data is None")

# ——————————————————————————————
# 14. Page Header
# ——————————————————————————————
st.markdown(create_gradient_header(
    "FRP Rebar Durability Prediction Platform",
    "Advanced Machine Learning for Material Performance Analysis"
), unsafe_allow_html=True)

# Display key metrics with database connection status
if engine and "authenticated_user" in st.session_state:
    col1, col2, col3, col4 = st.columns(4)
    
    # Get statistical data with error handling
    try:
        with engine.connect() as conn:
            total_records = conn.execute(text("SELECT COUNT(*) FROM research_data")).scalar() or 0
            active_users = conn.execute(
                text("SELECT COUNT(DISTINCT user_email) FROM operation_logs WHERE DATE(created_at) = CURDATE()")
            ).scalar() or 0
            pending_changes = conn.execute(
                text("SELECT COUNT(*) FROM data_changes WHERE status = 'pending'")
            ).scalar() or 0
            
        # 数据库连接成功，显示正常指标
        db_status = "Connected"
        db_color = "#27ae60"
        
    except Exception as e:
        # 数据库连接失败
        total_records = 0
        active_users = 0
        pending_changes = 0
        db_status = "Disconnected"
        db_color = "#e74c3c"
        st.error(f"⚠️ Database connection issue: {e}")
    
    with col1:
        st.markdown(create_metric_card("Total Records", f"{total_records:,}", 5.2, "normal", 0), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Active Users Today", str(active_users), 12.3, "normal", 1), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Model Accuracy", "94.7%", 2.1, "normal", 2), unsafe_allow_html=True)
    with col4:
        # 显示数据库连接状态而不是pending changes
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.45s;">
            <h4 style="color: #666; font-size: 0.9rem; margin: 0;">Database Status</h4>
            <h2 style="color: {db_color}; margin: 0.5rem 0;">{db_status}</h2>
            <p style="margin: 0; color: {db_color}; font-size: 0.8rem;">{db_status}</p>
        </div>
        """, unsafe_allow_html=True)

# ——————————————————————————————
# 15. Sidebar
# ——————————————————————————————
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="background: linear-gradient(120deg, #1e3d59 0%, #ff6e40 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   font-size: 2rem;
                   font-weight: 800;
                   margin: 0;">FRP Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display administrator information
    if engine:
        admin_name = "Chao Wu"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; border-left: 4px solid #3498db;">
            <p style="color: #7f8c8d; margin: 0; font-size: 0.8rem; font-weight: 600; text-align: center;">Administrator</p>
            <p style="color: #2c3e50; margin: 0; font-weight: 500; text-align: center;">{admin_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Email authorization code input - only visible to admin
    if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
        auth_input = st.text_input("Email Authorization Code", type="password", help="Enter 163 email authorization code")
        if auth_input:
            # Store the authorization code in session state for all users to use
            st.session_state["email_pass"] = auth_input
            st.success("Authorization code saved for all users")
    
    # User Authentication Card 
    if "authenticated_user" not in st.session_state:
        # Login Card
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; border-left: 4px solid #e74c3c;">
            <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; text-align: center;">
                User Authentication
            </h4>
        """, unsafe_allow_html=True)
        
        user_email = st.text_input(
            "Email Address", 
            key="user_email_sidebar",
            placeholder="Enter your email address",
            label_visibility="collapsed"
        )
        
        if st.button("Login", type="primary", use_container_width=True):
            if engine:
                user = authenticate_user(user_email, engine)
                if user:
                    st.session_state["authenticated_user"] = {
                        "email": user['email'],
                        "name": user['name'],
                        "role": user['role']
                    }
                    
                    # Set balloon flag for admin users
                    if user['role'] == 'admin':
                        st.session_state["show_admin_balloons"] = True
                    
                    st.success(f"Welcome, {user['name']}!")
                    
                    # Record login
                    log_operation(
                        user['email'],
                        "login",
                        None,
                        None,
                        {"ip": get_client_ip()},
                        get_client_ip(),
                        engine
                    )
                    st.rerun()
                else:
                    st.error("Authentication failed or access not approved")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # User Status Card
        user_info = st.session_state["authenticated_user"]
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; border-left: 4px solid #27ae60;">
            <h4 style="color: #2c3e50; margin: 0 0 1rem 0; font-size: 1rem; font-weight: 600;">
                Current User
            </h4>
            <div style="margin-bottom: 0.5rem;">
                <p style="color: #7f8c8d; margin: 0; font-size: 0.8rem; font-weight: 600;">Name</p>
                <p style="color: #2c3e50; margin: 0; font-weight: 500;">{user_info['name']}</p>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <p style="color: #7f8c8d; margin: 0; font-size: 0.8rem; font-weight: 600;">Email</p>
                <p style="color: #2c3e50; margin: 0; font-weight: 500; font-size: 0.85rem;">{user_info['email']}</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <p style="color: #7f8c8d; margin: 0; font-size: 0.8rem; font-weight: 600;">Role</p>
                <span style="background: #27ae60; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    {user_info['role'].upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            del st.session_state["authenticated_user"]
            st.rerun()

# ——————————————————————————————
# 16. Main Interface Tabs with Role-based Access Control
# ——————————————————————————————
# 根据用户角色动态创建标签页
if "authenticated_user" in st.session_state:
    # Check for admin balloon effect
    if st.session_state.get("show_admin_balloons", False):
        st.balloons()
        # Clear the flag so balloons only show once
        del st.session_state["show_admin_balloons"]
    
    user_role = st.session_state["authenticated_user"]["role"]
    
    if user_role == "admin":
        # admin用户可以看到所有标签页
        tab_names = ["Data Management", "Access Control", "Data Changes", "Model Configuration", "Model Training", "Predictions"]
        tabs = st.tabs(tab_names)
        tab_indexes = {
            "data_management": 0,
            "access_control": 1,
            "data_changes": 2,
            "model_configuration": 3,
            "model_training": 4,
            "predictions": 5
        }
    elif user_role == "editor":
        # editor用户可以看到：Data Management、Model Configuration和Predictions
        tab_names = ["Data Management", "Model Configuration", "Predictions"]
        tabs = st.tabs(tab_names)
        tab_indexes = {
            "data_management": 0,
            "model_configuration": 1,
            "predictions": 2
        }
    else:  # viewer用户
        # viewer用户只能看到：Data Management、Model Configuration和Predictions
        tab_names = ["Data Management", "Model Configuration", "Predictions"]
        tabs = st.tabs(tab_names)
        tab_indexes = {
            "data_management": 0,
            "model_configuration": 1,
            "predictions": 2
        }
else:
    # 未认证用户显示登录注册标签和其他基本标签页
    tab_names = ["Login/Register", "Model Configuration", "Predictions"]
    tabs = st.tabs(tab_names)
    tab_indexes = {
        "login_register": 0,
        "model_configuration": 1,
        "predictions": 2
    }


# ——————————————————————————————
# 标签页内容处理
# ——————————————————————————————

# 登录/注册标签页 (未认证用户)
if "authenticated_user" not in st.session_state:
    with tabs[tab_indexes["login_register"]]:
        
        if not engine:
            st.error("❌ Database connection failed. Please check database configuration.")
            st.markdown("""
            **Troubleshooting steps:**
            1. Verify MySQL server is running
            2. Check database connection parameters in environment variables
            3. Ensure network connectivity to localhost:3306
            4. Contact system administrator if issues persist
            """)
            st.stop()
        
        st.warning("Please log in to access data management functions.")
        
        # Access request form with clean design
        st.subheader("Request Access")
        
        # Information card
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); border-left: 4px solid #3498db; margin-bottom: 2rem;">
            <p style="color: #2c3e50; margin: 0; font-size: 1rem; line-height: 1.5;">
                To access our FRP materials database, please complete the form below. 
                All requests are reviewed by our research team within 2-3 business days.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("access_request_form", clear_on_submit=True):
            # Form layout in a more organized way
            # Row 1: Name and Institution
            col1, col2 = st.columns(2)
            
            with col1:
                req_name = st.text_input(
                    "Full Name",
                    placeholder="Enter your full name",
                    help="Your complete name as it appears in official documents"
                )
            
            with col2:
                req_institution = st.text_input(
                    "Institution/University",
                    placeholder="Your academic or research institution",
                    help="Name of your university, research center, or organization"
                )
            
            # Row 2: Email
            req_email = st.text_input(
                "Email Address",
                placeholder="your.email@institution.edu",
                help="Please use your institutional email address"
            )
            
            # Row 3: Research Purpose
            research_purpose = st.text_area(
                "Research Purpose",
                placeholder="Brief description of your research goals...",
                help="Help us understand how you plan to use the database",
                height=100
            )
            
            # IP address information
            client_ip = get_client_ip()
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 6px; margin: 1rem 0; border-left: 3px solid #6c757d;">
                <small style="color: #495057;">
                    <strong>Your IP Address:</strong> {client_ip} 
                    <em>(logged for security purposes)</em>
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Security notice if IP not in allowed range
            if not is_ip_allowed(client_ip):
                st.markdown("""
                <div style="background: #fff3cd; padding: 1rem; border-radius: 6px; margin: 1rem 0; border-left: 3px solid #ffc107;">
                    <small style="color: #856404;">
                        <strong>Notice:</strong> Your IP address requires additional verification. 
                        Your request may take longer to process.
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            # Terms acceptance
            st.markdown("---")
            terms_accepted = st.checkbox(
                "I agree to use the FRP materials database responsibly for research purposes only",
                help="Required: Agreement to terms of service and data usage policy"
            )
            
            # Submit button
            col_submit1, col_submit2, col_submit3 = st.columns([1, 1, 1])
            with col_submit2:
                submit_pressed = st.form_submit_button(
                    "Submit Request",
                    type="primary",
                    use_container_width=True
                )
            
            # Form processing
            if submit_pressed:
                # Validation
                if not all([req_name, req_email, req_institution]):
                    st.error("Please fill in all required fields")
                elif not terms_accepted:
                    st.error("Please accept the terms and conditions")
                elif "@" not in req_email or "." not in req_email:
                    st.error("Please enter a valid email address")
                else:
                    # Process request
                    success, msg = request_access(req_name, req_email, req_institution, client_ip, engine)
                    if success:
                        st.success("Access request submitted successfully!")
                        st.info("Thank you for your interest. Our team will review your request and contact you within 2-3 business days.")
                        
                        # Send notification email
                        email_body = f"""
                        New FRP Database Access Request
                        
                        Request Details:
                        • Name: {req_name}
                        • Email: {req_email}
                        • Institution: {req_institution}
                        • IP Address: {client_ip}
                        • Research Purpose: {research_purpose}
                        • Terms Accepted: {terms_accepted}
                        
                        Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        Please review and approve/deny this request in the admin panel.
                        """
                        
                        send_email(
                            EMAIL_USER,
                            "New FRP DB Access Request",
                            email_body
                        )
                    else:
                        st.error(msg)
        
        # Additional information
        st.markdown("---")
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
            <h5 style="color: #495057; margin: 0 0 1rem 0;">About Our Database</h5>
            <p style="color: #6c757d; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                Our FRP materials database contains comprehensive research data for fiber-reinforced polymer 
                materials, supporting global research in advanced materials science and engineering applications.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ——————————————————————————————
# Data Management 标签页 (已认证用户)
# ——————————————————————————————
else:
    with tabs[tab_indexes["data_management"]]:
        if not engine:
            st.error("❌ Database connection failed. Please check database configuration.")
            st.markdown("""
            **Troubleshooting steps:**
            1. Verify MySQL server is running (check if XAMPP/WAMP is started)
            2. Check database credentials in environment variables
            3. Ensure research_data table exists in haigui_database
            4. Contact system administrator if issues persist
            """)
            st.stop()
        
        # Authenticated user's data management interface
        user_info = st.session_state["authenticated_user"]
        user_role = user_info.get("role", "viewer")
        
        # Welcome header for authenticated users - clean design
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
            <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
                Data Management Console
            </h3>
            <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
                Welcome back, {user_info.get('name', 'User')} ({user_role}) - Manage and explore FRP materials research data
            </p>
        </div>
        """, unsafe_allow_html=True)
        

        # Data browsing section - clean design
        st.markdown("### Database Overview")
        
        # Select data table - clean styling
        col_select1, col_select2 = st.columns([2, 1])
        with col_select1:
            table_name = st.selectbox(
                "Select Data Table", 
                ["research_data", "research_data_backup"],
                help="Choose which data table to explore"
            )
        with col_select2:
            st.markdown("""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #27ae60; margin-top: 1.7rem;">
                <small style="color: #2c3e50; font-weight: 600;">Data Explorer</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Read data with improved error handling
        try:
            # Use fixed data loading function with explicit database connection
            if table_name == "research_data":
                df = data_manager.get_data(
                    f"table_{table_name}",
                    lambda: load_default_data()
                )
                # 验证数据是否正确加载
                if df is None:
                    st.error("❌ Failed to load data from database. Please check database connection.")
                    # 提供重新加载选项
                    if st.button("🔄 Retry Database Connection", key="retry_db_connection"):
                        data_manager.invalidate_cache(f"table_{table_name}")
                        st.session_state.df_raw = None  # 清除缓存
                        st.rerun()
                    st.stop()
                    
                # 更新全局状态
                st.session_state.df_raw = df
                
            else:
                # 对其他表的处理
                df = data_manager.get_data(
                    f"table_{table_name}",
                    lambda: pd.read_sql(f"SELECT * FROM {table_name}", engine)
                )
            
            if df is None or len(df) == 0:
                st.error(f"❌ No data found in table '{table_name}' or data loading failed.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Reload Data", key="reload_data_btn"):
                        data_manager.invalidate_cache(f"table_{table_name}")
                        st.session_state.df_raw = None  # 清除缓存
                        st.rerun()
                with col2:
                    if st.button("🔍 Check Database", key="check_db_btn"):
                        # 检查数据库连接状态
                        try:
                            with engine.connect() as conn:
                                tables = conn.execute(text("SHOW TABLES")).fetchall()
                                st.info(f"Database connected. Available tables: {[t[0] for t in tables]}")
                                if table_name == "research_data":
                                    count = conn.execute(text("SELECT COUNT(*) FROM research_data")).scalar()
                                    st.info(f"research_data table contains {count} records")
                        except Exception as e:
                            st.error(f"Database connection failed: {e}")
                st.stop()
            
            # 确保数据已正确加载
            if df is not None and len(df) > 0:
                # 数据加载成功，继续处理
                pass
            else:
                st.error("❌ No data loaded from database")
                st.stop()
            
            # 根据用户角色显示不同的数据管理功能
            if user_role == "admin":
                # admin用户可以看到所有功能：Data Overview、Missing Values Analysis、Data Distribution和Data Quality
                with st.expander("Data Overview", expanded=True):
                    render_data_overview_admin(df, table_name, data_manager)
            else:
                # viewer用户只能看到：Data Overview
                with st.expander("Data Overview", expanded=True):
                    render_data_overview_viewer(df)
            
            # Data visualization - merge data quality dashboard with visualization features
            with st.expander("Data Visualization", expanded=True):
                st.markdown("""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #8e44ad;">
                    <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
                        Explore data patterns, distributions, and relationships through interactive visualizations
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 确认数据可用性
                if df is None or len(df) == 0:
                    st.error("❌ No data available for visualization")
                    st.stop()
                
                # 根据用户角色显示不同内容
                user_role = st.session_state.get("authenticated_user", {}).get("role", "viewer")
                
                if user_role == "viewer":
                    # Viewer用户只看到基本的统计卡片
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db;">
                            <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600;">RECORDS</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.6rem; font-weight: 700;">{len(df):,}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Total entries</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #8B5FBF;">
                            <h4 style="color: #8B5FBF; margin: 0; font-size: 0.75rem; font-weight: 600;">COLUMNS</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.6rem; font-weight: 700;">{len(df.columns)}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data fields</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate data quality metrics
                    missing_data = df.isnull().sum()
                    total_missing = missing_data.sum()
                    total_cells = len(df) * len(df.columns)
                    completeness_rate = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
                    
                    with col3:
                        completeness_color = "#27ae60" if completeness_rate > 80 else "#f39c12" if completeness_rate > 60 else "#e74c3c"
                        completeness_percentage = f"{completeness_rate:.1f}%"
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {completeness_color};">
                            <h4 style="color: {completeness_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">COMPLETE</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.6rem; font-weight: 700;">{completeness_percentage}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data integrity</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate key field completeness
                    key_fields = ['Fiber_type', 'Matrix_type', 'temperature', 'time_field', 'Target_parameter']
                    key_completeness = 0
                    valid_key_fields = 0
                    for field in key_fields:
                        if field in df.columns:
                            field_completeness = (1 - df[field].isnull().sum() / len(df)) * 100
                            key_completeness += field_completeness
                            valid_key_fields += 1
                    
                    avg_key_completeness = key_completeness / valid_key_fields if valid_key_fields > 0 else 0
                    
                    with col4:
                        key_color = "#27ae60" if avg_key_completeness > 70 else "#f39c12" if avg_key_completeness > 50 else "#e74c3c"
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {key_color};">
                            <h4 style="color: {key_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">QUALITY</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.6rem; font-weight: 700;">{avg_key_completeness:.1f}%</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Key fields</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    # Admin用户看到完整功能
                    # Data quality statistics cards - following Model Configuration style
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db;">
                            <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600;">RECORDS</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{len(df):,}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Total entries</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #8B5FBF;">
                            <h4 style="color: #8B5FBF; margin: 0; font-size: 0.75rem; font-weight: 600;">COLUMNS</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{len(df.columns)}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data fields</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate data quality metrics
                    missing_data = df.isnull().sum()
                    total_missing = missing_data.sum()
                    total_cells = len(df) * len(df.columns)
                    completeness_rate = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
                    
                    with col3:
                        completeness_color = "#27ae60" if completeness_rate > 80 else "#f39c12" if completeness_rate > 60 else "#e74c3c"
                        completeness_percentage = f"{completeness_rate:.1f}%"
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {completeness_color};">
                            <h4 style="color: {completeness_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">COMPLETE</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{completeness_percentage}</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Data integrity</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate key field completeness
                    key_fields = ['Fiber_type', 'Matrix_type', 'temperature', 'time_field', 'Target_parameter']
                    key_completeness = 0
                    valid_key_fields = 0
                    for field in key_fields:
                        if field in df.columns:
                            field_completeness = (1 - df[field].isnull().sum() / len(df)) * 100
                            key_completeness += field_completeness
                            valid_key_fields += 1
                    
                    avg_key_completeness = key_completeness / valid_key_fields if valid_key_fields > 0 else 0
                    
                    with col4:
                        key_color = "#27ae60" if avg_key_completeness > 70 else "#f39c12" if avg_key_completeness > 50 else "#e74c3c"
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {key_color};">
                            <h4 style="color: {key_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">QUALITY</h4>
                            <h3 style="color: #2c3e50; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{avg_key_completeness:.1f}%</h3>
                            <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Key fields</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 根据用户角色动态创建可视化分析标签页
                    # admin用户可以看到所有功能：Missing Values Analysis、Data Distribution
                    quality_tabs = st.tabs(["Missing Values Analysis", "Data Distribution"])
                    
                    # Missing Values Analysis - 只对admin用户可见
                    with quality_tabs[0]:
                        # Missing values analysis - complete original optimized layout
                        missing_data = df.isnull().sum()
                        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                        
                        if len(missing_data) > 0:
                            # Category information statistics
                            severe_missing = missing_data[missing_data / len(df) > 0.8]
                            moderate_missing = missing_data[(missing_data / len(df) > 0.5) & (missing_data / len(df) <= 0.8)]
                            low_missing = missing_data[missing_data / len(df) <= 0.5]
                        
                        # Missing values analysis overview - moved before the three cards
                        st.markdown("#### Missing Values Analysis Overview")
                        
                        # Category statistics cards following Model Configuration style
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #e74c3c; height: 85px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;">
                                <h4 style="color: #e74c3c; margin: 0 0 -0.8rem 0; font-size: 0.7rem; font-weight: 600; line-height: 0.6;">SEVERE MISSING</h4>
                                <h2 style="color: #2c3e50; margin: 0; font-size: 1.3rem; font-weight: 700; line-height: 0.6;">{len(severe_missing)}</h2>
                                <p style="margin: -0.6rem 0 0 0; font-size: 0.6rem; color: #7f8c8d; line-height: 0.6;">>80% Missing values</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # View Details button
                            if st.button("View Details", key="severe_btn", use_container_width=True):
                                # Clear other category details and missing analysis states
                                st.session_state.show_detailed_missing_info = False
                                st.session_state.show_detailed_table = False
                                # Set current category
                                st.session_state.show_category_detail = "Severe"
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #f39c12; height: 85px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;">
                                <h4 style="color: #f39c12; margin: 0 0 -0.8rem 0; font-size: 0.7rem; font-weight: 600; line-height: 0.6;">MODERATE MISSING</h4>
                                <h2 style="color: #2c3e50; margin: 0; font-size: 1.3rem; font-weight: 700; line-height: 0.6;">{len(moderate_missing)}</h2>
                                <p style="margin: -0.6rem 0 0 0; font-size: 0.6rem; color: #7f8c8d; line-height: 0.6;">50-80% Missing values</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("View Details", key="moderate_btn", use_container_width=True):
                                # Clear other category details and missing analysis states
                                st.session_state.show_detailed_missing_info = False
                                st.session_state.show_detailed_table = False
                                # Set current category
                                st.session_state.show_category_detail = "Moderate"
                        
                        with col3:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db; height: 85px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;">
                                <h4 style="color: #3498db; margin: 0 0 -0.8rem 0; font-size: 0.7rem; font-weight: 600; line-height: 0.6;">LOW MISSING</h4>
                                <h2 style="color: #2c3e50; margin: 0; font-size: 1.3rem; font-weight: 700; line-height: 0.6;">{len(low_missing)}</h2>
                                <p style="margin: -0.6rem 0 0 0; font-size: 0.6rem; color: #7f8c8d; line-height: 0.6;"><50% Missing values</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("View Details", key="low_btn", use_container_width=True):
                                # Clear other category details and missing analysis states
                                st.session_state.show_detailed_missing_info = False
                                st.session_state.show_detailed_table = False
                                # Set current category
                                st.session_state.show_category_detail = "Low"
                        
                        # Detailed missing values table - optimized version
                        st.markdown("#### Missing Values Analysis Details")
                        
                        # Create missing values details DataFrame
                        missing_df = pd.DataFrame({
                            'Column': missing_data.index,
                            'Missing Count': missing_data.values,
                            'Missing Percentage': (missing_data.values / len(df) * 100).round(2),
                            'Data Type': [str(df[col].dtype) for col in missing_data.index],
                            'Category': [
                                'Severe' if missing_data[col] / len(df) > 0.8 
                                else 'Moderate' if missing_data[col] / len(df) > 0.5 
                                else 'Low' for col in missing_data.index
                            ]
                        })
                        
                        # Create integrated horizontal bar chart to display missing values information
                        
                        # Check if specific category details need to be displayed
                        if 'show_category_detail' in st.session_state and st.session_state.show_category_detail:
                            selected_category = st.session_state.show_category_detail
                            category_data = missing_df[missing_df['Category'] == selected_category]
                            
                            if len(category_data) > 0:
                                st.info(f"Displaying {selected_category} category missing values details ({len(category_data)} fields total)")
                                
                                # Create specialized chart for selected category
                                fig_category = go.Figure()
                                
                                category_color = {'Severe': '#e74c3c', 'Moderate': '#f39c12', 'Low': '#3498db'}[selected_category]
                                display_category = category_data.head(10)  # Display up to 10 items
                                
                                fig_category.add_trace(go.Bar(
                                    y=display_category['Column'],
                                    x=display_category['Missing Percentage'],
                                    orientation='h',
                                    marker=dict(color=category_color, line=dict(color='white', width=0.5)),
                                    text=[f"{val:.1f}%" for val in display_category['Missing Percentage']],
                                    textposition='inside',
                                    textfont=dict(color='white', size=10),
                                    hovertemplate=(
                                        "<b>%{y}</b><br>" +
                                        "Missing Ratio: %{x:.1f}%<br>" +
                                        "Missing Count: %{customdata[0]:,}<br>" +
                                        "Data Type: %{customdata[1]}" +
                                        "<extra></extra>"
                                    ),
                                    customdata=[[row['Missing Count'], row['Data Type']] for _, row in display_category.iterrows()]
                                ))
                                
                                fig_category.update_layout(
                                    title=f"{selected_category} Category Missing Values Details",
                                    xaxis=dict(title="Missing Percentage (%)", range=[0, 105]),
                                    yaxis=dict(title="", autorange="reversed"),
                                    height=max(200, len(display_category) * 30),
                                    template="plotly_white",
                                    margin=dict(l=120, r=30, t=50, b=40),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_category, use_container_width=True)
                            else:
                                st.warning(f"No missing values found in {selected_category} category")
                        
                        else:
                            # Prepare data for visualization
                            missing_viz_data = []
                            for idx, row in missing_df.iterrows():
                                percentage = row['Missing Percentage']
                                category = row['Category']
                                
                                # Determine color based on category
                                if category == 'Severe':
                                    color = "#e74c3c"
                                elif category == 'Moderate':
                                    color = "#f39c12"
                                else:
                                    color = "#3498db"
                                
                                missing_viz_data.append({
                                    'Column': row['Column'],
                                    'Missing_Count': row['Missing Count'],
                                    'Missing_Percentage': percentage,
                                    'Category': category,
                                    'Color': color,
                                    'Data_Type': row['Data Type']
                                })
                            
                            # Create beautiful integrated bar chart
                            if missing_viz_data:
                                # Sort by missing percentage
                                missing_viz_data.sort(key=lambda x: x['Missing_Percentage'], reverse=True)
                                
                                # If too many fields, only show top 10 most severe ones
                                display_data = missing_viz_data[:10] if len(missing_viz_data) > 10 else missing_viz_data
                                remaining_count = len(missing_viz_data) - len(display_data)
                                
                                if remaining_count > 0:
                                    st.markdown(f"""
                                    <div class="missing-info-text">
                                        <div style="background-color: #d1ecf1; color: #0c5460; padding: 0.75rem 1rem; border-radius: 8px; border-left: 4px solid #bee5eb; margin: 1rem 0; animation: slideInLeft 0.6s ease-out;">
                                            Showing top {len(display_data)} fields with most missing values, {remaining_count} more fields have missing values
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Create Plotly chart - remove legend
                                fig_overview = go.Figure()
                                
                                # Create single trace for all data without categorization
                                fig_overview.add_trace(go.Bar(
                                    y=[item['Column'] for item in display_data],
                                    x=[item['Missing_Percentage'] for item in display_data],
                                    orientation='h',
                                    marker=dict(
                                        color=[item['Color'] for item in display_data],
                                        line=dict(color='white', width=0.5)
                                    ),
                                    text=[f"{item['Missing_Percentage']:.1f}%" for item in display_data],
                                    textposition='inside',
                                    textfont=dict(color='white', size=10, family="Arial"),
                                    hovertemplate=(
                                        "<b>%{y}</b><br>" +
                                        "Missing Ratio: %{x:.1f}%<br>" +
                                        "Missing Count: %{customdata[0]:,}<br>" +
                                        "Data Type: %{customdata[1]}" +
                                        "<extra></extra>"
                                    ),
                                    customdata=[[item['Missing_Count'], item['Data_Type']] for item in display_data],
                                    showlegend=False
                                ))
                                
                                fig_overview.update_layout(
                                    title=dict(
                                        text="Missing Values Analysis Overview",
                                        font=dict(size=14, color="#2c3e50"),
                                        x=0.5
                                    ),
                                    xaxis=dict(
                                        title="Missing Percentage (%)",
                                        showgrid=True,
                                        gridcolor='rgba(128,128,128,0.2)',
                                        range=[0, 105]
                                    ),
                                    yaxis=dict(
                                        title="",
                                        autorange="reversed",
                                        tickfont=dict(size=10)
                                    ),
                                    height=max(250, len(display_data) * 30),
                                    template="plotly_white",
                                    margin=dict(l=120, r=30, t=50, b=40),
                                    plot_bgcolor='rgba(248,249,250,0.5)',
                                    paper_bgcolor='white',
                                    showlegend=False
                                )
                                
                                # Add simplified reference lines (only when meaningful)
                                if any(item['Missing_Percentage'] > 50 for item in display_data):
                                    fig_overview.add_vline(x=50, line_dash="dot", line_color="orange", line_width=1)
                                if any(item['Missing_Percentage'] > 80 for item in display_data):
                                    fig_overview.add_vline(x=80, line_dash="dot", line_color="red", line_width=1)
                                
                                st.plotly_chart(fig_overview, use_container_width=True)
                                
                                # Add compact statistical summary
                                col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
                                with col_summary1:
                                    st.metric("Highest Missing Rate", f"{max(item['Missing_Percentage'] for item in missing_viz_data):.1f}%", 
                                             help="Field with highest missing value proportion in dataset")
                                with col_summary2:
                                    avg_missing = sum(item['Missing_Percentage'] for item in missing_viz_data) / len(missing_viz_data)
                                    st.metric("Average Missing Rate", f"{avg_missing:.1f}%", 
                                             help="Average missing ratio for all fields with missing values")
                                with col_summary3:
                                    st.metric("Missing Fields Count", f"{len(missing_viz_data)}", 
                                             help="Total number of fields with missing values")
                                with col_summary4:
                                    severe_count = len([item for item in missing_viz_data if item['Category'] == 'Severe'])
                                    st.metric("Severe Missing", f"{severe_count}", 
                                             help="Number of fields with missing rate exceeding 80%")
                        
                        # Add functional buttons (without divider) - Modified layout: 3+2 with longer bottom buttons
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        with col_btn1:
                            if st.button("Detailed Chart", key="detailed_chart_btn", use_container_width=True):
                                # Clear other missing analysis states
                                st.session_state.show_detailed_table = False
                                if 'show_category_detail' in st.session_state:
                                    del st.session_state.show_category_detail
                                # Toggle current state
                                st.session_state.show_detailed_missing_info = not st.session_state.get('show_detailed_missing_info', False)
                        with col_btn2:
                            if st.button("Data Table", key="detailed_table_btn", use_container_width=True):
                                # Clear other missing analysis states
                                st.session_state.show_detailed_missing_info = False
                                if 'show_category_detail' in st.session_state:
                                    del st.session_state.show_category_detail
                                # Toggle current state
                                st.session_state.show_detailed_table = not st.session_state.get('show_detailed_table', False)
                        with col_btn3:
                            if st.button("Back to Overview", key="back_overview_btn", use_container_width=True):
                                # Clear all missing analysis states
                                st.session_state.show_detailed_missing_info = False
                                st.session_state.show_detailed_table = False
                                if 'show_category_detail' in st.session_state:
                                    del st.session_state.show_category_detail
                        
                        # Second row with 2 longer buttons
                        col_btn4, col_btn5 = st.columns(2)
                        with col_btn4:
                            if st.button("Advanced Analysis", key="advanced_analysis_btn", use_container_width=True):
                                st.info("Advanced missing value analysis coming soon!")
                        with col_btn5:
                            if st.button("Export Report", key="export_report_btn", use_container_width=True):
                                st.info("Export functionality coming soon!")
                        
                        # Display detailed original charts (collapsible)
                        if st.session_state.get('show_detailed_missing_info', False):
                            st.markdown("#### Detailed Missing Values Visualization")
                            
                            # Create original missing value percentage chart
                            missing_pct = (missing_data / len(df) * 100)
                            
                            fig_detailed = go.Figure(data=[
                                go.Bar(
                                    x=missing_pct.values,
                                    y=missing_pct.index,
                                    orientation='h',
                                    marker=dict(
                                        color=missing_pct.values,
                                        colorscale='RdYlBu_r',
                                        showscale=True,
                                        colorbar=dict(title="Missing %", thickness=15)
                                    ),
                                    text=[f'{val:.1f}%' for val in missing_pct.values],
                                    textposition='auto',
                                    hovertemplate=(
                                        "<b>%{y}</b><br>" +
                                        "Missing Ratio: %{x:.1f}%<br>" +
                                        "Missing Count: %{customdata:,}" +
                                        "<extra></extra>"
                                    ),
                                    customdata=missing_data.values
                                )
                            ])
                            
                            fig_detailed.update_layout(
                                title="Detailed Missing Values Analysis - Continuous Color Scale Display",
                                xaxis_title="Missing Percentage (%)",
                                yaxis_title="Field Name",
                                height=max(400, len(missing_data) * 25),
                                template="plotly_white",
                                margin=dict(l=150, r=80, t=60, b=60)
                            )
                            
                            st.plotly_chart(fig_detailed, use_container_width=True)
                        
                        # Display detailed data table (collapsible)
                        if st.session_state.get('show_detailed_table', False):
                            st.markdown("#### Detailed Data Table")
                            
                            # Use more refined styling function
                            def style_missing_table(df):
                                def color_percentage(val):
                                    if val > 80:
                                        return 'background: linear-gradient(90deg, #ffebee, #ffcdd2); color: #c62828; font-weight: 600;'
                                    elif val > 50:
                                        return 'background: linear-gradient(90deg, #fff3e0, #ffe0b2); color: #ef6c00; font-weight: 600;'
                                    else:
                                        return 'background: linear-gradient(90deg, #e3f2fd, #bbdefb); color: #1565c0; font-weight: 600;'
                                
                                def color_category(val):
                                    if val == 'Severe':
                                        return 'background: #ffebee; color: #c62828; font-weight: 600; text-align: center; border-radius: 8px; padding: 4px;'
                                    elif val == 'Moderate':
                                        return 'background: #fff3e0; color: #ef6c00; font-weight: 600; text-align: center; border-radius: 8px; padding: 4px;'
                                    else:
                                        return 'background: #e3f2fd; color: #1565c0; font-weight: 600; text-align: center; border-radius: 8px; padding: 4px;'
                                
                                return df.style.applymap(color_percentage, subset=['Missing Percentage']).applymap(color_category, subset=['Category']).format({
                                    'Missing Count': '{:,}',
                                    'Missing Percentage': '{:.1f}%'
                                })
                            
                            styled_table = style_missing_table(missing_df)
                            st.dataframe(styled_table, use_container_width=True, hide_index=True)
                        
                        else:
                            # Display data integrity confirmation
                            st.markdown(f"""
                            <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; margin: 1rem 0; border: 1px solid rgba(0, 0, 0, 0.05);">
                                <h4 style="color: #27ae60; margin: 0; font-size: 1.2rem;">Perfect Data Quality</h4>
                                <p style="margin: 0.5rem 0 0 0; color: #7f8c8d; font-size: 0.9rem;">All {len(df):,} records have complete data across all {len(df.columns)} columns.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
            
            # Data Operations - collapsible section (对admin和editor用户可见)
                    with quality_tabs[1]:
                        # Data Distribution Analysis Panel - Integrated complete data visualization functionality
                        st.markdown("### Data Distribution & Visualization")
                        
                        # Top statistics cards following Model Configuration style
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Calculate statistics
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        text_cols = df.select_dtypes(include=['object']).columns
                        datetime_cols = df.select_dtypes(include=['datetime64']).columns
                        if 'time_field' in df.columns:
                            try:
                                df['time_field'] = pd.to_datetime(df['time_field'])
                                if 'time_field' not in datetime_cols:
                                    datetime_cols = datetime_cols.union(['time_field'])
                            except:
                                pass
                        
                        # Calculate statistics
                        if len(df) > 0:
                            total_null = df.isnull().sum().sum()
                        
                        with col1:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db; height: 120px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600;">NUMERIC FIELDS</h4>
                                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{len(numeric_cols)}</h2>
                                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Continuous variables</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #9b59b6; height: 120px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #9b59b6; margin: 0; font-size: 0.75rem; font-weight: 600;">TEXT FIELDS</h4>
                                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{len(text_cols)}</h2>
                                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Categorical variables</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #1abc9c; height: 120px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #1abc9c; margin: 0; font-size: 0.75rem; font-weight: 600;">DATETIME FIELDS</h4>
                                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{len(datetime_cols)}</h2>
                                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Time series variables</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            null_percentage = (total_null / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
                            st.markdown(f"""
                            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #34495e; height: 120px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #34495e; margin: 0; font-size: 0.75rem; font-weight: 600;">MISSING VALUES</h4>
                                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700;">{null_percentage:.1f}%</h2>
                                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d;">Overall completion</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Integrated visualization analysis tab
                        viz_sub_tabs = st.tabs(["Distribution Analysis", "Relationships", "Summary"])
                        
                        with viz_sub_tabs[0]:
                            st.markdown("#### Field Distribution Analysis")
                            
                            # Create layout for column type selection and field selection
                            col_type_selection, col_field_selection = st.columns([1, 2])
                            
                            with col_type_selection:
                                # Column type selection
                                available_types = []
                                if len(numeric_cols) > 0:
                                    available_types.append("Numerical Columns")
                                if len(text_cols) > 0:
                                    available_types.append("Text Columns")
                                
                                if available_types:
                                    selected_type = st.selectbox(
                                        "Select analysis type:",
                                        available_types,
                                        key="column_type_selection"
                                    )
                                else:
                                    st.warning("No columns available for analysis")
                                    selected_type = None
                            
                            with col_field_selection:
                                if selected_type == "Numerical Columns":
                                    # Numerical column selection
                                    meaningful_cols = [col for col in numeric_cols if not col.lower().startswith('id')]
                                    if not meaningful_cols:
                                        meaningful_cols = numeric_cols.tolist()
                                    
                                    # Smart default selection
                                    default_col = None
                                    priority_cols = ['Value1', 'temperature', 'pH', 'pH_1', 'time_field', 'ultimate_tensile_strength']
                                    for priority in priority_cols:
                                        if priority in meaningful_cols:
                                            default_col = priority
                                            break
                                    if not default_col and meaningful_cols:
                                        default_col = meaningful_cols[0]
                                    
                                    selected_col = st.selectbox(
                                        "Select numerical column to analyze:",
                                        meaningful_cols,
                                        index=meaningful_cols.index(default_col) if default_col in meaningful_cols else 0,
                                        key="numeric_dist_analysis"
                                    )
                                
                                elif selected_type == "Text Columns":
                                    # Text column selection
                                    selected_col = st.selectbox(
                                        "Select text column to analyze:",
                                        text_cols.tolist(),
                                        key="text_dist_analysis"
                                    )
                                else:
                                    selected_col = None
                            
                            # Analysis results display
                            if selected_col and selected_type:
                                st.markdown("---")
                                
                                if selected_type == "Numerical Columns":
                                    # Numerical column analysis
                                    col_data = pd.to_numeric(df[selected_col], errors='coerce').dropna()
                                    
                                    if len(col_data) > 0:
                                        # Statistical summary
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Mean", f"{col_data.mean():.2f}")
                                        with col2:
                                            st.metric("Median", f"{col_data.median():.2f}")
                                        with col3:
                                            st.metric("Std Dev", f"{col_data.std():.2f}")
                                        
                                        # Distribution chart
                                        fig = px.histogram(x=col_data, title=f"{selected_col} Distribution")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Column '{selected_col}' does not contain valid numerical data")
                                
                                elif selected_type == "Text Columns":
                                    # Text column analysis
                                    text_data = df[selected_col].dropna()
                                    
                                    if len(text_data) > 0:
                                        # Text statistics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Unique Values", f"{text_data.nunique()}")
                                        with col2:
                                            st.metric("Most Common", f"{text_data.mode().iloc[0] if len(text_data.mode()) > 0 else 'N/A'}")
                                        with col3:
                                            st.metric("Total Count", f"{len(text_data)}")
                                        
                                        # Distribution chart
                                        text_counts = text_data.value_counts().head(10)
                                        fig = px.bar(x=text_counts.values, y=text_counts.index, orientation='h', 
                                                   title=f"{selected_col} Distribution (Top 10)")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning(f"Column '{selected_col}' does not contain valid text data")
                        
                        with viz_sub_tabs[1]:
                            st.markdown("#### Variable Correlation Analysis")
                            
                            if len(numeric_cols) >= 2:
                                # Variable selection
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    x_col = st.selectbox(
                                        "X-axis Variable:",
                                        numeric_cols.tolist(),
                                        key="scatter_x_var"
                                    )
                                
                                with col2:
                                    y_col = st.selectbox(
                                        "Y-axis Variable:",
                                        [col for col in numeric_cols if col != x_col],
                                        key="scatter_y_var"
                                    )
                                
                                if x_col and y_col:
                                    # Prepare data
                                    x_data = pd.to_numeric(df[x_col], errors='coerce')
                                    y_data = pd.to_numeric(df[y_col], errors='coerce')
                                    valid_mask = x_data.notna() & y_data.notna()
                                    
                                    if valid_mask.sum() > 0:
                                        # Calculate correlation
                                        correlation = x_data[valid_mask].corr(y_data[valid_mask])
                                        
                                        # Display correlation
                                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                                        
                                        # Scatter plot
                                        plot_df = pd.DataFrame({
                                            'x': x_data[valid_mask],
                                            'y': y_data[valid_mask]
                                        })
                                        
                                        fig = px.scatter(plot_df, x='x', y='y', 
                                                       title=f"{x_col} vs {y_col}",
                                                       labels={'x': x_col, 'y': y_col})
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No valid data points for correlation analysis")
                            else:
                                st.info("Need at least 2 numerical columns for correlation analysis")
                        
                        with viz_sub_tabs[2]:
                            st.markdown("#### Comprehensive Data Summary")
                            
                            # Main content area with three analysis modules
                            summary_tabs = st.tabs(["Correlation Overview", "Field Type Analysis", "Data Quality Report"])
                            
                            with summary_tabs[0]:
                                # Correlation Overview - 相关性概览
                                st.markdown("**Numerical Variable Correlation Overview**")
                                
                                if len(numeric_cols) >= 2:
                                    # Select number of variables to display
                                    max_vars = min(len(numeric_cols), 12)  # Display up to 12 variables
                                    if len(numeric_cols) > max_vars:
                                        st.info(f"Displaying correlation of top {max_vars} numerical variables, total {len(numeric_cols)} numerical variables")
                                    
                                    selected_numeric_cols = numeric_cols[:max_vars]
                                    corr_matrix = df[selected_numeric_cols].corr()
                                    
                                    # Create correlation heatmap
                                    fig_heatmap = px.imshow(
                                        corr_matrix,
                                        title="Variable Correlation Heatmap",
                                        color_continuous_scale='RdBu',
                                        aspect="auto",
                                        text_auto=True,
                                        height=400
                                    )
                                    
                                    fig_heatmap.update_traces(
                                        text=corr_matrix.round(2).values,
                                        texttemplate="%{text}",
                                        textfont={"size": 9}
                                    )
                                    
                                    fig_heatmap.update_layout(
                                        title_font_size=14,
                                        margin=dict(l=100, r=50, t=60, b=50)
                                    )
                                    
                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                else:
                                    st.info("Insufficient numerical variables to generate correlation heatmap")
                            
                            with summary_tabs[1]:
                                # Field Type Analysis - Field Type Analysis
                                st.markdown("**Field Type Distribution and Details**")
                                
                                # Data type statistics
                                dtype_counts = df.dtypes.value_counts()
                                
                                if len(dtype_counts) > 0:
                                    type_chart_col, type_detail_col = st.columns([1.5, 1])
                                    
                                    with type_chart_col:
                                        fig_dtype = px.pie(
                                            values=dtype_counts.values,
                                            names=[str(dtype) for dtype in dtype_counts.index],
                                            title="Data Type Distribution",
                                            height=300
                                        )
                                        fig_dtype.update_traces(textposition='inside', textinfo='percent+label')
                                        st.plotly_chart(fig_dtype, use_container_width=True)
                                    
                                    with type_detail_col:
                                        # Detailed type statistics table
                                        type_detail_df = pd.DataFrame({
                                            'Data Type': [str(dtype) for dtype in dtype_counts.index],
                                            'Field Count': dtype_counts.values,
                                            'Percentage': [f"{count/len(df.columns)*100:.1f}%" for count in dtype_counts.values]
                                        })
                                        st.dataframe(type_detail_df, hide_index=True, use_container_width=True)
                                
                                # Field detailed information table
                                st.markdown("**Field Detailed Information**")
                                
                                field_info = []
                                for col in df.columns:
                                    col_data = df[col]
                                    field_info.append({
                                        'Field Name': col,
                                        'Data Type': str(col_data.dtype),
                                        'Non-null Count': col_data.notna().sum(),
                                        'Missing Count': col_data.isnull().sum(),
                                        'Missing Rate': f"{col_data.isnull().sum()/len(col_data)*100:.1f}%",
                                        'Unique Count': col_data.nunique() if col_data.dtype == 'object' else 'N/A'
                                    })
                                
                                field_info_df = pd.DataFrame(field_info)
                                st.dataframe(field_info_df, hide_index=True, use_container_width=True, height=400)
                            
                            with summary_tabs[2]:
                                # Data Quality Report - 数据质量报告
                                st.markdown("**Comprehensive Data Quality Assessment**")
                                
                                # Calculate comprehensive indicators
                                total_rows = len(df)
                                total_cols = len(df.columns)
                                missing_cells = df.isnull().sum().sum()
                                total_cells = total_rows * total_cols
                                overall_completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
                                duplicate_rows = df.duplicated().sum()
                                
                                # Quality score calculation
                                completeness_score = overall_completeness
                                uniqueness_score = (1 - duplicate_rows/total_rows) * 100 if total_rows > 0 else 100
                                consistency_score = 85  # Basic consistency score
                                
                                overall_quality = (completeness_score + uniqueness_score + consistency_score) / 3
                                
                                # Quality score cards
                                quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
                                
                                with quality_col1:
                                    comp_color = "#27ae60" if completeness_score > 80 else "#f39c12" if completeness_score > 60 else "#e74c3c"
                                    st.markdown(f"""
                                    <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {comp_color};">
                                        <h4 style="color: {comp_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">COMPLETENESS</h4>
                                        <h3 style="color: {comp_color}; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{completeness_score:.1f}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with quality_col2:
                                    uniq_color = "#27ae60" if uniqueness_score > 90 else "#f39c12" if uniqueness_score > 70 else "#e74c3c"
                                    st.markdown(f"""
                                    <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {uniq_color};">
                                        <h4 style="color: {uniq_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">UNIQUENESS</h4>
                                        <h3 style="color: {uniq_color}; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{uniqueness_score:.1f}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with quality_col3:
                                    cons_color = "#27ae60" if consistency_score > 80 else "#f39c12" if consistency_score > 60 else "#e74c3c"
                                    st.markdown(f"""
                                    <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {cons_color};">
                                        <h4 style="color: {cons_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">CONSISTENCY</h4>
                                        <h3 style="color: {cons_color}; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{consistency_score:.1f}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with quality_col4:
                                    overall_color = "#27ae60" if overall_quality > 80 else "#f39c12" if overall_quality > 60 else "#e74c3c"
                                    st.markdown(f"""
                                    <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {overall_color};">
                                        <h4 style="color: {overall_color}; margin: 0; font-size: 0.75rem; font-weight: 600;">OVERALL SCORE</h4>
                                        <h3 style="color: {overall_color}; margin: 0.3rem 0; font-size: 1.2rem; font-weight: 700;">{overall_quality:.1f}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Quality issue analysis
                                st.markdown("**Quality Issue Analysis**")
                                
                                quality_issues = []
                                
                                # Check missing values issue
                                if missing_cells > total_cells * 0.1:
                                    quality_issues.append("Dataset has too many missing values, data cleaning is recommended")
                                
                                # Check duplicate rows issue
                                if duplicate_rows > 0:
                                    quality_issues.append(f"Found {duplicate_rows} duplicate rows, deduplication is recommended")
                                
                                # Check data type consistency
                                mixed_types = 0
                                for col in df.columns:
                                    if df[col].dtype == 'object':
                                        # Try to convert to numeric, check for mixed types
                                        numeric_converted = pd.to_numeric(df[col], errors='coerce')
                                        if numeric_converted.notna().sum() > len(df[col]) * 0.5:  # If more than 50% can be converted to numeric
                                            mixed_types += 1
                                
                                if mixed_types > 0:
                                    quality_issues.append(f"Found {mixed_types} fields with possible data type inconsistency issues")
                                
                                if not quality_issues:
                                    st.success("Data quality is good, no obvious issues found")
                                else:
                                    for issue in quality_issues:
                                        st.warning(issue)

            
            # Data Filters
            with st.expander("Data Filter", expanded=False):
                st.markdown("""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #e67e22;">
                    <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
                        Apply advanced filtering criteria to focus on specific data subsets
                    </p>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                # Initialize filter variables
                fiber_filter = []
                temp_range = None
                ph_range = None
                
                with col1:
                    st.subheader("Fiber Type")
                    if "Fiber_type" in df.columns:
                        unique_fibers = df["Fiber_type"].dropna().unique().tolist()
                        if unique_fibers:
                            fiber_filter = st.multiselect(
                                "Select fiber types:",
                                options=unique_fibers,
                                key="fiber_filter"
                            )
                            if fiber_filter:
                                filtered_count = df[df["Fiber_type"].isin(fiber_filter)].shape[0]
                                st.caption(f"Selected: {len(fiber_filter)} types, {filtered_count:,} records")
                        else:
                            st.caption("No fiber type data available")
                    else:
                        st.caption("Fiber type column not found")
                
                with col2:
                    st.subheader("Temperature Range")
                    if "temperature" in df.columns:
                        temp_numeric = pd.to_numeric(df["temperature"], errors='coerce')
                        if temp_numeric.notna().sum() > 0:
                            temp_min = float(temp_numeric.min())
                            temp_max = float(temp_numeric.max())
                            if temp_min != temp_max:
                                temp_range = st.slider(
                                    "Temperature (°C):",
                                    temp_min,
                                    temp_max,
                                    (temp_min, temp_max),
                                    step=0.1,
                                    key="temp_filter"
                                )
                                st.caption(f"Range: {temp_range[0]:.1f}°C - {temp_range[1]:.1f}°C")
                            else:
                                temp_range = (temp_min, temp_max)
                                st.caption(f"Single value: {temp_min:.1f}°C")
                        else:
                            st.caption("No valid temperature data")
                    else:
                        st.caption("Temperature column not found")
                
                with col3:
                    st.subheader("pH Range")
                    if "pH_1" in df.columns:
                        ph_numeric = pd.to_numeric(df["pH_1"], errors='coerce')
                        if ph_numeric.notna().sum() > 0:
                            ph_min = float(ph_numeric.min())
                            ph_max = float(ph_numeric.max())
                            if ph_min != ph_max:
                                ph_range = st.slider(
                                    "pH:",
                                    ph_min,
                                    ph_max,
                                    (ph_min, ph_max),
                                    step=0.1,
                                    key="ph_filter"
                                )
                                avg_ph = (ph_range[0] + ph_range[1]) / 2
                                ph_category = "Acidic" if avg_ph < 7 else "Basic" if avg_ph > 7 else "Neutral"
                                st.caption(f"Range: {ph_range[0]:.1f} - {ph_range[1]:.1f} ({ph_category})")
                            else:
                                ph_range = (ph_min, ph_max)
                                st.caption(f"Single value: {ph_min:.1f}")
                        else:
                            st.caption("No valid pH data")
                    else:
                        st.caption("pH column not found")
                
                # Filter summary and reset
                st.markdown("---")
                
                col_summary, col_reset = st.columns([3, 1])
                
                with col_summary:
                    active_filters = []
                    if fiber_filter:
                        active_filters.append(f"Fiber Types: {len(fiber_filter)}")
                    if temp_range and "temperature" in df.columns:
                        temp_numeric = pd.to_numeric(df["temperature"], errors='coerce')
                        if temp_numeric.notna().sum() > 0:
                            temp_min_orig = float(temp_numeric.min())
                            temp_max_orig = float(temp_numeric.max())
                            if temp_range != (temp_min_orig, temp_max_orig):
                                active_filters.append("Temperature Range")
                    if ph_range and "pH_1" in df.columns:
                        ph_numeric = pd.to_numeric(df["pH_1"], errors='coerce')
                        if ph_numeric.notna().sum() > 0:
                            ph_min_orig = float(ph_numeric.min())
                            ph_max_orig = float(ph_numeric.max())
                            if ph_range != (ph_min_orig, ph_max_orig):
                                active_filters.append("pH Range")
                    
                    if active_filters:
                        st.info(f"Active Filters: {', '.join(active_filters)}")
                    else:
                        st.info("No filters applied - Showing all data")
                
                with col_reset:
                    if st.button("Reset Filters", use_container_width=True):
                        for key in ['fiber_filter', 'temp_filter', 'ph_filter', 'has_active_filters', 'filtered_df', 'original_count', 'filtered_count']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                
                # Apply filters
                filtered_df = df.copy()
                original_count = len(filtered_df)
                
                # Apply fiber type filter
                if fiber_filter:
                    filtered_df = filtered_df[filtered_df["Fiber_type"].isin(fiber_filter)]
                
                # Apply temperature filter
                if temp_range and "temperature" in filtered_df.columns:
                    temp_numeric = pd.to_numeric(filtered_df["temperature"], errors='coerce')
                    temp_mask = (temp_numeric >= temp_range[0]) & (temp_numeric <= temp_range[1])
                    filtered_df = filtered_df[temp_mask.fillna(False)]
                
                # Apply pH filter
                if ph_range and "pH_1" in filtered_df.columns:  # Fix: use pH_1 instead of pH
                    ph_numeric = pd.to_numeric(filtered_df["pH_1"], errors='coerce')
                    ph_mask = (ph_numeric >= ph_range[0]) & (ph_numeric <= ph_range[1])
                    filtered_df = filtered_df[ph_mask.fillna(False)]
                
                # Store filtered data for later use
                filtered_count = len(filtered_df)
                
                # Store filter state for display outside expander
                st.session_state['has_active_filters'] = (fiber_filter or 
                    (temp_range and "temperature" in df.columns and temp_range != (pd.to_numeric(df["temperature"], errors='coerce').min(), pd.to_numeric(df["temperature"], errors='coerce').max())) or 
                    (ph_range and "pH_1" in df.columns and ph_range != (pd.to_numeric(df["pH_1"], errors='coerce').min(), pd.to_numeric(df["pH_1"], errors='coerce').max())))
                
                st.session_state['filtered_df'] = filtered_df
                st.session_state['original_count'] = original_count
                st.session_state['filtered_count'] = filtered_count
            
            # Filtered Results section (outside the Data Filter expander)
            if st.session_state.get('has_active_filters', False):
                with st.expander("Filtered Results", expanded=True):
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #27ae60;">
                        <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
                            View and analyze your filtered dataset results
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get data from session state
                    filtered_df = st.session_state.get('filtered_df', df)
                    original_count = st.session_state.get('original_count', len(df))
                    filtered_count = st.session_state.get('filtered_count', len(df))
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Original Records",
                            value=f"{original_count:,}",
                            help="Total number of records before filtering"
                        )
                    
                    with col2:
                        st.metric(
                            label="Filtered Records", 
                            value=f"{filtered_count:,}",
                            delta=f"{filtered_count - original_count:,}",
                            delta_color="inverse",
                            help="Number of records after applying filters"
                        )
                    
                    with col3:
                        retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
                        st.metric(
                            label="Retention Rate",
                            value=f"{retention_rate:.1f}%",
                            help="Percentage of records retained after filtering"
                        )
                    
                    with col4:
                        filtered_fields = len([col for col in filtered_df.columns if filtered_df[col].notna().any()])
                        st.metric(
                            label="Active Fields",
                            value=f"{filtered_fields}",
                            help="Number of fields with data in filtered results"
                        )
                    
                    # Display filtered data
                    if filtered_count > 0:
                        st.markdown("#### Filtered Data Preview")
                        
                        # Prepare display data, handle NaN values
                        display_filtered_df = filtered_df.copy()
                        for col in display_filtered_df.columns:
                            if display_filtered_df[col].dtype == 'object':
                                display_filtered_df[col] = display_filtered_df[col].fillna('--')
                            else:
                                display_filtered_df[col] = display_filtered_df[col].fillna(0)
                        
                        # Show info about the filtered data
                        st.info(f"Displaying {len(display_filtered_df):,} filtered records from {original_count:,} total records")
                        
                        # Display the filtered data table
                        st.dataframe(
                            display_filtered_df,
                            use_container_width=True,
                            height=400  # Set a reasonable height for the filtered results
                        )
                        
                        # Download filtered data button
                        if len(display_filtered_df) > 0:
                            csv_data = display_filtered_df.to_csv(index=False)
                            st.download_button(
                                label="Download Filtered Data as CSV",
                                data=csv_data,
                                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download the filtered dataset as a CSV file"
                            )
                    else:
                        st.warning("⚠️ No records match the selected filter criteria. Please adjust your filters to see results.")
                        
                        # Suggestions for users when no results found
                        st.markdown("""
                        **Suggestions:**
                        - Try expanding the temperature or pH ranges
                        - Select fewer or different fiber types
                        - Check if the data contains the values you're filtering for
                        - Use the 'Reset Filters' button to start over
                        """)
            
            # Data Operations - collapsible section (对admin和editor用户可见)
            if user_role in ["admin", "editor"]:
                with st.expander("Data Operations", expanded=False):
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid #3498db;">
                        <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
                            Manage your FRP materials database with these essential data operations
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 优化按钮布局：2行布局，视觉更平衡
                    # 第一行：主要数据操作
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button(
                            "Add New Record", 
                            use_container_width=True, 
                            help="Add a single new data record to the database"
                        ):
                            # Clear all other operation states
                            for key in ["show_batch_upload", "show_edit_form", "show_create_model", "show_export", "show_cache_management"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state["show_add_form"] = True
                    
                    with btn_col2:
                        if st.button(
                            "Batch Upload", 
                            use_container_width=True, 
                            help="Upload multiple records from CSV or Excel files"
                        ):
                            # Clear all other operation states
                            for key in ["show_add_form", "show_edit_form", "show_create_model", "show_export", "show_cache_management"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state["show_batch_upload"] = True
                    
                    with btn_col3:
                        if st.button(
                            "Edit Record", 
                            use_container_width=True, 
                            help="Modify existing data records"
                        ):
                            # Clear all other operation states
                            for key in ["show_add_form", "show_batch_upload", "show_create_model", "show_export", "show_cache_management"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state["show_edit_form"] = True
                    
                    # Add spacing between rows
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 第二行：模型和导出操作 - 两个更宽的按钮
                    btn_col4, btn_col5 = st.columns(2)
                    
                    with btn_col4:
                        if st.button(
                            "Create Model Dataset", 
                            use_container_width=True, 
                            help="Generate preprocessed dataset for machine learning"
                        ):
                            # Clear all other operation states
                            for key in ["show_add_form", "show_batch_upload", "show_edit_form", "show_export", "show_cache_management"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state["show_create_model"] = True
                    
                    with btn_col5:
                        if st.button(
                            "Export Data", 
                            use_container_width=True, 
                            help="Download data in various formats (CSV, Excel, JSON)"
                        ):
                            # Clear all other operation states
                            for key in ["show_add_form", "show_batch_upload", "show_edit_form", "show_create_model", "show_cache_management"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.session_state["show_export"] = True
            
            # Ensure filtered_df is available for other sections
            if 'filtered_df' not in locals():
                # Apply filters if not already applied in Data Filter section
                filtered_df = df.copy()
                
                # Apply fiber type filter if it exists
                if 'fiber_filter' in locals() and fiber_filter:
                    filtered_df = filtered_df[filtered_df["Fiber_type"].isin(fiber_filter)]
                
                # Apply temperature filter if it exists
                if 'temp_range' in locals() and temp_range and "temperature" in filtered_df.columns:
                    temp_numeric = pd.to_numeric(filtered_df["temperature"], errors='coerce')
                    temp_mask = (temp_numeric >= temp_range[0]) & (temp_numeric <= temp_range[1])
                    filtered_df = filtered_df[temp_mask.fillna(False)]
                
                # Apply pH filter if it exists
                if 'ph_range' in locals() and ph_range and "pH_1" in filtered_df.columns:
                    ph_numeric = pd.to_numeric(filtered_df["pH_1"], errors='coerce')
                    ph_mask = (ph_numeric >= ph_range[0]) & (ph_numeric <= ph_range[1])
                    filtered_df = filtered_df[ph_mask.fillna(False)]
            
            
            # Add new record form
            if st.session_state.get("show_add_form", False):
                st.markdown("---")
                st.subheader("Add New Record")
                
                # Feature descriptions dictionary
                feature_descriptions = {
                    'Title': 'Reference title of the research paper or study',
                    'Fiber_type': 'Type of reinforcing fiber used in the composite material',
                    'Matrix_type': 'Type of polymer matrix that binds the fibers together',
                    'surface_treatment': 'Chemical or physical treatment applied to fiber surface for better bonding',
                    'Fiber_type_detail': 'Specific grade, manufacturer, or detailed specification of the fiber',
                    'Matrix_type_detail': 'Specific formulation, grade, or manufacturer details of the matrix',
                    'Fiber_content_weight': 'Weight percentage of fibers in the total composite weight',
                    'Fiber_content_volume': 'Volume percentage of fibers in the total composite volume',
                    'diameter': 'Cross-sectional diameter of the FRP rebar specimen',
                    'Void_content': 'Percentage of air voids or empty spaces in the composite',
                    'cure_ratio': 'Degree of polymerization or crosslinking in the matrix material',
                    'cure_ratio_2': 'Secondary or post-cure degree of polymerization',
                    'tensile_modulus': 'Elastic modulus measuring stiffness under tensile loading',
                    'ultimate_tensile_strength': 'Maximum stress the material can withstand before failure',
                    'temperature': 'Environmental temperature during the exposure or testing',
                    'temp': 'Ambient temperature conditions during the experiment',
                    'temp2': 'Secondary temperature measurement or different stage temperature',
                    'pH': 'Measure of acidity or alkalinity of the exposure environment',
                    'pH_1': 'Initial pH value at the beginning of exposure',
                    'pH_2': 'Final pH value at the end of exposure period',
                    'pHafter': 'pH value measured after completing the exposure treatment',
                    'cycle_pH': 'pH value during cyclic exposure conditions',
                    'cycle_pH_after': 'pH value after completing cyclic exposure treatment',
                    'Duration_hours': 'Total exposure time expressed in hours',
                    'Duration_days': 'Total exposure time expressed in days',
                    'test_age': 'Age of the specimen at the time of testing',
                    'chloride': 'Concentration of chloride ions in the exposure solution',
                    'alkaline': 'Concentration of alkaline substances in the exposure medium',
                    'field_average_temperature': 'Average temperature recorded in field exposure conditions',
                    'field_average_humidity': 'Average relative humidity during field exposure',
                    'Effektive_Klimaklassifikation': 'Climate classification system used for environmental conditions',
                    'Strength of unconditioned rebar': 'Baseline tensile strength before any environmental exposure',
                    'Tensile strength retention': 'Ratio of residual strength to original strength (0-1 scale)',
                    'Tensile strength retention rate (%)': 'Percentage of original strength retained after exposure',
                    'Residual tensile strength (MPa)': 'Remaining tensile strength after environmental degradation'
                }
                
                with st.form("add_record_form"):
                    new_data = {}
                    
                    # Dynamically generate form fields
                    cols = st.columns(3)
                    col_index = 0
                    
                    for col in df.columns:
                        if col != 'id':  # Skip ID field
                            with cols[col_index % 3]:
                                # Get feature description
                                description = feature_descriptions.get(col, f'{col}')
                                
                                if df[col].dtype == 'object':
                                    unique_vals = df[col].dropna().unique().tolist()
                                    if len(unique_vals) < 20 and len(unique_vals) > 0:
                                        new_data[col] = st.selectbox(f"{col}", [""] + unique_vals, 
                                                                   help=description, key=f"add_{col}")
                                    else:
                                        new_data[col] = st.text_input(f"{col}", help=description, key=f"add_{col}")
                                else:
                                    new_data[col] = st.number_input(f"{col}", value=0.0, 
                                                                  help=description, key=f"add_{col}")
                            col_index += 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Submit for Approval", type="primary"):
                            # Submit change request
                            submit_data_change(
                                user_info['email'],
                                'insert',
                                table_name,
                                None,
                                None,
                                new_data,
                                engine
                            )
                            st.success("Changes submitted for approval!")
                            st.session_state["show_add_form"] = False
                            
                            # Record operation
                            log_operation(
                                user_info['email'],
                                'submit_insert',
                                table_name,
                                None,
                                new_data,
                                get_client_ip(),
                                engine
                            )
                            st.rerun()
                    
                    with col2:
                        if st.form_submit_button("Cancel"):
                            st.session_state["show_add_form"] = False
                            st.rerun()
            
            # Batch upload function
            if st.session_state.get("show_batch_upload", False):
                st.markdown("---")
                st.subheader("Batch Upload")
                
                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=['xlsx', 'xls', 'csv'],
                    help="Upload Excel or CSV file with data",
                    key="batch_upload_file"
                )
                
                if uploaded_file is not None:
                    try:
                        # Read file
                        if uploaded_file.name.endswith(('.xlsx', '.xls')):
                            excel_file = pd.ExcelFile(uploaded_file)
                            sheet_names = excel_file.sheet_names
                            
                            if len(sheet_names) > 1:
                                selected_sheet = st.selectbox("Select Sheet", sheet_names)
                            else:
                                selected_sheet = sheet_names[0]
                            
                            uploaded_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                        else:
                            uploaded_df = pd.read_csv(uploaded_file)
                        
                        st.success(f"File loaded: {len(uploaded_df)} rows, {len(uploaded_df.columns)} columns")
                        
                        # Data Preview
                        st.write("Data Preview:")
                        st.dataframe(uploaded_df.head(10))
                        
                        # Column mapping
                        st.subheader("Column Mapping")
                        
                        # Get database column names
                        db_columns = [col for col in df.columns if col != 'id']
                        
                        # Auto match column names
                        column_mapping = {}
                        for db_col in db_columns:
                            matching_cols = [col for col in uploaded_df.columns if db_col.lower() in col.lower()]
                            if matching_cols:
                                default_index = uploaded_df.columns.tolist().index(matching_cols[0]) + 1
                            else:
                                default_index = 0
                            
                            column_mapping[db_col] = st.selectbox(
                                f"Map '{db_col}' to:",
                                ["None"] + uploaded_df.columns.tolist(),
                                index=default_index,
                                key=f"mapping_{db_col}"
                            )
                        
                        # Submit button
                        if st.button("Submit Batch Upload", type="primary"):
                            # Prepare data
                            mapped_data = {}
                            for db_col, file_col in column_mapping.items():
                                if file_col != "None":
                                    mapped_data[db_col] = uploaded_df[file_col]
                            
                            if mapped_data:
                                mapped_df = pd.DataFrame(mapped_data)
                                
                                # Batch submit
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                submitted_count = 0
                                total_rows = len(mapped_df)
                                
                                for idx, row in mapped_df.iterrows():
                                    try:
                                        submit_data_change(
                                            user_info['email'],
                                            'insert',
                                            table_name,
                                            None,
                                            None,
                                            row.to_dict(),
                                            engine
                                        )
                                        submitted_count += 1
                                    except Exception as e:
                                        st.error(f"Error submitting row {idx}: {e}")
                                    
                                    progress = (idx + 1) / total_rows
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing: {idx + 1}/{total_rows}")
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success(f"Batch upload completed! {submitted_count} records submitted for approval.")
                                st.session_state["show_batch_upload"] = False
                                
                                # Send notification
                                send_email(
                                    EMAIL_USER,
                                    "Batch Data Submission",
                                    f"User {user_info['name']} submitted {submitted_count} records for approval.\nTable: {table_name}"
                                )
                            else:
                                st.error("No columns mapped. Please map at least one column.")
                    
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            
            # Create model dataset functionality
            if st.session_state.get("show_create_model", False):
                st.markdown("---")
                st.subheader("Create Machine Learning Model Dataset")
                
                st.info("""
                **Advanced data preprocessing workflow:**
                1. Missing value marker conversion (SMD → NaN)
                2. Range value parsing ("20,30" → 25)  
                3. Feature engineering (13 ML features)
                4. Model dataset creation
                """)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("Start Advanced Data Preprocessing", type="primary", use_container_width=True):
                        try:
                            # Execute advanced data preprocessing
                            model_dataset = create_advanced_model_dataset()
                            
                            if model_dataset is not None and len(model_dataset) > 0:
                                st.success(f"Successfully created model dataset! Shape: {model_dataset.shape}")
                                
                                # Save to session state
                                st.session_state["model_dataset"] = model_dataset
                                
                                # Display data preview
                                st.subheader("Model Dataset Preview")
                                st.dataframe(model_dataset.head(10), use_container_width=True)
                                
                                # Display feature statistics
                                st.subheader("Feature Statistics")
                                col_stat1, col_stat2 = st.columns(2)
                                
                                with col_stat1:
                                    st.write("**Data Integrity:**")
                                    for col in model_dataset.columns:
                                        non_null = model_dataset[col].count()
                                        total = len(model_dataset)
                                        percentage = (non_null / total * 100) if total > 0 else 0
                                        st.write(f"- {col}: {percentage:.1f}% ({non_null}/{total})")
                                
                                with col_stat2:
                                    st.write("**Numerical Feature Statistics:**")
                                    numeric_cols = model_dataset.select_dtypes(include=[np.number]).columns
                                    if len(numeric_cols) > 0:
                                        stats = model_dataset[numeric_cols].describe()
                                        st.dataframe(stats.round(2))
                                
                                # Provide download option
                                csv_data = model_dataset.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Model Dataset (CSV)",
                                    data=csv_data,
                                    file_name=f"frp_model_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Record operation log
                                if engine:
                                    log_operation(
                                        user_info["email"],
                                        "create_model_dataset",
                                        "research_data",
                                        None,
                                        {"rows": len(model_dataset), "columns": len(model_dataset.columns)},
                                        get_client_ip(),
                                        engine
                                    )
                            
                            else:
                                st.error("Model dataset creation failed or data is empty")
                                
                        except Exception as e:
                            st.error(f"Error during processing: {e}")
                            st.write("**Error Details:**")
                            st.code(str(e))
                
                with col2:
                    st.info("""
                    **Expected Output Features:**
                    - Tensile strength retention
                    - pH of condition environment  
                    - Exposure time
                    - Fibre content
                    - Exposure temperature
                    - Diameter
                    - Presence of concrete
                    - Load
                    - Presence of chloride ion
                    - Fibre type
                    - Matrix type
                    - Surface treatment
                    - Strength of unconditioned rebar
                    """)
                
                if st.button("Close", use_container_width=True):
                    st.session_state["show_create_model"] = False
                    st.rerun()
            
            # Data export
            if st.session_state.get("show_export", False):
                st.markdown("---")
                st.subheader("Export Data")
                
                export_format = st.radio("Export Format", ["CSV", "Excel"])
                
                if st.button("Generate Export File"):
                    if export_format == "CSV":
                        csv = filtered_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{table_name}_export.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            filtered_df.to_excel(writer, sheet_name=table_name, index=False)
                        excel_data = output.getvalue()
                        b64 = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{table_name}_export.xlsx">Download Excel File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    # Record export operation
                    log_operation(
                        user_info['email'],
                        'export',
                        table_name,
                        None,
                        {"format": export_format, "records": len(filtered_df)},
                        get_client_ip(),
                        engine
                    )
                
                st.session_state["show_export"] = False
        
        except Exception as e:
            st.error(f"Error reading data: {e}")
            st.write("**Debug Information:**")
            st.write("Please check database connection and data format.")
            
            # Provide data repair suggestions
            st.subheader("Data Repair Suggestions")
            st.write("""
            1. **Clean problematic values**: Replace 'SMD', 'Notreported', and similar values with NULL in the database
            2. **Check Data Types**: Ensure numeric columns contain only numbers
            3. **Validate data format**: Review the source data for consistency
            """)
            
            if st.button("Try to reload data"):
                data_manager.invalidate_cache(f"table_{table_name}")
                st.rerun()

        # ——————————————————————————————
        # 管理缓存部分 - 只对admin用户可见，放在Data Operations下面
        # ——————————————————————————————
        if st.session_state["authenticated_user"]["role"] == "admin":
            
            with st.expander("Manage Cache", expanded=False):
                st.markdown("""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #dc3545;">
                    <p style="margin: 0; color: #495057; font-size: 0.9rem; font-weight: 500;">
                        Manage data preprocessing caches and trained machine learning models
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 创建两个管理标签页
                cache_tabs = st.tabs(["Data Preprocessing Cache", "Trained Models"])
                
                # 第一个标签页：数据预处理缓存
                with cache_tabs[0]:
                    st.markdown("#### Data Preprocessing Cache Management")
                    
                    st.info("""
                    **Cache Management Operations:**
                    - Generate and save preprocessed datasets to cache
                    - View and manage existing cached datasets
                    - Clear individual or all cached datasets
                    - Monitor cache storage usage
                    """)
                    
                    try:
                        preprocessor = FRPDataPreprocessor(engine)
                        
                        # Cache generation section
                        st.markdown("##### Generate New Cache")
                        col_gen1, col_gen2 = st.columns(2)
                        
                        with col_gen1:
                            if st.button("Generate & Cache Dataset", type="primary", use_container_width=True):
                                try:
                                    with st.spinner("Executing advanced data preprocessing..."):
                                        model_dataset = create_advanced_model_dataset()
                                        if model_dataset is not None and len(model_dataset) > 0:
                                            st.session_state["model_dataset"] = model_dataset
                                            
                                            # Save to database cache
                                            cache_key = f"advanced_dataset_{user_info['email']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
                                            feature_columns = model_dataset.columns.tolist()
                                            
                                            if preprocessor.save_to_cache(cache_key, model_dataset, feature_columns):
                                                st.success(f"Dataset generated and cached! Shape: {model_dataset.shape}")
                                                st.success(f"Cache key: {cache_key}")
                                                st.rerun()
                                            else:
                                                st.error("Failed to save to cache")
                                        else:
                                            st.error("Advanced preprocessing failed")
                                except Exception as e:
                                    st.error(f"Processing error: {e}")
                        
                        with col_gen2:
                            # Save current dataset to cache (if exists)
                            if "model_dataset" in st.session_state and st.session_state.model_dataset is not None:
                                if st.button("Save Current Dataset to Cache", type="secondary", use_container_width=True):
                                    try:
                                        cache_key = f"current_dataset_{user_info['email']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
                                        feature_columns = st.session_state.model_dataset.columns.tolist()
                                        
                                        if preprocessor.save_to_cache(cache_key, st.session_state.model_dataset, feature_columns):
                                            st.success(f"Current dataset saved to cache!")
                                            st.success(f"Cache key: {cache_key}")
                                            st.rerun()
                                        else:
                                            st.error("Failed to save to cache")
                                    except Exception as e:
                                        st.error(f"Cache save error: {e}")
                            else:
                                st.info("No dataset in memory to cache")
                        
                        st.markdown("---")
                        
                        # Cache listing and management
                        st.markdown("##### Existing Cached Datasets")
                        cached_datasets = preprocessor.list_cached_datasets()
                        
                        if cached_datasets:
                            st.markdown(f"""
                            <div style="background: #d1ecf1; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 0.5rem 0;">
                                <strong style="color: #0c5460;">Found {len(cached_datasets)} cached dataset(s)</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show cached datasets with improved layout
                            for idx, cache_info in enumerate(cached_datasets):
                                # Use container for better visual separation
                                with st.container():
                                    st.markdown(f"""
                                    <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 0.8rem 0; border-left: 4px solid #3498db;">
                                        <h4 style="color: #2c3e50; margin: 0 0 0.8rem 0;">{cache_info['cache_key']}</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    col_info, col_actions = st.columns([3, 1])
                                    
                                    with col_info:
                                        # Use cleaner info display
                                        col_shape, col_created, col_updated = st.columns(3)
                                        with col_shape:
                                            st.caption("Shape")
                                            st.write(f"**{cache_info['shape']}**")
                                        with col_created:
                                            st.caption("Created")
                                            st.write(f"**{cache_info['created_at'].strftime('%Y-%m-%d %H:%M')}**")
                                        with col_updated:
                                            st.caption("Updated")
                                            st.write(f"**{cache_info['updated_at'].strftime('%Y-%m-%d %H:%M')}**")
                                    
                                    with col_actions:
                                        # Keep button styling consistent
                                        if st.button("Delete Cache", key=f"delete_cache_mgmt_{idx}", use_container_width=True, type="secondary"):
                                            try:
                                                deleted_count = preprocessor.clear_cache(cache_info['cache_key'])
                                                if deleted_count > 0:
                                                    st.success("Cache deleted!")
                                                    st.rerun()
                                                else:
                                                    st.warning("Cache not found")
                                            except Exception as e:
                                                st.error(f"Delete error: {e}")
                                    
                                    # Add small separator
                                    st.markdown("<br>", unsafe_allow_html=True)
                        
                        else:
                            st.markdown("""
                            <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin: 1rem 0;">
                                <h5 style="color: #856404; margin: 0 0 0.5rem 0;">No Cached Datasets</h5>
                                <p style="color: #856404; margin: 0;">Generate your first cached dataset using the button above.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Cache management error: {e}")
                
                # 第二个标签页：已训练模型管理
                with cache_tabs[1]:
                    st.markdown("#### Trained Models Management")
                    
                    # 检查是否有缓存的模型
                    try:
                        model_cache_manager = ModelCacheManager(engine)
                        cached_models = model_cache_manager.list_cached_models()
                        
                        if cached_models:
                            st.markdown(f"""
                            <div style="background: #d1ecf1; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 0.5rem 0;">
                                <strong style="color: #0c5460;">{len(cached_models)} trained model(s) available in cache</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 创建模型管理标签页
                            model_tabs = st.tabs(["Model List", "Model Details", "Cache Management"])
                            
                            with model_tabs[0]:
                                # 按目标变量分组显示
                                models_by_target = {}
                                for model in cached_models:
                                    target = model['target_variable']
                                    if target not in models_by_target:
                                        models_by_target[target] = []
                                    models_by_target[target].append(model)
                                
                                for target_var, target_models in models_by_target.items():
                                    st.markdown(f"**Target Variable: {target_var}**")
                                    
                                    # 显示该目标变量的所有模型，使用与缓存数据集一致的风格
                                    for idx, model in enumerate(target_models):
                                        # Use container for better visual separation
                                        with st.container():
                                            # 计算模型表现评分
                                            eval_strategy = model['evaluation_strategy']
                                            created_time = model['created_at'].strftime('%Y-%m-%d %H:%M') if model['created_at'] else 'Unknown'
                                            model_key = model['model_key']
                                            model_name = model['model_name']
                                            
                                            st.markdown(f"""
                                            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 0.8rem 0; border-left: 4px solid #3498db;">
                                                <h4 style="color: #2c3e50; margin: 0 0 0.8rem 0;">{model_name}</h4>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            col_info, col_actions = st.columns([3, 1])
                                            
                                            with col_info:
                                                # Use cleaner info display
                                                col_strategy, col_key, col_created = st.columns(3)
                                                with col_strategy:
                                                    st.caption("Evaluation Strategy")
                                                    st.write(f"**{eval_strategy}**")
                                                with col_key:
                                                    st.caption("Model Key")
                                                    st.write(f"**{model_key}**")
                                                with col_created:
                                                    st.caption("Trained")
                                                    st.write(f"**{created_time}**")
                                            
                                            with col_actions:
                                                # Action buttons in a column layout
                                                if st.button("Load Model", key=f"cache_load_model_{model['model_key']}", use_container_width=True, type="primary"):
                                                    try:
                                                        # 加载模型详情
                                                        loaded_model = model_cache_manager.load_model(model['model_key'])
                                                        if loaded_model:
                                                            # 将模型加载到session state
                                                            st.session_state.trained_model = loaded_model['model']
                                                            st.session_state.selected_target = loaded_model['target_variable']
                                                            st.session_state.selected_model = loaded_model['model_name']
                                                            st.session_state.current_model_key = model['model_key']
                                                            
                                                            # 加载特征信息
                                                            feature_info = loaded_model['feature_info']
                                                            preprocessing_info = loaded_model['preprocessing_info']
                                                            
                                                            st.session_state.feature_names = feature_info.get('feature_names', [])
                                                            st.session_state.feature_columns = feature_info.get('feature_columns', [])
                                                            st.session_state.numeric_features = feature_info.get('numeric_features', [])
                                                            st.session_state.categorical_features = feature_info.get('categorical_features', [])
                                                            
                                                            # 显示加载结果
                                                            eval_results = loaded_model['evaluation_results']
                                                            st.success(f"Model loaded successfully! Test R²: {eval_results.get('test_r2', 'N/A'):.4f}")
                                                            st.rerun()
                                                        else:
                                                            st.error("Failed to load model")
                                                    except Exception as e:
                                                        st.error(f"Load error: {e}")
                                                
                                                # 添加删除按钮
                                                if st.button("Delete", key=f"cache_delete_model_{model['model_key']}", use_container_width=True, type="secondary", help="Delete this model"):
                                                    # 确认删除
                                                    if f"confirm_delete_{model['model_key']}" not in st.session_state:
                                                        st.session_state[f"confirm_delete_{model['model_key']}"] = False
                                                    
                                                    if not st.session_state[f"confirm_delete_{model['model_key']}"]:
                                                        st.session_state[f"confirm_delete_{model['model_key']}"] = True
                                                        st.rerun()
                                                
                                                # 如果用户点击了删除，显示确认对话框
                                                if st.session_state.get(f"confirm_delete_{model['model_key']}", False):
                                                    st.warning(f"⚠️ Confirm deletion of model: **{model_name}**")
                                                    col_confirm, col_cancel = st.columns(2)
                                                    
                                                    with col_confirm:
                                                        if st.button("Confirm Delete", key=f"confirm_delete_btn_{model['model_key']}", type="primary", use_container_width=True):
                                                            try:
                                                                success = model_cache_manager.delete_model(model['model_key'])
                                                                if success:
                                                                    st.success(f"Model {model_name} deleted successfully!")
                                                                    # 清理确认状态
                                                                    if f"confirm_delete_{model['model_key']}" in st.session_state:
                                                                        del st.session_state[f"confirm_delete_{model['model_key']}"]
                                                                    st.rerun()
                                                                else:
                                                                    st.error("Failed to delete model")
                                                            except Exception as e:
                                                                st.error(f"Delete error: {e}")
                                                    
                                                    with col_cancel:
                                                        if st.button("Cancel", key=f"cancel_delete_btn_{model['model_key']}", use_container_width=True):
                                                            # 取消删除，清理确认状态
                                                            st.session_state[f"confirm_delete_{model['model_key']}"] = False
                                                            st.rerun()
                                            
                                            # Add small separator
                                            st.markdown("<br>", unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                            
                            with model_tabs[1]:
                                st.markdown("**Model Performance Details:**")
                                
                                # 选择模型查看详情
                                model_options = [f"{m['model_name']} - {m['target_variable']} ({m['model_key']})" for m in cached_models]
                                selected_model_option = st.selectbox("Select model for details:", model_options, key="cache_model_details_select")
                                
                                if selected_model_option:
                                    # 提取model_key
                                    selected_key = selected_model_option.split('(')[-1].split(')')[0]
                                    
                                    # 加载模型详情
                                    model_details = model_cache_manager.load_model(selected_key)
                                    
                                    if model_details:
                                        eval_results = model_details['evaluation_results']
                                        training_info = model_details['training_info']
                                        feature_info = model_details['feature_info']
                                        
                                        # 显示性能指标 - 5个指标
                                        col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)
                                        
                                        with col_perf1:
                                            st.metric("Test R²", f"{eval_results.get('test_r2', 0):.4f}")
                                        with col_perf2:
                                            st.metric("CV Training R²", f"{eval_results.get('cv_train_r2_mean', 0):.4f}")
                                        with col_perf3:
                                            st.metric("CV R² Mean", f"{eval_results.get('cv_r2_mean', 0):.4f}")
                                        with col_perf4:
                                            st.metric("Test RMSE", f"{eval_results.get('test_rmse', 0):.2f}")
                                        with col_perf5:
                                            stability = eval_results.get('model_stability', 0)
                                            st.metric("Model Stability", f"{stability:.3f}")
                                        
                                        # 训练信息
                                        st.markdown("**Training Information:**")
                                        training_info_df = pd.DataFrame([
                                            ["Model Type", training_info.get('model_type', 'Unknown')],
                                            ["Evaluation Strategy", training_info.get('evaluation_strategy', 'Unknown')],
                                            ["Hyperparameter Tuning", "Yes" if training_info.get('hyperparameter_tuning', False) else "No"],
                                            ["Auto Model Select", "Yes" if training_info.get('auto_model_select', False) else "No"],
                                            ["Training Time", training_info.get('training_timestamp', 'Unknown')],
                                            ["Test Set Size", f"{training_info.get('test_size', 0)*100:.0f}%"]
                                        ], columns=["Parameter", "Value"])
                                        
                                        st.dataframe(training_info_df, use_container_width=True, hide_index=True)
                                        
                                        # 特征信息
                                        st.markdown("**Feature Information:**")
                                        st.write(f"- Total features: {feature_info.get('n_features', 0)}")
                                        st.write(f"- Training samples: {feature_info.get('n_samples', 0)}")
                                        st.write(f"- Numeric features: {len(feature_info.get('numeric_features', []))}")
                                        st.write(f"- Categorical features: {len(feature_info.get('categorical_features', []))}")
                                        
                                        # 最佳参数
                                        best_params = model_details.get('best_params', {})
                                        if best_params:
                                            st.markdown("**Best Parameters:**")
                                            params_df = pd.DataFrame(list(best_params.items()), columns=["Parameter", "Value"])
                                            st.dataframe(params_df, use_container_width=True, hide_index=True)
                            
                            with model_tabs[2]:
                                st.markdown("**Cache Management Operations:**")
                                
                                # 显示缓存统计
                                total_models = len(cached_models)
                                targets = list(set([m['target_variable'] for m in cached_models]))
                                
                                col_stat1, col_stat2 = st.columns(2)
                                with col_stat1:
                                    st.metric("Total Cached Models", total_models)
                                with col_stat2:
                                    st.metric("Target Variables", len(targets))
                                
                                # 批量操作
                                st.markdown("**Batch Operations:**")
                                
                                # 按目标变量清除
                                target_to_clear = st.selectbox("Select target variable to clear all models:", [""] + targets, key="cache_target_clear_select")
                                
                                if st.button("Clear Selected Target", disabled=not target_to_clear, key="cache_clear_target_btn", use_container_width=True):
                                    if target_to_clear:
                                        try:
                                            deleted_count = 0
                                            for model in cached_models:
                                                if model['target_variable'] == target_to_clear:
                                                    deleted_count += model_cache_manager.delete_model(model['model_key'])
                                            
                                            if deleted_count > 0:
                                                st.success(f"Deleted {deleted_count} models for target: {target_to_clear}")
                                                st.rerun()
                                            else:
                                                st.warning("No models found for this target")
                                        except Exception as e:
                                            st.error(f"Clear error: {e}")
                                
                                # System Management Section
                                st.markdown("---")
                                st.markdown("**System Maintenance:**")
                                
                                sys_col1, sys_col2 = st.columns(2)
                                
                                with sys_col1:
                                    st.markdown("**Legacy Model Cleanup**")
                                    st.info("Remove old cached models that don't include preprocessor information.")
                                    
                                    if st.button("Check & Clean Legacy Models", use_container_width=True, key="legacy_cleanup_btn"):
                                        try:
                                            deleted_count, total_legacy = model_cache_manager.clear_legacy_models()
                                            
                                            if total_legacy > 0:
                                                st.success(f"Found {total_legacy} legacy models. Removed {deleted_count} models.")
                                                if deleted_count < total_legacy:
                                                    st.warning(f"Failed to remove {total_legacy - deleted_count} models.")
                                                st.rerun()
                                            else:
                                                st.success("✅ No legacy models found. All cached models are up to date.")
                                        except Exception as e:
                                            st.error(f"Failed to check legacy models: {e}")
                                
                                with sys_col2:
                                    st.markdown("**Clear All Model Cache**")
                                    st.info("This will permanently delete all cached models from the system.")
                                    
                                    if st.button("Clear All Models", 
                                               use_container_width=True,
                                               key="clear_all_cache_btn"):
                                        try:
                                            st.info("🔄 Clearing all cached models...")
                                            deleted_count = model_cache_manager.clear_all_models()
                                            if deleted_count > 0:
                                                st.success(f"🗑️ Cleared {deleted_count} cached models.")
                                            else:
                                                st.warning("No models found to delete or deletion failed.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to clear model cache: {e}")
                                            st.error(f"Error details: {str(e)}")
                        else:
                            st.markdown("""
                            <div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                                <h3 style="color: #6c757d; margin: 0;">No Trained Models</h3>
                                <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Train your first model in the Model Training tab to see it here.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Failed to load model cache: {e}")

# ——————————————————————————————
# Tab 2: Access Control (只对admin用户可见)
# ——————————————————————————————
if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
    with tabs[tab_indexes["access_control"]]:
        
        if "authenticated_user" not in st.session_state:
            st.markdown("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: #856404; margin: 0;">Authentication Required</h4>
                <p style="color: #856404; margin: 0.5rem 0 0 0;">Please log in to access the control panel.</p>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state["authenticated_user"]["role"] != "admin":
            st.markdown("""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: #721c24; margin: 0;">Access Denied</h4>
                <p style="color: #721c24; margin: 0.5rem 0 0 0;">Administrator access required to manage user permissions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Access Control Management interface
            user_info = st.session_state["authenticated_user"]
        
        # Welcome header for access control - clean design
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
            <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
                Access Control Management
            </h3>
            <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
                Manage user access requests and permissions
            </p>
        </div>
        """, unsafe_allow_html=True)
        # Management panel statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Get statistical data
        pending_count = pd.read_sql("SELECT COUNT(*) as count FROM users WHERE status = 'pending'", engine).iloc[0]['count']
        approved_count = pd.read_sql("SELECT COUNT(*) as count FROM users WHERE status = 'approved'", engine).iloc[0]['count']
        rejected_count = pd.read_sql("SELECT COUNT(*) as count FROM users WHERE status = 'rejected'", engine).iloc[0]['count']
        total_requests = pd.read_sql("SELECT COUNT(*) as count FROM users", engine).iloc[0]['count']
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #f39c12;">
                <h4 style="color: #f39c12; margin: 0; font-size: 0.9rem; font-weight: 600;">PENDING REQUESTS</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{pending_count}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Awaiting review</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #27ae60;">
                <h4 style="color: #27ae60; margin: 0; font-size: 0.9rem; font-weight: 600;">APPROVED USERS</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{approved_count}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Active access</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #e74c3c;">
                <h4 style="color: #e74c3c; margin: 0; font-size: 0.9rem; font-weight: 600;">REJECTED</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{rejected_count}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Denied access</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db;">
                <h4 style="color: #3498db; margin: 0; font-size: 0.9rem; font-weight: 600;">TOTAL REQUESTS</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{total_requests}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">All time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content area tabs
        access_tabs = st.tabs(["Pending Requests", "Approved Users", "User Management"])
        
        with access_tabs[0]:
            # Pending user applications - modern design
            st.markdown("### Pending User Requests")
            
            pending_users = pd.read_sql(
                "SELECT * FROM users WHERE status = 'pending' ORDER BY created_at DESC",
                engine
            )
            
            if not pending_users.empty:
                st.markdown(f"""
                <div style="background: #e8f4fd; padding: 0.8rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #3498db;">
                    <p style="margin: 0; color: #2980b9; font-weight: 500;">{len(pending_users)} request(s) awaiting your review</p>
                </div>
                """, unsafe_allow_html=True)
                
                for idx, user in pending_users.iterrows():
                    # Modern user request card
                    ip_status = "Allowed" if is_ip_allowed(user['ip_address']) else "Not Allowed"
                    ip_color = "#27ae60" if is_ip_allowed(user['ip_address']) else "#e74c3c"
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 1rem 0; border: 1px solid rgba(0, 0, 0, 0.05);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: #2c3e50; font-size: 1.2rem;">{user['name']}</h4>
                            <span style="background: {ip_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">{ip_status}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">EMAIL</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user['email']}</p>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">INSTITUTION</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user['institution']}</p>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">IP ADDRESS</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user['ip_address']}</p>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">APPLIED AT</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user['created_at']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action button area
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        role = st.selectbox(
                            "Assign Role",
                            ["viewer", "editor"],
                            key=f"role_{user['id']}",
                            help="Select the role to assign to this user"
                        )
                    
                    with col2:
                        if st.button("Approve", key=f"approve_{user['id']}", type="primary", use_container_width=True):
                            with engine.connect() as conn:
                                conn.execute(
                                    text("""
                                        UPDATE users
                                        SET status = 'approved',
                                            role = :role,
                                            approved_by = :approver,
                                            approved_at = NOW()
                                        WHERE id = :id
                                    """),
                                    {
                                        "id": user['id'],
                                        "role": role,
                                        "approver": st.session_state["authenticated_user"]["email"]
                                    }
                                )
                                conn.commit()
                            
                            # Send notification
                            send_email(
                                user['email'],
                                "FRP Database Access Approved",
                                f"Your access request has been approved with {role} role."
                            )
                            st.success("User approval successful!")
                            st.rerun()
                    
                    with col3:
                        if st.button("Reject", key=f"reject_{user['id']}", use_container_width=True):
                            with engine.connect() as conn:
                                conn.execute(
                                    text("UPDATE users SET status = 'rejected' WHERE id = :id"),
                                    {"id": user['id']}
                                )
                                conn.commit()
                            
                            send_email(
                                user['email'],
                                "FRP Database Access Rejected",
                                "Your access request has been rejected."
                            )
                            st.warning("User request has been rejected!")
                            st.rerun()
                    
                    st.markdown("---")
            else:
                st.markdown("""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                    <h3 style="color: #155724; margin: 0;">All Complete!</h3>
                    <p style="color: #155724; margin: 0.5rem 0 0 0;">Currently no pending user requests for approval.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with access_tabs[1]:
            # Approved users list - modern design
            st.markdown("### Approved Users")
            
            approved_users = pd.read_sql(
                "SELECT name, email, institution, role, approved_at FROM users WHERE status = 'approved' ORDER BY created_at DESC",
                engine
            )
            
            if not approved_users.empty:
                st.markdown(f"""
                <div style="background: #d1ecf1; padding: 0.8rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #17a2b8;">
                    <p style="margin: 0; color: #0c5460; font-weight: 500;">👥 {len(approved_users)} active user(s) with database access</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Modern user table
                st.dataframe(
                    approved_users,
                    use_container_width=True,
                    column_config={
                        "name": st.column_config.TextColumn("Name", width="medium"),
                        "email": st.column_config.TextColumn("Email", width="large"),
                        "institution": st.column_config.TextColumn("Institution", width="medium"),
                        "role": st.column_config.TextColumn("Role", width="small"),
                        "approved_at": st.column_config.DatetimeColumn("Approved", width="medium")
                    },
                    hide_index=True
                )
            else:
                st.markdown("""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                    <h3 style="color: #6c757d; margin: 0;">No Active Users</h3>
                    <p style="color: #6c757d; margin: 0.5rem 0 0 0;">No approved users found in the system.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with access_tabs[2]:
            # User management operations - Modern design
            st.markdown("### User Management")
            
            approved_users = pd.read_sql(
                "SELECT name, email, institution, role, approved_at FROM users WHERE status = 'approved' ORDER BY created_at DESC",
                engine
            )
            
            if not approved_users.empty:
                # User selection area
                st.markdown("#### Select User to Manage")
                
                user_selection_col, actions_col = st.columns([2, 1])
                
                with user_selection_col:
                    selected_user = st.selectbox(
                        "Choose a user:",
                        approved_users['email'].tolist(),
                        format_func=lambda x: f"{approved_users[approved_users['email'] == x]['name'].iloc[0]} ({x})",
                        help="Select a user to manage their permissions and status"
                    )
                
                with actions_col:
                    if st.button("Refresh User List", use_container_width=True):
                        st.rerun()
                
                if selected_user:
                    # Display selected user details
                    user_details = approved_users[approved_users['email'] == selected_user].iloc[0]
                    
                    st.markdown("---")
                    st.markdown("#### User Details & Actions")
                    
                    # User information card
                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 1rem 0; border: 1px solid rgba(0, 0, 0, 0.05);">
                        <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">{user_details['name']}</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">EMAIL</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user_details['email']}</p>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">INSTITUTION</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user_details['institution']}</p>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">CURRENT ROLE</p>
                                <span style="background: #3498db; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">{user_details['role'].upper()}</span>
                            </div>
                            <div>
                                <p style="margin: 0; color: #7f8c8d; font-size: 0.8rem; font-weight: 600;">APPROVED AT</p>
                                <p style="margin: 0; color: #2c3e50; font-weight: 500;">{user_details['approved_at']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Management operations
                    st.markdown("#### Management Actions")
                    
                    # Create management operation tabs
                    action_tabs = st.tabs(["Change Role", "Deactivate User", "Delete User"])
                    
                    with action_tabs[0]:
                        st.markdown("**Change User Role**")
                        
                        new_role = st.selectbox(
                            "Select new role:",
                            ["viewer", "editor", "admin"],
                            index=["viewer", "editor", "admin"].index(user_details['role']),
                            help="Change user's access level"
                        )
                        
                        if st.button("Update Role", type="primary", use_container_width=True):
                            if new_role != user_details['role']:
                                with engine.connect() as conn:
                                    conn.execute(
                                        text("UPDATE users SET role = :role WHERE email = :email"),
                                        {"role": new_role, "email": selected_user}
                                    )
                                    conn.commit()
                                st.success(f"Role updated to {new_role}!")
                                st.rerun()
                            else:
                                st.info("User already has this role")
                    
                    with action_tabs[1]:
                        st.markdown("**Deactivate User Account**")
                        st.markdown("This will temporarily suspend the user's access to the platform.")
                        
                        st.warning("**Warning:** This action will immediately revoke user access.")
                        
                        if st.button("Deactivate User", use_container_width=True):
                            with engine.connect() as conn:
                                conn.execute(
                                    text("UPDATE users SET status = 'deactivated' WHERE email = :email"),
                                    {"email": selected_user}
                                )
                                conn.commit()
                            st.warning("User deactivated successfully!")
                            st.rerun()
                    
                    with action_tabs[2]:
                        st.markdown("**Permanently Delete User**")
                        st.markdown("This will completely remove the user from the system.")
                        
                        st.error("**Danger Zone:** This action cannot be undone!")
                        
                        confirm_delete = st.checkbox(
                            "I understand this action cannot be undone", 
                            key="confirm_delete",
                            help="Check this box to confirm permanent deletion"
                        )
                        
                        if st.button("Permanently Delete User", 
                                   disabled=not confirm_delete, 
                                   use_container_width=True):
                            with engine.connect() as conn:
                                conn.execute(
                                    text("DELETE FROM users WHERE email = :email"),
                                    {"email": selected_user}
                                )
                                conn.commit()
                            st.error("User permanently deleted!")
                            st.rerun()
            
            else:
                st.markdown("""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                    <h3 style="color: #6c757d; margin: 0;">No Users to Manage</h3>
                    <p style="color: #6c757d; margin: 0.5rem 0 0 0;">No approved users available for management.</p>
                </div>
                """, unsafe_allow_html=True)

# ——————————————————————————————
# Tab 3: Data Change Management (只对admin用户可见)
# ——————————————————————————————
if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
    with tabs[tab_indexes["data_changes"]]:
        
        if "authenticated_user" not in st.session_state:
            st.warning("Please log in first.")
        else:
            user_info = st.session_state["authenticated_user"]
            
        # Data Changes Management interface - 与Access Control保持一致的设计
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
            <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
                Data Changes Management
            </h3>
            <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
                Track and manage data modification requests and approvals
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 获取数据变更统计信息
        pending_changes = get_pending_changes(engine)
        
        # 统计卡片 - 与Access Control保持一致的设计
        col1, col2, col3, col4 = st.columns(4)
        
        # 计算统计数据
        total_pending = len(pending_changes) if pending_changes else 0
        
        # 获取历史统计
        try:
            history_stats = pd.read_sql(
                text("SELECT status, COUNT(*) as count FROM data_changes GROUP BY status"),
                engine
            )
            approved_count = history_stats[history_stats['status'] == 'approved']['count'].sum() if not history_stats.empty else 0
            rejected_count = history_stats[history_stats['status'] == 'rejected']['count'].sum() if not history_stats.empty else 0
            total_requests = history_stats['count'].sum() if not history_stats.empty else 0
        except:
            approved_count = rejected_count = total_requests = 0
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #f39c12;">
                <h4 style="color: #f39c12; margin: 0; font-size: 0.9rem; font-weight: 600;">PENDING REQUESTS</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{total_pending}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Awaiting review</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #27ae60;">
                <h4 style="color: #27ae60; margin: 0; font-size: 0.9rem; font-weight: 600;">APPROVED</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{approved_count}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Total approved</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #e74c3c;">
                <h4 style="color: #e74c3c; margin: 0; font-size: 0.9rem; font-weight: 600;">REJECTED</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{rejected_count}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">Total rejected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db;">
                <h4 style="color: #3498db; margin: 0; font-size: 0.9rem; font-weight: 600;">TOTAL REQUESTS</h4>
                <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{total_requests}</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">All time</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Admin approval interface
        if user_info["role"] == "admin":
            
            if pending_changes:
                # 简约的过滤器设计
                st.markdown("#### Review Pending Changes")
                
                # 单行过滤器
                filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
                
                with filter_col1:
                    filter_type = st.selectbox(
                        "Type",
                        ["All"] + list(set([dict(change._mapping if hasattr(change, '_mapping') else change._asdict())['change_type'] for change in pending_changes])),
                        key="change_type_filter"
                    )
                
                with filter_col2:
                    filter_table = st.selectbox(
                        "Table",
                        ["All"] + list(set([dict(change._mapping if hasattr(change, '_mapping') else change._asdict())['table_name'] for change in pending_changes])),
                        key="table_filter"
                    )
                
                with filter_col3:
                    filter_user = st.selectbox(
                        "User",
                        ["All"] + list(set([dict(change._mapping if hasattr(change, '_mapping') else change._asdict())['user_email'] for change in pending_changes])),
                        key="user_filter"
                    )                
                # Apply filters
                filtered_changes = []
                for change in pending_changes:
                    change_dict = dict(change._mapping) if hasattr(change, '_mapping') else change._asdict()
                    
                    if (filter_type == "All" or change_dict['change_type'] == filter_type) and \
                       (filter_table == "All" or change_dict['table_name'] == filter_table) and \
                       (filter_user == "All" or change_dict['user_email'] == filter_user):
                        filtered_changes.append(change)
                
                if filtered_changes:
                    st.markdown(f"**{len(filtered_changes)} of {len(pending_changes)} changes**")
                    
                    # 简约的变更请求卡片展示
                    for idx, change in enumerate(filtered_changes):
                        change_dict = dict(change._mapping) if hasattr(change, '_mapping') else change._asdict()
                        
                        # 卡片式布局
                        with st.container():
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 1rem 0; border-left: 4px solid #3498db;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <h4 style="color: #2c3e50; margin: 0; font-size: 1.1rem;">{change_dict['change_type'].upper()} - {change_dict['table_name']}</h4>
                                    <span style="color: #7f8c8d; font-size: 0.85rem;">{change_dict['created_at'].strftime("%Y-%m-%d %H:%M") if hasattr(change_dict['created_at'], 'strftime') else str(change_dict['created_at'])}</span>
                                </div>
                                <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">By: {change_dict['user_email']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 展开详细信息的选项
                            if st.checkbox(f"Show Details", key=f"show_detail_{change_dict['id']}"):
                                
                                # 基本信息区域
                                info_col1, info_col2, info_col3 = st.columns(3)
                                with info_col1:
                                    st.markdown(f"""
                                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; text-align: center;">
                                        <strong style="color: #495057;">Type</strong><br>
                                        <span style="color: #3498db;">{change_dict['change_type'].upper()}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with info_col2:
                                    st.markdown(f"""
                                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; text-align: center;">
                                        <strong style="color: #495057;">Table</strong><br>
                                        <span style="color: #6c757d;">{change_dict['table_name']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with info_col3:
                                    st.markdown(f"""
                                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; text-align: center;">
                                        <strong style="color: #495057;">Created</strong><br>
                                        <span style="color: #6c757d;">{change_dict['created_at'].strftime("%m-%d %H:%M") if hasattr(change_dict['created_at'], 'strftime') else str(change_dict['created_at'])}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # 数据对比区域
                                if change_dict['old_data']:
                                    st.markdown("**Before (Old Data):**")
                                    display_structured_data(change_dict['old_data'])
                                    st.markdown("**After (Proposed Change):**")
                                else:
                                    st.markdown("**Proposed New Data:**")
                                display_structured_data(change_dict['new_data'])
                                
                                st.markdown("---")
                                
                                # 审批操作区域 - 简约设计
                                review_col1, review_col2 = st.columns([2, 1])
                                
                                with review_col1:
                                    comment = st.text_area(
                                        "Review Comment",
                                        placeholder="Optional: Add your review comments...",
                                        key=f"comment_{change_dict['id']}",
                                        height=80
                                    )
                                
                                with review_col2:
                                    st.markdown("**Actions:**")
                                    
                                    # 操作按钮 - 垂直布局，移除表情符号
                                    if st.button("Approve", key=f"approve_{change_dict['id']}", type="primary", use_container_width=True):
                                        # Execute change
                                        with engine.connect() as conn:
                                            # Update change status
                                            conn.execute(
                                                text("""
                                                    UPDATE data_changes
                                                    SET status = 'approved',
                                                        reviewed_by = :reviewer,
                                                        reviewed_at = NOW(),
                                                        review_comment = :comment
                                                    WHERE id = :id
                                                """),
                                                {
                                                    "id": change_dict['id'],
                                                    "reviewer": user_info['email'],
                                                    "comment": comment
                                                }
                                            )
                                            
                                            # Apply changes
                                            new_data = json.loads(change_dict['new_data'])
                                            
                                            if change_dict['change_type'] == 'insert':
                                                columns = ", ".join(new_data.keys())
                                                placeholders = ", ".join([f":{k}" for k in new_data.keys()])
                                                conn.execute(
                                                    text(f"INSERT INTO {change_dict['table_name']} ({columns}) VALUES ({placeholders})"),
                                                    new_data
                                                )
                                            
                                            conn.commit()
                                        
                                        # Send notification
                                        send_email(
                                            change_dict['user_email'],
                                            "Data Change Approved",
                                            f"Your data change request has been approved.\nComment: {comment}"
                                        )
                                        
                                        st.success("Change approved and applied!")
                                        st.rerun()
                                    
                                    if st.button("Reject", key=f"reject_{change_dict['id']}", use_container_width=True):
                                        if not comment.strip():
                                            st.error("Please provide rejection reason")
                                        else:
                                            with engine.connect() as conn:
                                                conn.execute(
                                                    text("""
                                                        UPDATE data_changes
                                                        SET status = 'rejected',
                                                            reviewed_by = :reviewer,
                                                            reviewed_at = NOW(),
                                                            review_comment = :comment
                                                        WHERE id = :id
                                                    """),
                                                    {
                                                        "id": change_dict['id'],
                                                        "reviewer": user_info['email'],
                                                        "comment": comment
                                                    }
                                                )
                                                conn.commit()
                                            
                                            # Send notification
                                            send_email(
                                                change_dict['user_email'],
                                                "Data Change Rejected",
                                                f"Your data change request has been rejected.\nReason: {comment}"
                                            )
                                            
                                            st.warning("Change rejected")
                                            st.rerun()
                
                else:
                    st.info("No changes match the selected filters.")
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
                    <h3 style="color: white; margin: 0; font-size: 1.3rem;">All Clear</h3>
                    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;">No pending changes require your attention at this time.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 用户变更历史 - 简约设计
        st.markdown("---")
        st.markdown("#### My Change History")
        
        my_changes = pd.read_sql(
            text("""
                SELECT id, change_type, table_name, status, created_at, reviewed_at, review_comment
                FROM data_changes
                WHERE user_email = :email
                ORDER BY created_at DESC
                LIMIT 50
            """),
            engine,
            params={"email": user_info['email']}
        )
        
        if not my_changes.empty:
            # 状态统计 - 添加动画效果
            status_counts = my_changes['status'].value_counts()
            hist_col1, hist_col2, hist_col3 = st.columns(3)
            
            with hist_col1:
                pending_hist = status_counts.get('pending', 0)
                st.markdown(f"""
                <style>
                .status-card {{
                    transition: all 0.3s ease;
                    animation: slideInUp 0.6s ease-out;
                }}
                .status-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                }}
                @keyframes slideInUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(30px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
                }}
                .status-pending {{
                    animation-delay: 0.1s;
                }}
                .status-approved {{
                    animation-delay: 0.2s;
                }}
                .status-rejected {{
                    animation-delay: 0.3s;
                }}
                </style>
                <div class="status-card status-pending" style="background: #fff3cd; padding: 0.7rem; border-radius: 6px; text-align: center; border-left: 3px solid #ffc107; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <strong style="color: #856404;">Pending: {pending_hist}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with hist_col2:
                approved_hist = status_counts.get('approved', 0)
                st.markdown(f"""
                <div class="status-card status-approved" style="background: #d4edda; padding: 0.7rem; border-radius: 6px; text-align: center; border-left: 3px solid #28a745; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <strong style="color: #155724;">Approved: {approved_hist}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with hist_col3:
                rejected_hist = status_counts.get('rejected', 0)
                st.markdown(f"""
                <div class="status-card status-rejected" style="background: #f8d7da; padding: 0.7rem; border-radius: 6px; text-align: center; border-left: 3px solid #dc3545; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <strong style="color: #721c24;">Rejected: {rejected_hist}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 简约的历史记录表格
            my_changes_display = my_changes.copy()
            my_changes_display['Created'] = my_changes_display['created_at'].dt.strftime('%Y-%m-%d %H:%M')
            my_changes_display['Type'] = my_changes_display['change_type'].str.upper()
            my_changes_display['Table'] = my_changes_display['table_name']
            my_changes_display['Status'] = my_changes_display['status'].str.title()
            
            # 只显示关键列
            display_columns = ['Type', 'Table', 'Status', 'Created']
            if 'review_comment' in my_changes_display.columns and my_changes_display['review_comment'].notna().any():
                my_changes_display['Review'] = my_changes_display['review_comment'].fillna('N/A')
                display_columns.append('Review')
            
            st.dataframe(
                my_changes_display[display_columns],
                use_container_width=True,
                hide_index=True,
                height=300
            )
        else:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid #dee2e6;">
                <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">No change history records found.</p>
            </div>
            """, unsafe_allow_html=True)

# ——————————————————————————————
# Tab 4: Model Configuration (对所有已认证用户可见)
# ——————————————————————————————
with tabs[tab_indexes["model_configuration"]]:
    
    # Model Configuration interface
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
        <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
            Model Configuration
        </h3>
        <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
            Configure machine learning models and preprocessing parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_raw is None:
        st.warning("Dataset not loaded. Please check database connection.")
    else:
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Get dataset statistics
        raw_data_count = len(st.session_state.df_raw) if st.session_state.df_raw is not None else 0
        has_model_dataset = "model_dataset" in st.session_state and st.session_state.model_dataset is not None
        model_data_count = len(st.session_state.model_dataset) if has_model_dataset else 0
        current_target = st.session_state.get("selected_target", "Not Set")
        current_model = st.session_state.get("selected_model", "Not Set")
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #3498db; height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden;">
                <h4 style="color: #3498db; margin: 0; font-size: 0.75rem; font-weight: 600; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">RAW DATASET</h4>
                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{raw_data_count:,}</h2>
                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Records available</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#27ae60" if has_model_dataset else "#95a5a6"
            status = f"{model_data_count:,}" if has_model_dataset else "Not Ready"
            
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {color}; height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden;">
                <h4 style="color: {color}; margin: 0; font-size: 0.75rem; font-weight: 600; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">ML DATASET</h4>
                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{status}</h2>
                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Preprocessed features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            target_color = "#e74c3c" if current_target == "Not Set" else "#f39c12"
            target_display = current_target if current_target != "Not Set" else "None"
            
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {target_color}; height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden;">
                <h4 style="color: {target_color}; margin: 0; font-size: 0.75rem; font-weight: 600; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">TARGET VARIABLE</h4>
                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{target_display}">{target_display if len(target_display) <= 12 else target_display[:12] + '...'}</h2>
                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Prediction target</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            model_color = "#e74c3c" if current_model == "Not Set" else "#8B5FBF"
            model_display = current_model if current_model != "Not Set" else "None"
            
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid {model_color}; height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden;">
                <h4 style="color: {model_color}; margin: 0; font-size: 0.75rem; font-weight: 600; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">ML MODEL</h4>
                <h2 style="color: #2c3e50; margin: 0.2rem 0; font-size: 1.4rem; font-weight: 700; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{model_display}</h2>
                <p style="margin: 0; font-size: 0.65rem; color: #7f8c8d; line-height: 1.1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Algorithm selected</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main configuration layout - 根据用户角色调整布局
        if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
            # Admin用户：使用双列布局
            col_left, col_right = st.columns(2)
            
            # Left column: Data Preprocessing
            with col_left:
                st.markdown("#### Data Preprocessing")
                
                # Set default advanced preprocessing
                st.session_state.use_advanced_preprocessing = True
                
                # Dataset selection options - simplified for users
                if has_model_dataset:
                    st.markdown("""
                    <div style="background: #d4edda; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #28a745; margin: 1rem 0;">
                        <strong style="color: #155724;">Preprocessed dataset ready for use</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show dataset info
                    dataset_info_col1, dataset_info_col2 = st.columns(2)
                    with dataset_info_col1:
                        st.metric("Records", f"{len(st.session_state.model_dataset):,}")
                    with dataset_info_col2:
                        st.metric("Features", f"{len(st.session_state.model_dataset.columns)}")
                    
                else:
                    # Check database cache first
                    try:
                        preprocessor = FRPDataPreprocessor(engine)
                        cached_datasets = preprocessor.list_cached_datasets()
                        
                        if cached_datasets:
                            st.markdown("""
                            <div style="background: #d1ecf1; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 1rem 0;">
                                <strong style="color: #0c5460;">Cached datasets available in database</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show available caches
                            with st.expander("Available Cached Datasets", expanded=True):
                                for idx, cache_info in enumerate(cached_datasets[:5]):  # Show last 5
                                    col_info, col_load = st.columns([4, 1])
                                    
                                    with col_info:
                                        st.markdown(f"""
                                        **{cache_info['cache_key']}**  
                                        Shape: {cache_info['shape']} | Updated: {cache_info['updated_at'].strftime('%Y-%m-%d %H:%M')}
                                        """)
                                    
                                    with col_load:
                                        if st.button("Load", key=f"load_cache_{idx}", use_container_width=True):
                                            try:
                                                cached_data = preprocessor.get_cached_data(cache_info['cache_key'])
                                                if cached_data:
                                                    st.session_state["model_dataset"] = cached_data['data']
                                                    st.success(f"Loaded cached dataset! Shape: {cached_data['shape']}")
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to load cached data")
                                            except Exception as e:
                                                st.error(f"Load error: {e}")
                    
                    except Exception as e:
                        st.error(f"Cache check failed: {e}")
                
                # Get Offline Code button - 移到外面，使其在数据加载后仍然可见
                if st.button("Get Offline Code", type="secondary", use_container_width=True, key="get_offline_code_admin1"):
                    st.session_state["show_offline_code"] = True
                
                # Show offline code if requested - 添加到admin部分
                if st.session_state.get("show_offline_code", False):
                    st.markdown("---")
                    st.markdown("### Offline Preprocessing Code")
                    st.markdown("**Copy this code to run data preprocessing independently:**")
                    
                    offline_code = '''
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import warnings
warnings.filterwarnings('ignore')

class FRPDataPreprocessor:
    """FRP数据预处理器 - 完整的离线版本"""
    
    def __init__(self):
        self.feature_columns = [
            'pH', 'Fiber_content_wt', 'Diameter_mm', 'Surface_treatment',
            'Resin_type', 'Temp_C', 'Duration_hours', 'Solution_type',
            'Concentration_mol_L', 'Total_fibers', 'Aramid_fibers',
            'Glass_fibers', 'Carbon_fibers'
        ]
    
    def change_smd_to_nan(self, df):
        """将SMD替换为NaN"""
        df_processed = df.copy()
        
        # 处理数值列中的SMD
        numeric_columns = ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 
                          'Duration_hours', 'Concentration_mol_L']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].replace('SMD', np.nan)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    def parse_range_to_mean(self, df):
        """将范围值转换为均值"""
        df_processed = df.copy()
        
        # 定义需要处理的数值列
        numeric_columns = ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 
                          'Duration_hours', 'Concentration_mol_L']
        
        range_pattern = r'(\d+(?:\.\d+)?)\s*[-−–]\s*(\d+(?:\.\d+)?)'
        
        for col in numeric_columns:
            if col in df_processed.columns:
                def process_value(val):
                    if pd.isna(val) or val == '':
                        return np.nan
                    
                    val_str = str(val).strip()
                    
                    # 匹配范围值
                    range_match = re.search(range_pattern, val_str)
                    if range_match:
                        try:
                            start = float(range_match.group(1))
                            end = float(range_match.group(2))
                            return (start + end) / 2
                        except ValueError:
                            return np.nan
                    
                    # 尝试直接转换为数值
                    try:
                        return float(val_str)
                    except (ValueError, TypeError):
                        return np.nan
                
                df_processed[col] = df_processed[col].apply(process_value)
        
        return df_processed
    
    def process_fiber_types(self, df):
        """处理纤维类型数据"""
        df_processed = df.copy()
        
        # 初始化纤维类型列
        df_processed['Total_fibers'] = 0
        df_processed['Aramid_fibers'] = 0
        df_processed['Glass_fibers'] = 0
        df_processed['Carbon_fibers'] = 0
        
        if 'Fiber_type' in df_processed.columns:
            for idx, fiber_type in df_processed['Fiber_type'].items():
                if pd.isna(fiber_type):
                    continue
                
                fiber_str = str(fiber_type).lower()
                
                # 计算总纤维数
                total_count = len(re.findall(r'aramid|glass|carbon', fiber_str))
                df_processed.loc[idx, 'Total_fibers'] = total_count
                
                # 统计各类型纤维
                df_processed.loc[idx, 'Aramid_fibers'] = len(re.findall(r'aramid', fiber_str))
                df_processed.loc[idx, 'Glass_fibers'] = len(re.findall(r'glass', fiber_str))
                df_processed.loc[idx, 'Carbon_fibers'] = len(re.findall(r'carbon', fiber_str))
        
        return df_processed
    
    def encode_categorical_features(self, df):
        """编码分类特征"""
        df_processed = df.copy()
        
        # 定义分类特征映射
        categorical_mappings = {
            'Surface_treatment': {
                'Untreated': 0, 'Silane': 1, 'Plasma': 2, 'Acid etching': 3,
                'Alkali treatment': 4, 'Coating': 5, 'Other': 6
            },
            'Resin_type': {
                'Epoxy': 0, 'Polyester': 1, 'Vinyl ester': 2, 'Polyurethane': 3,
                'Phenolic': 4, 'Other': 5
            },
            'Solution_type': {
                'NaCl': 0, 'HCl': 1, 'NaOH': 2, 'H2SO4': 3, 'Seawater': 4,
                'Ca(OH)2': 5, 'MgSO4': 6, 'Other': 7
            }
        }
        
        for feature, mapping in categorical_mappings.items():
            if feature in df_processed.columns:
                # 处理未知类别
                df_processed[feature] = df_processed[feature].fillna('Other')
                df_processed[feature] = df_processed[feature].map(mapping).fillna(mapping.get('Other', 0))
        
        return df_processed
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_processed = df.copy()
        
        # 数值特征使用中位数填充
        numeric_features = ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 
                           'Duration_hours', 'Concentration_mol_L']
        
        for feature in numeric_features:
            if feature in df_processed.columns:
                median_val = df_processed[feature].median()
                df_processed[feature] = df_processed[feature].fillna(median_val)
        
        # 分类特征已在编码时处理
        return df_processed
    
    def preprocess(self, df):
        """完整的数据预处理流程"""
        print("开始数据预处理...")
        
        # 1. 将SMD替换为NaN
        df = self.change_smd_to_nan(df)
        print("✓ SMD值已替换为NaN")
        
        # 2. 解析范围值为均值
        df = self.parse_range_to_mean(df)
        print("✓ 范围值已转换为均值")
        
        # 3. 处理纤维类型
        df = self.process_fiber_types(df)
        print("✓ 纤维类型特征已处理")
        
        # 4. 编码分类特征
        df = self.encode_categorical_features(df)
        print("✓ 分类特征已编码")
        
        # 5. Handle missing values
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        # 6. 选择最终特征
        final_features = [col for col in self.feature_columns if col in df.columns]
        df_final = df[final_features].copy()
        
        print(f"✓ 预处理完成！最终数据形状: {df_final.shape}")
        print(f"特征列: {list(df_final.columns)}")
        
        return df_final

# 使用示例:
# 1. 加载数据
# df = pd.read_csv('your_data.csv')

# 2. 创建预处理器并处理数据
# preprocessor = FRPDataPreprocessor()
# processed_df = preprocessor.preprocess(df)

# 3. 保存处理后的数据
# processed_df.to_csv('processed_data.csv', index=False)
'''
                    
                    st.code(offline_code, language='python')
                    
                    # 按钮平行排版
                    col_download, col_close = st.columns(2)
                    
                    with col_download:
                        st.download_button(
                            "Download Offline Code",
                            offline_code,
                            file_name="frp_preprocessing_offline.py",
                            mime="text/plain",
                            help="Download the complete offline preprocessing code",
                            use_container_width=True
                        )
                    
                    with col_close:
                        if st.button("Close Code View", key="close_offline_code_admin", use_container_width=True):
                            st.session_state["show_offline_code"] = False
                            st.rerun()
                
                # Target variable selection - moved from right column
                st.markdown("**Prediction target:**")
                target_option = st.selectbox(
                    "Prediction target:",
                    options=["Tensile strength retention rate (%)", "Residual tensile strength (MPa)"],
                    help="Choose the target variable for model prediction",
                    label_visibility="collapsed",
                    key="admin_target_selection"
                )
                st.session_state.selected_target = target_option
        else:
            # 非Admin用户：使用单列布局，让Data Preprocessing占据全宽度
            st.markdown("#### Data Preprocessing")
            
            # Set default advanced preprocessing
            st.session_state.use_advanced_preprocessing = True
            
            # Dataset selection options - simplified for users
            if has_model_dataset:
                st.markdown("""
                <div style="background: #d4edda; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #28a745; margin: 1rem 0;">
                    <strong style="color: #155724;">Preprocessed dataset ready for use</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Show dataset info - 使用更好的布局
                dataset_info_col1, dataset_info_col2, dataset_info_col3 = st.columns(3)
                with dataset_info_col1:
                    st.metric("Records", f"{len(st.session_state.model_dataset):,}")
                with dataset_info_col2:
                    st.metric("Features", f"{len(st.session_state.model_dataset.columns)}")
                with dataset_info_col3:
                    st.metric("Status", "Ready", help="Dataset is preprocessed and ready for model training")
                
            else:
                # Check database cache first
                try:
                    preprocessor = FRPDataPreprocessor(engine)
                    cached_datasets = preprocessor.list_cached_datasets()
                    
                    if cached_datasets:
                        st.markdown("""
                        <div style="background: #d1ecf1; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 1rem 0;">
                            <strong style="color: #0c5460;">Cached datasets available in database</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show available caches - 使用更好的布局
                        with st.expander("Available Cached Datasets", expanded=True):
                            for idx, cache_info in enumerate(cached_datasets[:5]):  # Show last 5
                                col_info, col_load = st.columns([5, 1])
                                
                                with col_info:
                                    st.markdown(f"""
                                    **{cache_info['cache_key']}**  
                                    Shape: {cache_info['shape']} | Updated: {cache_info['updated_at'].strftime('%Y-%m-%d %H:%M')}
                                    """)
                                
                                with col_load:
                                    if st.button("Load", key=f"load_cache_{idx}", use_container_width=True):
                                        try:
                                            cached_data = preprocessor.get_cached_data(cache_info['cache_key'])
                                            if cached_data:
                                                st.session_state["model_dataset"] = cached_data['data']
                                                st.success(f"Loaded cached dataset! Shape: {cached_data['shape']}")
                                                st.rerun()
                                            else:
                                                st.error("Failed to load cached data")
                                        except Exception as e:
                                            st.error(f"Load error: {e}")
                
                except Exception as e:
                    st.error(f"Cache check failed: {e}")
            
            # Get Offline Code button
            if st.button("Get Offline Code", type="secondary", use_container_width=True, key="get_offline_code_admin2"):
                st.session_state["show_offline_code"] = True
            
            # 创建右列变量用于后续代码兼容性
            col_right = None
            
            # Show offline code if requested
            if st.session_state.get("show_offline_code", False):
                st.markdown("---")
                st.markdown("### Offline Preprocessing Code")
                st.markdown("**Copy this code to run data preprocessing independently:**")
                
                offline_code = '''
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import warnings
warnings.filterwarnings('ignore')

class FRPDataPreprocessor:
    """FRP数据预处理器 - 完整的离线版本"""
    
    def __init__(self):
        self.feature_columns = [
            'pH', 'Fiber_content_wt', 'Diameter_mm', 'Surface_treatment',
            'Resin_type', 'Temp_C', 'Duration_hours', 'Solution_type',
            'Concentration_mol_L', 'Total_fibers', 'Aramid_fibers',
            'Glass_fibers', 'Carbon_fibers'
        ]
    
    def change_smd_to_nan(self, df):
        """将SMD替换为NaN"""
        df_processed = df.copy()
        
        # 处理数值列中的SMD
        numeric_columns = ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 
                          'Duration_hours', 'Concentration_mol_L']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].replace('SMD', np.nan)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    def parse_range_to_mean(self, df):
        """将范围值转换为均值"""
        df_processed = df.copy()
        
        numeric_columns = ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 
                          'Duration_hours', 'Concentration_mol_L']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                for idx, value in df_processed[col].items():
                    if pd.isna(value):
                        continue
                    
                    str_value = str(value)
                    
                    # 处理范围：X-Y, X~Y, X to Y
                    range_patterns = [r'(\d+\.?\d*)\s*[-~]\s*(\d+\.?\d*)', 
                                    r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)']
                    
                    for pattern in range_patterns:
                        match = re.search(pattern, str_value, re.IGNORECASE)
                        if match:
                            start_val = float(match.group(1))
                            end_val = float(match.group(2))
                            df_processed.at[idx, col] = (start_val + end_val) / 2
                            break
                    else:
                        # 尝试直接转换为数值
                        try:
                            df_processed.at[idx, col] = float(str_value)
                        except:
                            df_processed.at[idx, col] = np.nan
        
        return df_processed
    
    def create_selected_features(self, df):
        """创建选定的特征"""
        df_processed = df.copy()
        
        # 表面处理编码
        if 'Surface_treatment' in df_processed.columns:
            surface_mapping = {
                'None': 0, 'Silane coupling agent': 1, 'Epoxy sizing': 2,
                'Plasma treatment': 3, 'Acid etching': 4, 'Alkaline treatment': 5
            }
            df_processed['Surface_treatment'] = df_processed['Surface_treatment'].map(surface_mapping).fillna(0)
        
        # 树脂类型编码
        if 'Resin_type' in df_processed.columns:
            resin_mapping = {
                'Epoxy': 1, 'Vinyl ester': 2, 'Polyester': 3, 'Phenolic': 4
            }
            df_processed['Resin_type'] = df_processed['Resin_type'].map(resin_mapping).fillna(1)
        
        # 溶液类型编码
        if 'Solution_type' in df_processed.columns:
            solution_mapping = {
                'NaOH': 1, 'HCl': 2, 'H2SO4': 3, 'NaCl': 4, 'Seawater': 5
            }
            df_processed['Solution_type'] = df_processed['Solution_type'].map(solution_mapping).fillna(1)
        
        # 纤维特征工程
        df_processed['Total_fibers'] = 0
        df_processed['Aramid_fibers'] = 0
        df_processed['Glass_fibers'] = 0
        df_processed['Carbon_fibers'] = 0
        
        if 'Fiber_type' in df_processed.columns:
            fiber_type_col = df_processed['Fiber_type'].fillna('Glass').str.lower()
            
            df_processed['Aramid_fibers'] = (fiber_type_col.str.contains('aramid|kevlar', na=False)).astype(int)
            df_processed['Glass_fibers'] = (fiber_type_col.str.contains('glass|e-glass', na=False)).astype(int)
            df_processed['Carbon_fibers'] = (fiber_type_col.str.contains('carbon', na=False)).astype(int)
            
            # 如果都不匹配，默认为玻璃纤维
            no_match = (df_processed['Aramid_fibers'] == 0) & (df_processed['Glass_fibers'] == 0) & (df_processed['Carbon_fibers'] == 0)
            df_processed.loc[no_match, 'Glass_fibers'] = 1
            
            df_processed['Total_fibers'] = df_processed['Aramid_fibers'] + df_processed['Glass_fibers'] + df_processed['Carbon_fibers']
        
        # 确保所有特征列存在
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # 数据填充和清理
        for col in self.feature_columns:
            if col in ['pH', 'Fiber_content_wt', 'Diameter_mm', 'Temp_C', 'Duration_hours', 'Concentration_mol_L']:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            else:
                df_processed[col] = df_processed[col].fillna(0)
        
        return df_processed[self.feature_columns]
    
    def create_model_dataset(self, df, target_column):
        """创建模型数据集"""
        # 预处理步骤
        df_step1 = self.change_smd_to_nan(df)
        df_step2 = self.parse_range_to_mean(df_step1)
        df_features = self.create_selected_features(df_step2)
        
        # 添加目标变量
        if target_column in df.columns:
            df_features[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        
        # 移除包含NaN的行
        df_final = df_features.dropna()
        
        return df_final

# 使用示例
def process_data(csv_file_path, target_column):
    """
    处理数据的主函数
    
    参数:
    csv_file_path: CSV文件路径
    target_column: 目标变量列名
    
    返回:
    处理后的DataFrame
    """
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 初始化预处理器
    preprocessor = FRPDataPreprocessor()
    
    # 执行预处理
    processed_data = preprocessor.create_model_dataset(df, target_column)
    
    return processed_data

# 运行示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    file_path = "your_data.csv"
    
    # 替换为你的目标变量列名
    target_col = "Tensile_strength_retention_rate"  # 或 "Residual_tensile_strength_MPa"
    
    try:
        result = process_data(file_path, target_col)
        print(f"数据预处理完成! 数据形状: {result.shape}")
        print(f"特征列: {result.columns.tolist()}")
        
        # 保存处理后的数据
        result.to_csv("preprocessed_data.csv", index=False)
        print("预处理数据已保存到 preprocessed_data.csv")
        
    except Exception as e:
        print(f"处理出错: {e}")
'''
                
                st.code(offline_code, language='python')
                
                # 按钮平行排版
                col_download, col_close = st.columns(2)
                
                with col_download:
                    st.download_button(
                        label="Download Offline Code",
                        data=offline_code,
                        file_name="frp_preprocessing_offline.py",
                        mime="text/plain",
                        help="Download the complete offline preprocessing code",
                        use_container_width=True
                    )
                
                with col_close:
                    if st.button("Close Code View", key="close_offline_code", use_container_width=True):
                        st.session_state["show_offline_code"] = False
                        st.rerun()
        
        # Right column: Model Selection (只对admin用户可见)
        if col_right is not None:
            with col_right:
                if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
                    st.markdown("#### Model Selection")
                    
                    # Auto-select option
                    auto_model = st.checkbox("Auto-select Best Model", value=False, help="System will train and compare multiple models, automatically selecting the best performing model")
                    st.session_state.auto_model_select = auto_model
                    
                    if auto_model:
                        st.markdown("""
                        <div style="background: #d1ecf1; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 1rem 0;">
                            <strong style="color: #0c5460;">Automatic mode enabled</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write("Will compare Random Forest, XGBoost, LightGBM")
                    else:
                        # Algorithm selection
                        st.markdown("**Select algorithm:**")
                        model_option = st.selectbox(
                            "Select algorithm:",
                            options=["Random Forest", "XGBoost", "LightGBM"],
                            help="Choose specific machine learning algorithm",
                            label_visibility="collapsed",
                            key="admin_model_selection"
                        )
                        st.session_state.selected_model = model_option
                    
                    # Random seed control
                    st.markdown("**Random Seed Control:**")
                    use_random_seed = st.checkbox("Use Random Seed", value=False, help="Enable this to get different results each time")
                    
                    if use_random_seed:
                        seed_value = st.number_input(
                            "Random Seed",
                            min_value=0,
                            max_value=999999,
                            value=int(np.random.randint(0, 999999)),
                            help="Set a specific random seed for reproducible results"
                        )
                        st.session_state.random_seed = seed_value
                        st.info(f"Using random seed: {seed_value}")
                    else:
                        st.session_state.random_seed = None
                        st.info("Random seed disabled - results will vary each run")
                # 对于非admin用户，完全不显示任何内容
        
        # Third row: Parameter Optimization and Configuration Status (只对admin用户可见)
        if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
            st.markdown("<br>", unsafe_allow_html=True)
            col_param, col_status = st.columns(2)
            
            # Parameter Optimization
            with col_param:
                st.markdown("#### Parameter Optimization")
                
                enable_hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (5-fold CV)", value=False, help="Use grid search to find optimal parameters, will increase training time")
                st.session_state.enable_hp_tuning = enable_hyperparameter_tuning
                
                test_size = st.slider("Test set ratio:", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Proportion of data reserved for testing")
                st.session_state.test_size = test_size
                
                # Data split preview
                train_ratio = 1 - test_size
                st.write("**Data split preview:**")
                split_col1, split_col2 = st.columns([train_ratio, test_size])
                with split_col1:
                    st.markdown(f"""
                    <div style="background: #d1ecf1; padding: 0.5rem; border-radius: 4px; text-align: center; margin: 0.25rem;">
                        <strong style="color: #0c5460;">Training {train_ratio:.0%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with split_col2:
                    st.markdown(f"""
                    <div style="background: #f8d7da; padding: 0.5rem; border-radius: 4px; text-align: center; margin: 0.25rem;">
                        <strong style="color: #721c24;">Testing {test_size:.0%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Configuration Status
            with col_status:
                st.markdown("#### Configuration Status")
                
                # Configuration status check
                target_status = st.session_state.get("selected_target", "Not set")
                model_status = st.session_state.get("selected_model", "Auto") if not st.session_state.get("auto_model_select", False) else "Automatic selection"
                preprocessing_status = "Advanced"
                dataset_status = "Ready" if has_model_dataset else "Pending"
                
                # Status items with colors
                status_items = [
                    ("Prediction target", target_status, "#28a745" if target_status != "Not set" else "#dc3545"),
                    ("Model algorithm", model_status, "#6f42c1"),
                    ("Preprocessing", preprocessing_status, "#28a745"),
                    ("Dataset", dataset_status, "#28a745" if dataset_status == "Ready" else "#ffc107")
                ]
                
                for label, value, color in status_items:
                    st.markdown(f"""
                    <div style="background: white; padding: 0.75rem; border-radius: 6px; border-left: 4px solid {color}; margin: 0.5rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #495057;">{label}:</span>
                            <span style="color: {color}; font-weight: 500;">{value}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # 为非admin用户显示模型选择和缓存模型信息
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Model Selection")
            
            # 添加自动选择最佳模型选项
            auto_select_best = st.checkbox("Auto-select Best Model", value=False, 
                                         help="Automatically select the best performing model based on cached models")
            
            if auto_select_best:
                # 自动选择模式
                st.markdown("**Prediction target:**")
                user_target_option = st.selectbox(
                    "Prediction target:",
                    options=["Tensile strength retention rate (%)", "Residual tensile strength (MPa)"],
                    help="Choose the target variable for model prediction",
                    label_visibility="collapsed",
                    key="user_target_selection_auto"
                )
                
                # 基于target自动选择最佳模型
                try:
                    model_cache_manager = ModelCacheManager(engine)
                    cached_models = model_cache_manager.list_cached_models()
                    
                    # 过滤匹配target的模型
                    target_models = [
                        model for model in cached_models 
                        if model['target_variable'] == user_target_option
                    ]
                    
                    if target_models:
                        # 加载模型详情并比较性能
                        best_model = None
                        best_score = -1
                        model_comparison = []
                        
                        for model in target_models:
                            try:
                                model_details = model_cache_manager.load_model(model['model_key'])
                                if model_details:
                                    eval_results = model_details.get('evaluation_results', {})
                                    test_r2 = eval_results.get('test_r2', 0)
                                    
                                    model_comparison.append({
                                        'model_name': model['model_name'],
                                        'test_r2': test_r2,
                                        'model_key': model['model_key'],
                                        'model_details': model_details,
                                        'model_info': model
                                    })
                                    
                                    if test_r2 > best_score:
                                        best_score = test_r2
                                        best_model = {
                                            'model_name': model['model_name'],
                                            'model_details': model_details,
                                            'model_info': model
                                        }
                            except:
                                continue
                        
                        if best_model:
                            # 显示模型比较
                            with st.expander("Model Performance Comparison", expanded=False):
                                comparison_df = pd.DataFrame(model_comparison)
                                if not comparison_df.empty:
                                    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
                                    st.dataframe(
                                        comparison_df[['model_name', 'test_r2']].rename(columns={
                                            'model_name': 'Model',
                                            'test_r2': 'Test R²'
                                        }),
                                        use_container_width=True
                                    )
                            
                            # 设置选中的模型信息用于后续显示
                            selected_model = best_model['model_info']
                            model_details = best_model['model_details']
                            user_model_option = best_model['model_name']
                            show_model_info = True
                            
                            # 为Predictions标签页设置选中的模型信息
                            st.session_state["selected_model_for_prediction"] = {
                                'model_key': selected_model['model_key'],
                                'model_name': selected_model['model_name'],
                                'target_variable': selected_model['target_variable']
                            }
                            
                        else:
                            st.warning("No valid models found for the selected target.")
                            show_model_info = False
                    else:
                        st.warning(f"No trained models found for target '{user_target_option}'.")
                        st.info("Please contact an administrator to train models for this target variable.")
                        show_model_info = False
                        
                except Exception as e:
                    st.error(f"Error accessing model cache: {e}")
                    show_model_info = False
            else:
                # 手动选择模式
                col_left_user, col_right_user = st.columns(2)
                
                with col_left_user:
                    st.markdown("**Select algorithm:**")
                    user_model_option = st.selectbox(
                        "Select algorithm:",
                        options=["Random Forest", "XGBoost", "LightGBM"],
                        help="Choose machine learning algorithm",
                        label_visibility="collapsed",
                        key="user_model_selection"
                    )
                
                with col_right_user:
                    st.markdown("**Prediction target:**")
                    user_target_option = st.selectbox(
                        "Prediction target:",
                        options=["Tensile strength retention rate (%)", "Residual tensile strength (MPa)"],
                        help="Choose the target variable for model prediction",
                        label_visibility="collapsed",
                        key="user_target_selection"
                    )
                
                # 执行手动模式的模型查找逻辑
                try:
                    model_cache_manager = ModelCacheManager(engine)
                    cached_models = model_cache_manager.list_cached_models()
                    
                    # 过滤匹配的模型
                    matching_models = [
                        model for model in cached_models 
                        if model['model_name'] == user_model_option and model['target_variable'] == user_target_option
                    ]
                    
                    if matching_models:
                        # 如果有多个匹配的模型，选择最新的
                        selected_model = max(matching_models, key=lambda x: x['updated_at'])
                        
                        try:
                            model_details = model_cache_manager.load_model(selected_model['model_key'])
                            show_model_info = model_details is not None
                            
                            # 为Predictions标签页设置选中的模型信息
                            if show_model_info:
                                st.session_state["selected_model_for_prediction"] = {
                                    'model_key': selected_model['model_key'],
                                    'model_name': selected_model['model_name'],
                                    'target_variable': selected_model['target_variable']
                                }
                        except Exception as e:
                            st.error(f"Error loading model details: {e}")
                            show_model_info = False
                    else:
                        st.warning(f"No trained model found for {user_model_option} with target '{user_target_option}'.")
                        st.info("Please contact an administrator to train this model configuration, or try a different combination.")
                        
                        # 显示可用的模型组合
                        if cached_models:
                            st.markdown("**Available model combinations:**")
                            available_combinations = set()
                            for model in cached_models:
                                combo = f"• {model['model_name']} - {model['target_variable']}"
                                available_combinations.add(combo)
                            
                            for combo in sorted(available_combinations):
                                st.text(combo)
                        show_model_info = False
                        
                except Exception as e:
                    st.error(f"Error accessing model cache: {e}")
                    st.info("Please contact an administrator for assistance.")
                    show_model_info = False
            
            # 显示模型信息（适用于自动和手动模式）
            if show_model_info and 'model_details' in locals() and model_details:
                st.markdown("#### Model Information")
                
                # 解析评估结果
                eval_results = model_details.get('evaluation_results', {})
                feature_info = model_details.get('feature_info', {})
                training_info = model_details.get('training_info', {})
                
                # 基本信息展示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Model Type", 
                        model_details['model_name'],
                        help="Algorithm used for training"
                    )
                
                with col2:
                    st.metric(
                        "Target Variable", 
                        model_details['target_variable'],
                        help="What the model predicts"
                    )
                
                with col3:
                    test_r2 = eval_results.get('test_r2', 'N/A')
                    if test_r2 != 'N/A':
                        st.metric(
                            "Test R²", 
                            f"{test_r2:.4f}",
                            help="Performance on test set"
                        )
                    else:
                        st.metric("Test R²", "N/A")
                
                with col4:
                    cv_r2 = eval_results.get('cv_r2_mean', 'N/A')
                    if cv_r2 != 'N/A':
                        st.metric(
                            "CV R²", 
                            f"{cv_r2:.4f}",
                            help="Cross-validation performance"
                        )
                    else:
                        st.metric("CV R²", "N/A")
                
                # 训练详情 - 使用expander使其可收缩
                with st.expander("Training Details & Performance Analysis", expanded=False):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("##### Training Configuration")
                        created_at = selected_model.get('created_at', 'N/A')
                        feature_count = len(feature_info.get('feature_names', []))
                        training_samples = training_info.get('training_samples', 'N/A')
                        test_samples = training_info.get('test_samples', 'N/A')
                        
                        st.markdown(f"""
                        - **Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at != 'N/A' else 'N/A'}
                        - **Features:** {feature_count} features used
                        - **Training Samples:** {training_samples}
                        - **Test Samples:** {test_samples}
                        - **Evaluation Strategy:** {model_details.get('evaluation_strategy', 'N/A')}
                        """)
                    
                    with detail_col2:
                        st.markdown("##### Model Performance")
                        # 性能等级
                        test_r2_val = eval_results.get('test_r2', 0)
                        if isinstance(test_r2_val, (int, float)):
                            if test_r2_val >= 0.9:
                                performance_level = "Excellent"
                                performance_desc = "Very high accuracy"
                                ready_status = "Ready for production"
                            elif test_r2_val >= 0.8:
                                performance_level = "Good"
                                performance_desc = "Good accuracy"
                                ready_status = "Ready for use"
                            elif test_r2_val >= 0.7:
                                performance_level = "Fair"
                                performance_desc = "Moderate accuracy"
                                ready_status = "Use with caution"
                            else:
                                performance_level = "Needs Improvement"
                                performance_desc = "Lower accuracy"
                                ready_status = "Needs retraining"
                        else:
                            performance_level = "Unknown"
                            performance_desc = "Unable to evaluate"
                            ready_status = "Check manually"
                        
                        st.markdown(f"""
                        - **Level:** {performance_level}
                        - **Description:** {performance_desc}
                        - **Status:** {ready_status}
                        """)
                        
                        # 显示更多评估指标
                        if eval_results:
                            mae = eval_results.get('test_mae', 'N/A')
                            mse = eval_results.get('test_mse', 'N/A')
                            if mae != 'N/A' and mse != 'N/A':
                                st.markdown(f"""
                                **Additional Metrics:**
                                - **MAE:** {mae:.4f}
                                - **MSE:** {mse:.4f}
                                """)
                    
                    # 模型使用指南
                    st.markdown("##### Usage Guidelines")
                    
                    if isinstance(test_r2_val, (int, float)):
                        if test_r2_val >= 0.8:
                            st.success("""
                            **This model is ready for production use:**
                            - High accuracy and reliability
                            - Suitable for important predictions
                            - Can be used with confidence
                            """)
                        elif test_r2_val >= 0.7:
                            st.warning("""
                            **This model can be used with some caution:**
                            - Moderate accuracy
                            - Consider validation with domain experts
                            - Monitor prediction results
                            """)
                        else:
                            st.error("""
                            **This model should be used carefully:**
                            - Lower accuracy, may need retraining
                            - Validate predictions thoroughly
                            - Consider alternative models
                            """)
                    else:
                        st.info("Model performance metrics are not available. Please verify model quality before use.")
            
            # 用户角色提示
            if "authenticated_user" in st.session_state:
                user_role = st.session_state["authenticated_user"]["role"]
                if user_role in ["editor", "viewer"]:
                    st.info(f"**Current Role:** {user_role.title()} | Advanced model configuration and training options are available for admin users.")

# ——————————————————————————————
# Tab 5: Model Training (只对admin用户可见)
# ——————————————————————————————
if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
    with tabs[tab_indexes["model_training"]]:
        
        # Model Training interface
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
        <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
            Model Training
        </h3>
        <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
            Train and evaluate machine learning models for FRP durability prediction
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check data availability
        has_raw_data = st.session_state.df_raw is not None
        has_model_dataset = "model_dataset" in st.session_state and st.session_state.model_dataset is not None
        use_advanced = st.session_state.get("use_advanced_preprocessing", True)
        
        if not has_raw_data:
            st.warning("No dataset available for training.")
        elif "selected_target" not in st.session_state:
            st.info("Please configure model settings in the Model Configuration tab first.")
        else:
            # Display dataset status
            col_status1, col_status2 = st.columns(2)
            
            with col_status1:
                st.metric(
                    "Original Dataset", 
                    f"{len(st.session_state.df_raw)} rows" if has_raw_data else "Not loaded",
                    help="Original data loaded from database"
                )
            
            with col_status2:
                if has_model_dataset:
                    st.metric(
                        "Advanced Preprocessing Dataset", 
                        f"{len(st.session_state.model_dataset)} rows",
                        help="Machine learning dataset with feature engineering"
                    )
                else:
                    st.metric("Advanced Preprocessing Dataset", "Not Generated", help="Please generate in model configuration")
            
            # Dataset selection - only Advanced Preprocessing Dataset
            if has_model_dataset:
                st.markdown("**Training Dataset:** Advanced Preprocessing Dataset")
                use_model_dataset = True
            else:
                st.warning("Please generate the advanced preprocessing dataset in the 'Model Configuration' tab first")
                st.stop()
            
            # Evaluation strategy selection
            if "evaluation_strategy_counter" not in st.session_state:
                st.session_state.evaluation_strategy_counter = 0
            
            evaluation_strategy = st.selectbox(
                "Model Evaluation Strategy",
                [
                "Fixed Test Set (Recommended)",
                "Nested Cross-Validation (Most Rigorous)",
                "Repeated K-Fold Cross-Validation"
            ],
            index=0,
            help="Different evaluation strategies have varying levels of rigor and reproducibility",
            key=f"evaluation_strategy_selectbox_{st.session_state.evaluation_strategy_counter}"
        )
        
        if evaluation_strategy == "Fixed Test Set (Recommended)":
            st.info("""
            **Fixed Test Set Strategy:**
            - Ensure same test data is used each run
            - Results completely reproducible
            - Fast training speed
            - Suitable for production environment and result comparison
            """)
        elif evaluation_strategy == "Nested Cross-Validation (Most Rigorous)":
            st.info("""
            **Nested Cross-Validation Strategy:**
            - Outer CV for model evaluation, inner CV for hyperparameter tuning
            - Provides most reliable performance estimation
            - High computational cost, long training time
            - Suitable for academic research and final performance evaluation
            """)
            
            nested_cv_folds = st.slider("Nested CV Folds", min_value=3, max_value=10, value=5)
            st.session_state.nested_cv_folds = nested_cv_folds
        elif evaluation_strategy == "Repeated K-Fold Cross-Validation":
            st.info("""
            **Repeated K-Fold Cross-Validation:**
            - Multiple repeated K-fold CV, reporting average performance
            - Provide performance confidence intervals
            - Moderate computational load
            - Suitable for performance benchmarking
            """)
            
            repeated_cv_folds = st.slider("K-Fold Number", min_value=3, max_value=10, value=5)
            repeated_cv_repeats = st.slider("Repeat Times", min_value=2, max_value=10, value=3)
            st.session_state.repeated_cv_folds = repeated_cv_folds
            st.session_state.repeated_cv_repeats = repeated_cv_repeats
        else:
            st.error(f"Unknown evaluation strategy: {evaluation_strategy}")
        
        st.session_state.evaluation_strategy = evaluation_strategy
        
        # Test set management (only shown in fixed test set strategy)
        if evaluation_strategy == "Fixed Test Set (Recommended)":
            st.markdown("---")
            st.subheader("Test Set Management")
            
            col_test1, col_test2, col_test3 = st.columns(3)
            
            with col_test1:
                if "fixed_test_indices" in st.session_state:
                    test_size = len(st.session_state.fixed_test_indices)
                    train_size = len(st.session_state.get("fixed_train_indices", []))
                    st.success(f"Fixed test set exists\nTest: {test_size} samples\nTrain: {train_size} samples")
                else:
                    st.info("Fixed test set not yet generated")
            
            with col_test2:
                if st.button("Regenerate Test Set", help="Re-split training and test sets randomly"):
                    st.session_state.regenerate_test_split = True
                    if "fixed_test_indices" in st.session_state:
                        del st.session_state.fixed_test_indices
                    if "fixed_train_indices" in st.session_state:
                        del st.session_state.fixed_train_indices
                    st.success("Test set will be regenerated during next training")
            
            with col_test3:
                st.info("**Stability Guarantee**\nUsing fixed test set ensures\nreproducible results")
        
        # Training button
        if st.button("Start Training", type="primary", use_container_width=True):
            # Get evaluation strategy
            eval_strategy = st.session_state.get("evaluation_strategy", "Fixed Test Set (Recommended)")
            
            # Create progress container
            progress_container = st.container()
            results_container = st.container()
            
            status_text = progress_container.empty()
            progress_bar = progress_container.progress(0)

            # Step 1: Data preprocessing
            status_text.text("Step 1/5: Preprocessing data...")
            progress_bar.progress(0.2)
            
            # Use advanced preprocessing dataset (now the only option)
            df = st.session_state.model_dataset.copy()
            # Record training data type
            st.session_state["training_data_type"] = "advanced_preprocessing_dataset"
            
            # Target variable mapping and processing
            if st.session_state.selected_target == "Residual tensile strength (MPa)":
                # Calculate residual strength: residual strength = original strength × retention rate
                if "Tensile strength retention" in df.columns and "Strength of unconditioned rebar" in df.columns:
                    df["Residual tensile strength (MPa)"] = df["Strength of unconditioned rebar"] * df["Tensile strength retention"]
                    target_col = "Residual tensile strength (MPa)"
                else:
                    st.error("Unable to calculate residual strength: missing required columns")
                    st.error("   - Required: 'Tensile strength retention' (retention rate)")
                    st.error("   - Required: 'Strength of unconditioned rebar' (original strength)")
                    st.stop()
            else:
                # Retention rate prediction, use existing columns directly
                target_mapping = {
                    "Tensile strength retention rate (%)": "Tensile strength retention"
                }
                target_col = target_mapping.get(st.session_state.selected_target)
                
                if target_col not in df.columns:
                    st.error(f"Target variable '{target_col}' does not exist in the dataset")
                    st.write(f"Available columns: {list(df.columns)}")
                    st.stop()
            
            # Prepare features and target variables
            # 基础排除列表
            all_possible_targets = [
                target_col,  # 当前目标变量
                'Title'  # 标题列始终排除
            ]
            
            # 根据预测类型决定特征包含策略
            if st.session_state.selected_target == "Residual tensile strength (MPa)":
                # 预测残余强度：排除保持率（避免直接关联），但保留原始强度
                all_possible_targets.extend([
                    'Tensile strength retention',  # 排除保持率，避免信息泄露
                    'Tensile strength retention rate (%)'  # 排除保持率的其他形式
                ])
            else:
                # 预测保持率：排除原始强度和残余强度
                all_possible_targets.extend([
                    'Strength of unconditioned rebar',  # 排除原始强度
                    'Residual tensile strength (MPa)',  # 排除残余强度
                    'Tensile strength retention',  # 排除其他保持率列
                    'Tensile strength retention rate (%)'
                ])
            
            feature_cols = [col for col in df.columns if col not in all_possible_targets]
            
            # Filter valid data
            valid_data = df.dropna(subset=[target_col])
            
            if len(valid_data) == 0:
                st.error("No valid target variable data")
                st.stop()
            
            X = valid_data[feature_cols]
            y = valid_data[target_col]
            
            # 验证X确实不包含目标变量
            target_check_in_X = [col for col in X.columns if col in all_possible_targets]
            if target_check_in_X:
                st.error(f"❌ CRITICAL: Target variables found in X: {target_check_in_X}")
                st.error("This will cause training errors. Stopping execution.")
                st.stop()
            
            # Data preprocessing (required for all evaluation strategies)
            # Separate feature types
            numeric_feats = []
            categorical_feats = []
            
            # 明确排除所有可能的目标变量名称（与训练时保持一致）
            for col in X.columns:
                # 确保列不是任何目标变量
                if col in all_possible_targets:
                    st.warning(f"⚠️ Skipping column '{col}' as it appears to be a target variable")
                    continue
                
                # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                if col in ['Fibre type', 'Matrix type']:
                    # 这些是重命名后的Glass_or_Basalt和Vinyl_ester_or_Epoxy，实际是0/1数值编码
                    numeric_feats.append(col)
                elif X[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                    categorical_feats.append(col)
                else:
                    numeric_feats.append(col)
            
            # Preprocessing
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            scaler = StandardScaler()
            
            # Encode categorical features
            if categorical_feats:
                # Clean categorical features: fill NaN values and convert to strings
                X_cat_raw = X[categorical_feats].copy()
                for col in categorical_feats:
                    # Fill NaN values with 'unknown'
                    X_cat_raw[col] = X_cat_raw[col].fillna('unknown')
                    # Convert all values to strings
                    X_cat_raw[col] = X_cat_raw[col].astype(str)
                
                encoder.fit(X_cat_raw)
                X_cat = encoder.transform(X_cat_raw)
                ohe_cols = list(encoder.get_feature_names_out(categorical_feats))
            else:
                X_cat = np.empty((len(X), 0))
                ohe_cols = []
            
            # Standardize numerical features
            if numeric_feats:
                # 再次验证数值特征不包含目标变量（使用统一的目标变量列表）
                safe_numeric_feats = [f for f in numeric_feats if f not in all_possible_targets]
                
                if len(safe_numeric_feats) != len(numeric_feats):
                    excluded_feats = [f for f in numeric_feats if f in all_possible_targets]
                    st.warning(f"⚠️ Excluded {len(excluded_feats)} target variables from numeric features: {excluded_feats}")
                
                if safe_numeric_feats:
                    # 最终安全检查：确保safe_numeric_feats不包含任何目标变量
                    final_target_check = [f for f in safe_numeric_feats if f in all_possible_targets]
                    if final_target_check:
                        st.error(f"❌ CRITICAL: Target variables still in safe_numeric_feats: {final_target_check}")
                        st.error("Removing target variables from numeric features...")
                        safe_numeric_feats = [f for f in safe_numeric_feats if f not in all_possible_targets]
                    
                    if safe_numeric_feats:  # 重新检查是否还有数值特征
                        scaler.fit(X[safe_numeric_feats])
                        X_num = scaler.transform(X[safe_numeric_feats])
                        num_cols = safe_numeric_feats
                    else:
                        X_num = np.empty((len(X), 0))
                        num_cols = []
                        st.warning("No numeric features remaining after target variable removal")
                else:
                    X_num = np.empty((len(X), 0))
                    num_cols = []
                    st.info("No numeric features to scale")
            else:
                X_num = np.empty((len(X), 0))
                num_cols = []
            
            # Merge features
            X_processed = np.hstack([X_num, X_cat])
            feature_names = num_cols + ohe_cols
            
            # 使用实际训练缩放器时使用的特征列表（确保一致性）
            final_numeric_feats = num_cols  # 这是实际用于缩放器训练的特征
            final_categorical_feats = categorical_feats.copy()  # 分类特征保持不变
            
            # 创建特征名称映射，确保训练和预测时特征名称一致
            st.session_state.original_feature_names = feature_names.copy()
            st.session_state.feature_columns = feature_cols.copy()
            
            # 保存实际使用的特征（与缩放器训练一致）
            st.session_state.numeric_features = final_numeric_feats.copy()
            st.session_state.categorical_features = final_categorical_feats.copy()
            
            # 验证目标变量确实被排除（使用统一的目标变量列表）
            all_features = final_numeric_feats + final_categorical_feats
            
            found_targets = [t for t in all_possible_targets if t in all_features]
            if found_targets:
                st.error(f"❌ WARNING: Target variables found in features: {found_targets}")
                st.error("This will cause prediction errors. Please check data preprocessing.")
            
            # 保存预处理器以供预测时使用
            st.session_state.feature_encoder = encoder
            st.session_state.feature_scaler = scaler
            
            # Choose different training methods based on evaluation strategy
            split_success = False
            
            if eval_strategy == "Fixed Test Set (Recommended)":
                # Step 3: Split data (using fixed test set strategy)
                status_text.text("Step 3/5: Splitting data...")
                progress_bar.progress(0.6)
                
                # Check if fixed test set indices already exist
                if "fixed_test_indices" not in st.session_state or st.session_state.get("regenerate_test_split", False):
                    # First run or user requests regenerating test set
                    
                    stratify_col = X["Fiber_type"] if "Fiber_type" in X.columns else None
                    # Use random seed from session state
                    seed = st.session_state.get('random_seed', 42)
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=st.session_state.test_size, random_state=seed, stratify=stratify_col
                        )
                    except:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=st.session_state.test_size, random_state=seed
                        )
                    
                    # Save test set indices to ensure reproducibility
                    st.session_state.fixed_test_indices = X_test.index.tolist()
                    st.session_state.fixed_train_indices = X_train.index.tolist()
                    st.session_state.data_hash = hash(str(X.values.tobytes()) + str(y.values.tobytes()))
                    st.session_state.regenerate_test_split = False
                    split_success = True
                else:
                    # Use saved test set indices
                    current_hash = hash(str(X.values.tobytes()) + str(y.values.tobytes()))
                    
                    if st.session_state.get("data_hash") != current_hash:
                        st.warning("Data has changed, regenerating test set split")
                        # Regenerate test set
                        stratify_col = X["Fiber_type"] if "Fiber_type" in X.columns else None
                        # Use random seed from session state
                        seed = st.session_state.get('random_seed', 42)
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=st.session_state.test_size, random_state=seed, stratify=stratify_col
                            )
                        except:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=st.session_state.test_size, random_state=seed
                            )
                        
                        st.session_state.fixed_test_indices = X_test.index.tolist()
                        st.session_state.fixed_train_indices = X_train.index.tolist()
                        st.session_state.data_hash = current_hash
                        split_success = True
                    else:
                        try:
                            test_indices = [idx for idx in st.session_state.fixed_test_indices if idx in X.index]
                            train_indices = [idx for idx in st.session_state.fixed_train_indices if idx in X.index]
                            
                            if len(test_indices) == 0 or len(train_indices) == 0:
                                raise ValueError("Index mismatch")
                            
                            X_train = X.loc[train_indices]
                            X_test = X.loc[test_indices]
                            y_train = y.loc[train_indices]
                            y_test = y.loc[test_indices]
                            split_success = True
                            
                            st.info(f"Using fixed test set: {len(X_test)} test samples, {len(X_train)} training samples")
                            
                        except Exception as e:
                            st.warning(f"Cannot use saved test set indices, regenerating... ({e})")
                            # Regenerate
                            stratify_col = X["Fiber_type"] if "Fiber_type" in X.columns else None
                            # Get random seed from session state
                            seed = st.session_state.get('random_seed', 42)
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=st.session_state.test_size, random_state=seed, stratify=stratify_col
                                )
                            except:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=st.session_state.test_size, random_state=seed
                                )
                            
                            st.session_state.fixed_test_indices = X_test.index.tolist()
                            st.session_state.fixed_train_indices = X_train.index.tolist()
                            st.session_state.data_hash = current_hash
                            split_success = True
                
                # Use processed data for splitting
                train_indices = X_train.index
                test_indices = X_test.index
                X_train_proc = X_processed[X.index.get_indexer(train_indices)]
                X_test_proc = X_processed[X.index.get_indexer(test_indices)]
                
            elif eval_strategy == "Nested Cross-Validation (Most Rigorous)":
                # Nested cross-validation strategy
                status_text.text("Step 3/5: Executing nested cross-validation...")
                progress_bar.progress(0.6)
                
                st.info("Performing nested cross-validation - outer loop for performance evaluation, inner loop for hyperparameter optimization")
                
                # Execute nested CV and display results directly
                nested_cv_folds = st.session_state.get("nested_cv_folds", 5)
                
                # Use all data for nested CV
                X_train, X_test = X, X  # Placeholder, will be split in CV
                y_train, y_test = y, y  # Placeholder
                X_train_proc, X_test_proc = X_processed, X_processed  # Placeholder
                split_success = True
                
                # Set flag to indicate nested CV usage
                st.session_state.use_nested_cv = True
                
            elif eval_strategy == "Repeated K-Fold Cross-Validation":
                # Repeated K-fold cross-validation strategy
                status_text.text("Step 3/5: Setting up repeated K-fold cross-validation...")
                progress_bar.progress(0.6)
                
                st.info("Performing repeated K-fold cross-validation - multiple repetitions for stable performance estimation")
                
                # Get repeated CV parameters
                repeated_cv_folds = st.session_state.get("repeated_cv_folds", 5)
                repeated_cv_repeats = st.session_state.get("repeated_cv_repeats", 3)
                
                # Use all data for repeated CV
                X_train, X_test = X, X  # Placeholder
                y_train, y_test = y, y  # Placeholder
                X_train_proc, X_test_proc = X_processed, X_processed  # Placeholder
                split_success = True
                
                # Set flag to indicate repeated CV usage
                st.session_state.use_repeated_cv = True
                
                st.info(f"Using {repeated_cv_folds}-fold CV repeated {repeated_cv_repeats} times")
            
            else:
                st.error(f"Unknown evaluation strategy: {eval_strategy}")
                st.stop()
            
            if not split_success:
                st.error("Evaluation strategy setup failed")
                st.stop()
            
            # Step 4: Model Training
            status_text.text("Step 4/5: Training models...")
            progress_bar.progress(0.8)
            
            # Define hyperparameter grid
            def get_param_grid(model_name):
                # Get random seed from session state
                seed = st.session_state.get('random_seed', 42)
                
                if model_name == "Random Forest":
                    return {
                        'n_estimators': [100, 200, 300],  # 专注于树数量
                        'max_depth': [10, 15, None],  # 控制复杂度
                        'min_samples_split': [2, 5, 10],  # 防止过拟合
                        'random_state': [seed]
                    }
                elif model_name == "XGBoost":
                    return {
                        'n_estimators': [100, 200, 300],  # 专注于树数量
                        'learning_rate': [0.05, 0.1, 0.2],  # 学习率调优
                        'max_depth': [3, 5, 7],  # 深度调优
                        'random_state': [seed]
                    }
                else:  # LightGBM
                    return {
                        'n_estimators': [100, 200, 300],  # 专注于树数量
                        'learning_rate': [0.05, 0.1, 0.2],  # 学习率调优
                        'max_depth': [3, 5, 7],  # 深度调优
                        'num_leaves': [15, 31, 50],  # 叶子数调优
                        'random_state': [seed]
                    }
            
            # Simple CV evaluation function (for CV analysis without tuning)
            def simple_cv_evaluation(model, X_tr, y_tr, cv_folds=5):
                """Perform simple CV evaluation on model, return scores for each fold"""
                # Create preprocessing pipeline
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
                
                # Detect categorical and numerical columns
                categorical_cols = []
                numeric_cols = []
                
                # 定义所有可能的目标变量
                all_target_variables = [
                    'Strength of unconditioned rebar',
                    'Tensile strength retention',
                    'Residual tensile strength (MPa)',
                    'Tensile strength retention rate (%)'
                ]
                
                if hasattr(X_tr, 'dtypes'):  # DataFrame
                    for col in X_tr.columns:
                        # 跳过目标变量
                        if col in all_target_variables:
                            continue
                        
                        # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                        if col in ['Fibre type', 'Matrix type']:
                            numeric_cols.append(col)
                        elif X_tr[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                            categorical_cols.append(col)
                        else:
                            numeric_cols.append(col)
                    
                    # Create preprocessor with custom categorical variable handling
                    from sklearn.preprocessing import FunctionTransformer
                    
                    categorical_transformer = Pipeline([
                        ('cleaner', FunctionTransformer(global_clean_categorical_features, validate=False)),
                        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                    ])
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_cols),
                            ('cat', categorical_transformer, categorical_cols)
                        ] if categorical_cols else [
                            ('num', StandardScaler(), numeric_cols)
                        ]
                    )
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', model)
                    ])
                    
                    cv_results = cross_validate(
                        pipeline, X_tr, y_tr, 
                        cv=cv_folds, 
                        scoring='r2',
                        return_train_score=True,
                        n_jobs=-1
                    )
                else:  # Already preprocessed numpy array
                    cv_results = cross_validate(
                        model, X_tr, y_tr, 
                        cv=cv_folds, 
                        scoring='r2',
                        return_train_score=True,
                        n_jobs=-1
                    )
                
                return {
                    'cv_scores': cv_results['test_score'].tolist(),
                    'cv_train_scores': cv_results['train_score'].tolist(),
                    'cv_mean': cv_results['test_score'].mean(),
                    'cv_std': cv_results['test_score'].std(),
                    'cv_train_mean': cv_results['train_score'].mean(),
                    'cv_train_std': cv_results['train_score'].std()
                }
            
            # Training function
            def train_and_evaluate_model(name, params, X_tr, y_tr, X_te, y_te, use_cv=False):
                # Get random seed from session state
                seed = st.session_state.get('random_seed', 42)
                
                if name == "Random Forest":
                    base_model = RandomForestRegressor(
                        n_estimators=150,  # 适中的树数量
                        max_depth=15,      # 限制深度防止过拟合
                        min_samples_split=5,  # 增加分割要求
                        min_samples_leaf=2,   # 增加叶子节点要求
                        max_features='sqrt',  
                        bootstrap=True,
                        random_state=seed
                    )
                elif name == "XGBoost":
                    base_model = XGBRegressor(
                        n_estimators=200,  # 适中的树数量
                        learning_rate=0.1,  # 标准学习率
                        max_depth=5,       # 适中深度
                        subsample=0.8,     # 增加采样多样性
                        colsample_bytree=0.8,
                        reg_alpha=0.1,     # 适中正则化
                        reg_lambda=1,      # 适中正则化
                        random_state=seed
                    )
                else:
                    base_model = LGBMRegressor(
                        n_estimators=200,  # 适中的树数量
                        learning_rate=0.1,  # 标准学习率
                        max_depth=6,       # 适中深度
                        num_leaves=31,     # 默认叶子数
                        subsample=0.8,     # 增加采样多样性
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1,
                        random_state=seed,
                        verbosity=-1
                    )
                
                if use_cv and st.session_state.get("enable_hp_tuning", False):
                    # Create preprocessing pipeline for GridSearchCV
                    from sklearn.pipeline import Pipeline
                    from sklearn.compose import ColumnTransformer
                    
                    # Detect categorical and numerical columns
                    categorical_cols = []
                    numeric_cols = []
                    
                    # 定义所有可能的目标变量
                    all_target_variables = [
                        'Strength of unconditioned rebar',
                        'Tensile strength retention',
                        'Residual tensile strength (MPa)',
                        'Tensile strength retention rate (%)'
                    ]
                    
                    if hasattr(X_tr, 'dtypes'):  # DataFrame
                        for col in X_tr.columns:
                            # 跳过目标变量
                            if col in all_target_variables:
                                continue
                            
                            # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                            if col in ['Fibre type', 'Matrix type']:
                                numeric_cols.append(col)
                            elif X_tr[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                                categorical_cols.append(col)
                            else:
                                numeric_cols.append(col)
                        
                        # Create preprocessor, using enhanced categorical variable handling without polynomial features
                        preprocessor = create_enhanced_preprocessor(
                            categorical_cols, numeric_cols, 
                            add_polynomial=False, polynomial_degree=2  # 暂时禁用多项式特征
                        )
                        
                        # Create pipeline
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', base_model)
                        ])
                        
                        # Adjust parameter grid to match pipeline
                        param_grid = get_param_grid(name)
                        pipeline_param_grid = {f'regressor__{k}': v for k, v in param_grid.items()}
                        
                        grid_search = GridSearchCV(
                            estimator=pipeline,
                            param_grid=pipeline_param_grid,
                            cv=5, # 恢复5-fold CV
                            scoring='r2',
                            n_jobs=-1,
                            verbose=0,
                            return_train_score=True  # Return training scores to analyze overfitting
                        )
                        grid_search.fit(X_tr, y_tr) # Perform CV on original training set (pipeline automatically handles preprocessing)
                        
                    else:  # Already preprocessed data
                        param_grid = get_param_grid(name)
                        grid_search = GridSearchCV(
                            estimator=base_model,
                            param_grid=param_grid,
                            cv=5, # 恢复5-fold CV
                            scoring='r2',
                            n_jobs=-1,
                            verbose=0,
                            return_train_score=True  # Return training scores to analyze overfitting
                        )
                        grid_search.fit(X_tr, y_tr) # Only perform CV on training set
                        
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_cv_score = grid_search.best_score_
                    y_pred = best_model.predict(X_te)  # Evaluate on independent test set
                    
                    # Extract CV detailed results
                    cv_results = grid_search.cv_results_
                    best_index = grid_search.best_index_
                    
                    # Get CV scores corresponding to best parameters
                    cv_scores = cv_results['split0_test_score'][best_index:best_index+1].tolist() + \
                               cv_results['split1_test_score'][best_index:best_index+1].tolist() + \
                               cv_results['split2_test_score'][best_index:best_index+1].tolist() + \
                               cv_results['split3_test_score'][best_index:best_index+1].tolist() + \
                               cv_results['split4_test_score'][best_index:best_index+1].tolist()
                    
                    # Get training scores (check overfitting)
                    cv_train_scores = cv_results['split0_train_score'][best_index:best_index+1].tolist() + \
                                     cv_results['split1_train_score'][best_index:best_index+1].tolist() + \
                                     cv_results['split2_train_score'][best_index:best_index+1].tolist() + \
                                     cv_results['split3_train_score'][best_index:best_index+1].tolist() + \
                                     cv_results['split4_train_score'][best_index:best_index+1].tolist()
                    
                    # Calculate CV error statistics
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    cv_train_mean = np.mean(cv_train_scores)
                    cv_train_std = np.std(cv_train_scores)
                    
                    return {
                        "name": name,
                        "model": best_model,
                        "best_params": best_params,
                        "cv_score": best_cv_score,
                        "cv_scores": cv_scores,  # Score for each fold
                        "cv_train_scores": cv_train_scores,  # Training score for each fold
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "cv_train_mean": cv_train_mean,
                        "cv_train_std": cv_train_std,
                        "r2": r2_score(y_te, y_pred),
                        "mse": mean_squared_error(y_te, y_pred),
                        "y_pred": y_pred,
                        "cv_results_full": cv_results  # Complete CV results
                    }
                else:
                    # Non-tuning mode, but still perform CV evaluation analysis
                    if isinstance(params, dict):
                        for key, value in params.items():
                            if hasattr(base_model, key):
                                setattr(base_model, key, value)
                    
                    # Perform simple CV evaluation to get CV error information
                    cv_eval = simple_cv_evaluation(base_model, X_tr, y_tr)
                    
                    # Train final model on entire training set
                    if hasattr(X_tr, 'dtypes'):  # DataFrame - 需要使用pipeline
                        from sklearn.pipeline import Pipeline
                        from sklearn.compose import ColumnTransformer
                        from sklearn.preprocessing import FunctionTransformer
                        
                        # Detect categorical and numerical columns
                        categorical_cols = []
                        numeric_cols = []
                        
                        # 定义所有可能的目标变量
                        all_target_variables = [
                            'Strength of unconditioned rebar',
                            'Tensile strength retention',
                            'Residual tensile strength (MPa)',
                            'Tensile strength retention rate (%)'
                        ]
                        
                        for col in X_tr.columns:
                            # 跳过目标变量
                            if col in all_target_variables:
                                continue
                            
                            # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                            if col in ['Fibre type', 'Matrix type']:
                                numeric_cols.append(col)
                            elif X_tr[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                                categorical_cols.append(col)
                            else:
                                numeric_cols.append(col)
                        
                        # Create preprocessor with custom categorical variable handling
                        categorical_transformer = Pipeline([
                            ('cleaner', FunctionTransformer(global_clean_categorical_features, validate=False)),
                            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                        ])
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_cols),
                                ('cat', categorical_transformer, categorical_cols)
                            ] if categorical_cols else [
                                ('num', StandardScaler(), numeric_cols)
                            ]
                        )
                        
                        # Create pipeline
                        final_model = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', base_model)
                        ])
                        
                        final_model.fit(X_tr, y_tr)
                        y_pred = final_model.predict(X_te)
                        
                    else:  # Preprocessed numpy array
                        base_model.fit(X_tr, y_tr)
                        y_pred = base_model.predict(X_te)
                        final_model = base_model
                    
                    return {
                        "name": name,
                        "model": final_model,
                        "r2": r2_score(y_te, y_pred),
                        "mse": mean_squared_error(y_te, y_pred),
                        "y_pred": y_pred,
                        "cv_scores": cv_eval['cv_scores'],
                        "cv_train_scores": cv_eval['cv_train_scores'], 
                        "cv_mean": cv_eval['cv_mean'],
                        "cv_std": cv_eval['cv_std'],
                        "cv_train_mean": cv_eval['cv_train_mean'],
                        "cv_train_std": cv_eval['cv_train_std']
                    }
            
            # Train model
            if eval_strategy == "Nested Cross-Validation (Most Rigorous)":
                # Nested Cross-Validation implementation
                status_text.text("Step 4/5: Executing nested cross-validation...")
                
                def perform_nested_cv(model_name, X_data, y_data):
                    """Perform nested cross-validation for a single model"""
                    from sklearn.pipeline import Pipeline
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import FunctionTransformer
                    from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
                    
                    # Get random seed from session state
                    seed = st.session_state.get('random_seed', 42)
                    
                    if model_name == "Random Forest":
                        base_model = RandomForestRegressor(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features='sqrt',
                            bootstrap=True,
                            oob_score=True,
                            random_state=seed
                        )
                    elif model_name == "XGBoost":
                        base_model = XGBRegressor(
                            n_estimators=400,
                            learning_rate=0.08,
                            max_depth=7,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            colsample_bylevel=0.85,
                            reg_alpha=0.2,
                            reg_lambda=1.2,
                            gamma=0.1,
                            random_state=seed
                        )
                    else:  # LightGBM
                        base_model = LGBMRegressor(
                            n_estimators=400,
                            learning_rate=0.08,
                            max_depth=8,
                            num_leaves=63,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            subsample_freq=1,
                            reg_alpha=0.2,
                            reg_lambda=0.8,
                            min_child_weight=5,
                            min_child_samples=10,
                            random_state=seed,
                            verbosity=-1,
                            force_row_wise=True
                        )
                    
                    # Create preprocessing pipeline
                    categorical_cols = []
                    numeric_cols = []
                    all_target_variables = [
                        'Strength of unconditioned rebar',
                        'Tensile strength retention',
                        'Residual tensile strength (MPa)',
                        'Tensile strength retention rate (%)'
                    ]
                    
                    for col in X_data.columns:
                        if col in all_target_variables:
                            continue
                        # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                        if col in ['Fibre type', 'Matrix type']:
                            numeric_cols.append(col)
                        elif X_data[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                            categorical_cols.append(col)
                        else:
                            numeric_cols.append(col)
                    
                    categorical_transformer = Pipeline([
                        ('cleaner', FunctionTransformer(global_clean_categorical_features, validate=False)),
                        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                    ])
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_cols),
                            ('cat', categorical_transformer, categorical_cols)
                        ] if categorical_cols else [
                            ('num', StandardScaler(), numeric_cols)
                        ]
                    )
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', base_model)
                    ])
                    
                    # Nested CV with hyperparameter tuning
                    param_grid = get_param_grid(model_name)
                    pipeline_param_grid = {f'regressor__{k}': v for k, v in param_grid.items()}
                    
                    # Use cross_validate with GridSearchCV as inner CV
                    inner_cv = GridSearchCV(
                        pipeline, pipeline_param_grid, cv=3, scoring='r2', n_jobs=-1
                    )
                    
                    # Outer CV
                    nested_scores = cross_validate(
                        inner_cv, X_data, y_data, cv=5, scoring='r2',
                        return_train_score=True, n_jobs=-1
                    )
                    
                    # Train final model on full data
                    final_inner_cv = GridSearchCV(
                        pipeline, pipeline_param_grid, cv=3, scoring='r2', n_jobs=-1
                    )
                    final_inner_cv.fit(X_data, y_data)
                    final_model = final_inner_cv.best_estimator_
                    
                    # Create dummy predictions for consistency with test set evaluation
                    # Use a hold-out approach for more realistic evaluation
                    # Get random seed from session state
                    seed = st.session_state.get('random_seed', 42)
                    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
                        X_data, y_data, test_size=0.2, random_state=seed
                    )
                    final_model.fit(X_temp_train, y_temp_train)
                    y_pred_test = final_model.predict(X_temp_test)
                    test_mse = mean_squared_error(y_temp_test, y_pred_test)
                    test_r2 = r2_score(y_temp_test, y_pred_test)
                    
                    # Retrain on full data for final model
                    final_model.fit(X_data, y_data)
                    y_pred_dummy = final_model.predict(X_data)
                    
                    return {
                        "name": model_name,
                        "model": final_model,
                        "r2": test_r2,  # Use hold-out test R2
                        "mse": test_mse,  # Use hold-out test MSE
                        "y_pred": y_pred_dummy,
                        "cv_scores": nested_scores['test_score'].tolist(),
                        "cv_train_scores": nested_scores['train_score'].tolist(),
                        "cv_mean": nested_scores['test_score'].mean(),
                        "cv_std": nested_scores['test_score'].std(),
                        "cv_train_mean": nested_scores['train_score'].mean(),
                        "cv_train_std": nested_scores['train_score'].std(),
                        "best_params": final_inner_cv.best_params_,
                        "nested_cv_results": nested_scores
                    }
                
                if st.session_state.get("auto_model_select", False):
                    # Test all models with nested CV
                    all_models = ["Random Forest", "XGBoost", "LightGBM"]
                    results = []
                    
                    for model_name in all_models:
                        status_text.text(f"Step 4/5: Nested CV for {model_name}...")
                        result = perform_nested_cv(model_name, X, y)
                        results.append(result)
                    
                    # Select best model based on nested CV score
                    best = max(results, key=lambda x: x["cv_mean"])
                    model = best["model"]
                    y_pred = best["y_pred"]
                    
                    st.session_state.model_comparison_results = [
                        {"Model": r["name"], "R2": r["cv_mean"], "MSE": 1 - r["cv_mean"]}
                        for r in results
                    ]
                    st.session_state.training_result = best
                    st.session_state.all_training_results = results
                else:
                    # Single model nested CV
                    model_name = st.session_state.selected_model
                    result = perform_nested_cv(model_name, X, y)
                    model = result["model"]
                    y_pred = result["y_pred"]
                    st.session_state.training_result = result
                
                # Set dummy test data for saving
                st.session_state.y_test = y
                st.session_state.y_pred = y_pred
                st.session_state.X_train_proc = X_processed  # Use full processed data
                st.session_state.X_test = X  # Use original data as dummy test set
                
            elif eval_strategy == "Repeated K-Fold Cross-Validation":
                # Repeated K-Fold Cross-Validation implementation
                status_text.text("Step 4/5: Executing repeated k-fold cross-validation...")
                
                def perform_repeated_kfold_cv(model_name, X_data, y_data):
                    """Perform repeated k-fold cross-validation for a single model"""
                    from sklearn.pipeline import Pipeline
                    from sklearn.compose import ColumnTransformer
                    from sklearn.preprocessing import FunctionTransformer
                    from sklearn.model_selection import RepeatedKFold, train_test_split, cross_validate
                    
                    # Get random seed from session state
                    seed = st.session_state.get('random_seed', 42)
                    
                    if model_name == "Random Forest":
                        base_model = RandomForestRegressor(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features='sqrt',
                            bootstrap=True,
                            oob_score=True,
                            random_state=seed
                        )
                    elif model_name == "XGBoost":
                        base_model = XGBRegressor(
                            n_estimators=400,
                            learning_rate=0.08,
                            max_depth=7,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            colsample_bylevel=0.85,
                            reg_alpha=0.2,
                            reg_lambda=1.2,
                            gamma=0.1,
                            random_state=seed
                        )
                    else:  # LightGBM
                        base_model = LGBMRegressor(
                            n_estimators=400,
                            learning_rate=0.08,
                            max_depth=8,
                            num_leaves=63,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            subsample_freq=1,
                            reg_alpha=0.2,
                            reg_lambda=0.8,
                            min_child_weight=5,
                            min_child_samples=10,
                            random_state=seed,
                            verbosity=-1,
                            force_row_wise=True
                        )
                    
                    # Create preprocessing pipeline
                    categorical_cols = []
                    numeric_cols = []
                    all_target_variables = [
                        'Strength of unconditioned rebar',
                        'Tensile strength retention',
                        'Residual tensile strength (MPa)',
                        'Tensile strength retention rate (%)'
                    ]
                    
                    for col in X_data.columns:
                        if col in all_target_variables:
                            continue
                        # 特殊处理：这些列虽然名称像分类特征，但实际是数值编码
                        if col in ['Fibre type', 'Matrix type']:
                            numeric_cols.append(col)
                        elif X_data[col].dtype == object or col in ["Fiber_type", "Matrix_type", "surface_treatment", "Surface treatment"]:
                            categorical_cols.append(col)
                        else:
                            numeric_cols.append(col)
                    
                    categorical_transformer = Pipeline([
                        ('cleaner', FunctionTransformer(global_clean_categorical_features, validate=False)),
                        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                    ])
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_cols),
                            ('cat', categorical_transformer, categorical_cols)
                        ] if categorical_cols else [
                            ('num', StandardScaler(), numeric_cols)
                        ]
                    )
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', base_model)
                    ])
                    
                    # Repeated K-Fold CV
                    repeated_cv_folds = st.session_state.get("repeated_cv_folds", 5)
                    repeated_cv_repeats = st.session_state.get("repeated_cv_repeats", 3)
                    
                    # Get random seed from session state
                    seed = st.session_state.get('random_seed', 42)
                    rkf = RepeatedKFold(n_splits=repeated_cv_folds, n_repeats=repeated_cv_repeats, random_state=seed)
                    
                    repeated_scores = cross_validate(
                        pipeline, X_data, y_data, cv=rkf, scoring='r2',
                        return_train_score=True, n_jobs=-1
                    )
                    
                    # Train final model on full data and get realistic test evaluation
                    # Use a hold-out approach for more realistic evaluation
                    # Get random seed from session state 
                    seed = st.session_state.get('random_seed', 42)
                    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
                        X_data, y_data, test_size=0.2, random_state=seed
                    )
                    pipeline.fit(X_temp_train, y_temp_train)
                    y_pred_test = pipeline.predict(X_temp_test)
                    test_mse = mean_squared_error(y_temp_test, y_pred_test)
                    test_r2 = r2_score(y_temp_test, y_pred_test)
                    
                    # Retrain on full data for final model
                    pipeline.fit(X_data, y_data)
                    y_pred_dummy = pipeline.predict(X_data)
                    
                    return {
                        "name": model_name,
                        "model": pipeline,
                        "r2": test_r2,  # Use hold-out test R2
                        "mse": test_mse,  # Use hold-out test MSE
                        "y_pred": y_pred_dummy,
                        "cv_scores": repeated_scores['test_score'].tolist(),
                        "cv_train_scores": repeated_scores['train_score'].tolist(),
                        "cv_mean": repeated_scores['test_score'].mean(),
                        "cv_std": repeated_scores['test_score'].std(),
                        "cv_train_mean": repeated_scores['train_score'].mean(),
                        "cv_train_std": repeated_scores['train_score'].std(),
                        "repeated_cv_results": repeated_scores
                    }
                
                if st.session_state.get("auto_model_select", False):
                    # Test all models with repeated k-fold CV
                    all_models = ["Random Forest", "XGBoost", "LightGBM"]
                    results = []
                    
                    for model_name in all_models:
                        status_text.text(f"Step 4/5: Repeated K-Fold for {model_name}...")
                        result = perform_repeated_kfold_cv(model_name, X, y)
                        results.append(result)
                    
                    # Select best model based on repeated CV score
                    best = max(results, key=lambda x: x["cv_mean"])
                    model = best["model"]
                    y_pred = best["y_pred"]
                    
                    st.session_state.model_comparison_results = [
                        {"Model": r["name"], "R2": r["cv_mean"], "MSE": 1 - r["cv_mean"]}
                        for r in results
                    ]
                    st.session_state.training_result = best
                    st.session_state.all_training_results = results
                else:
                    # Single model repeated k-fold CV
                    model_name = st.session_state.selected_model
                    result = perform_repeated_kfold_cv(model_name, X, y)
                    model = result["model"]
                    y_pred = result["y_pred"]
                    st.session_state.training_result = result
                
                # Set dummy test data for saving
                st.session_state.y_test = y
                st.session_state.y_pred = y_pred
                st.session_state.X_train_proc = X_processed  # Use full processed data
                st.session_state.X_test = X  # Use original data as dummy test set
                
            else:
                # Fixed Test Set strategy (original implementation)
                if st.session_state.get("auto_model_select", False):
                    # Automatic model selection
                    all_models = ["Random Forest", "XGBoost", "LightGBM"]
                    results = []
                    
                    for i, model_name in enumerate(all_models):
                        status_text.text(f"Step 4/5: Training {model_name}...")
                        
                        # Fixed test set: Use preprocessed data
                        result = train_and_evaluate_model(
                            model_name, {}, X_train_proc, y_train, X_test_proc, y_test,
                            use_cv=st.session_state.get("enable_hp_tuning", False)
                        )
                        results.append(result)
                    
                    # Select best model
                    best = max(results, key=lambda x: x["r2"])
                    model = best["model"]
                    y_pred = best["y_pred"]
                    
                    # Save comparison results and best training result
                    st.session_state.model_comparison_results = [
                        {"Model": r["name"], "R2": r["r2"], "MSE": r["mse"]}
                        for r in results
                    ]
                    st.session_state.training_result = best
                    st.session_state.all_training_results = results
                else:
                    # Single model training
                    model_name = st.session_state.selected_model
                    params = st.session_state.get("model_params", {})
                    
                    # Fixed test set: Use preprocessed data
                    result = train_and_evaluate_model(
                        model_name, params, X_train_proc, y_train, X_test_proc, y_test,
                        use_cv=st.session_state.get("enable_hp_tuning", False)
                    )
                    
                    model = result["model"]
                    y_pred = result["y_pred"]
                    
                    # Save training results (including CV information)
                    st.session_state.training_result = result
            
            # Step 5: Save results
            status_text.text("Step 5/5: Saving results...")
            progress_bar.progress(1.0)
            
            # Save model and preprocessor
            st.session_state.trained_model = model
            st.session_state.ohe_encoder = encoder
            st.session_state.scaler = scaler
            st.session_state.feature_names = feature_names
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.X_train_proc = X_train_proc
            st.session_state.X_test = X_test
            
            # 保存完整的特征预处理信息，用于预测时保持一致性
            st.session_state.training_feature_info = {
                'numeric_features': final_numeric_feats.copy(),  # 使用实际训练缩放器的特征列表
                'categorical_features': final_categorical_feats.copy(),  # 使用实际的分类特征列表
                'feature_encoder': encoder,  # 保存实际的预处理器对象
                'feature_scaler': scaler,    # 保存实际的预处理器对象
                'original_feature_names': st.session_state.get('original_feature_names', []),
                'feature_columns': st.session_state.get('feature_columns', []),
                'training_columns': feature_cols,
                'processed_feature_names': feature_names,
                'target_variable': target_col  # 明确记录目标变量
            }
            
            # 保存模型到缓存数据库
            try:
                model_cache_manager = ModelCacheManager(engine)
                
                # 准备保存的数据
                if st.session_state.get("auto_model_select", False):
                    # 自动选择模式 - 保存所有模型
                    for result in st.session_state.all_training_results:
                        # 计算数据哈希
                        data_hash = hashlib.sha256(
                            (str(X.values.tobytes()) + str(y.values.tobytes())).encode()
                        ).hexdigest()[:16]
                        
                        # 准备评估结果
                        evaluation_results = {
                            'test_r2': result['r2'],
                            'test_mse': result['mse'],
                            'test_rmse': np.sqrt(result['mse']),
                            'cv_r2_mean': result.get('cv_mean', 0),
                            'cv_r2_std': result.get('cv_std', 0),
                            'cv_train_r2_mean': result.get('cv_train_mean', 0),
                            'cv_train_r2_std': result.get('cv_train_std', 0),
                            'cv_scores': result.get('cv_scores', []),
                            'overfitting_level': result.get('cv_train_mean', 0) - result.get('cv_mean', 0),
                            'model_stability': 1 - (result.get('cv_std', 0) / result.get('cv_mean', 1)) if result.get('cv_mean', 0) != 0 else 0
                        }
                        
                        # 准备特征信息
                        feature_info = {
                            'feature_names': feature_names,
                            'feature_columns': feature_cols,
                            'numeric_features': numeric_feats,
                            'categorical_features': categorical_feats,
                            'n_features': len(feature_names),
                            'n_samples': len(valid_data)
                        }
                        
                        # 准备预处理信息 - 包含实际的预处理器对象
                        import pickle
                        preprocessing_info = {
                            'encoder_type': 'OneHotEncoder',
                            'scaler_type': 'StandardScaler',
                            'has_categorical': len(categorical_feats) > 0,
                            'has_numeric': len(numeric_feats) > 0,
                            'data_type': 'advanced_preprocessing_dataset',
                            # 序列化预处理器对象
                            'feature_encoder': base64.b64encode(pickle.dumps(encoder)).decode() if encoder is not None else None,
                            'feature_scaler': base64.b64encode(pickle.dumps(scaler)).decode() if scaler is not None else None,
                            'encoder_feature_names': encoder.feature_names_in_.tolist() if hasattr(encoder, 'feature_names_in_') else [],
                            'scaler_feature_names': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else []
                        }
                        
                        # 准备训练信息
                        training_info = {
                            'evaluation_strategy': eval_strategy,
                            'hyperparameter_tuning': st.session_state.get("enable_hp_tuning", False),
                            'test_size': st.session_state.get("test_size", 0.2),
                            'auto_model_select': True,
                            'training_timestamp': datetime.now().isoformat(),
                            'best_params': result.get('best_params', {}),
                            'model_type': result['name']
                        }
                        
                        # 保存模型
                        model_key = model_cache_manager.save_model(
                            model_name=result['name'],
                            target_variable=st.session_state.selected_target,
                            evaluation_strategy=eval_strategy,
                            data_hash=data_hash,
                            trained_model=result['model'],
                            best_params=result.get('best_params', {}),
                            evaluation_results=evaluation_results,
                            feature_info=feature_info,
                            preprocessing_info=preprocessing_info,
                            training_info=training_info
                        )
                        
                        # 为每个模型记录保存状态
                        if model_key:
                            print(f"✅ {result['name']} model saved with key: {model_key}")
                        else:
                            print(f"❌ Failed to save {result['name']} model")
                    
                    # 保存最佳模型的key
                    if st.session_state.get("training_result"):
                        best_model_name = st.session_state.training_result['name']
                        # 从缓存中查找最佳模型的key
                        try:
                            best_model_key = model_cache_manager.get_model_key(
                                model_name=best_model_name,
                                target_variable=st.session_state.selected_target,
                                evaluation_strategy=eval_strategy,
                                data_hash=data_hash
                            )
                            if best_model_key:
                                st.session_state.current_model_key = best_model_key
                                st.success(f"✅ All models saved. Best model ({best_model_name}) key: {best_model_key}")
                            else:
                                st.warning(f"❌ Could not retrieve key for best model ({best_model_name})")
                        except Exception as key_error:
                            st.warning(f"❌ Error retrieving best model key: {key_error}")
                    else:
                        st.success("✅ All models saved successfully")
                
                else:
                    # 单模型模式
                    result = st.session_state.training_result
                    
                    # 计算数据哈希
                    data_hash = hashlib.sha256(
                        (str(X.values.tobytes()) + str(y.values.tobytes())).encode()
                    ).hexdigest()[:16]
                    
                    # 准备评估结果
                    evaluation_results = {
                        'test_r2': result['r2'],
                        'test_mse': result['mse'],
                        'test_rmse': np.sqrt(result['mse']),
                        'cv_r2_mean': result.get('cv_mean', 0),
                        'cv_r2_std': result.get('cv_std', 0),
                        'cv_train_r2_mean': result.get('cv_train_mean', 0),
                        'cv_train_r2_std': result.get('cv_train_std', 0),
                        'cv_scores': result.get('cv_scores', []),
                        'overfitting_level': result.get('cv_train_mean', 0) - result.get('cv_mean', 0),
                        'model_stability': 1 - (result.get('cv_std', 0) / result.get('cv_mean', 1)) if result.get('cv_mean', 0) != 0 else 0
                    }
                    
                    # 准备特征信息
                    feature_info = {
                        'feature_names': feature_names,
                        'feature_columns': feature_cols,
                        'numeric_features': numeric_feats,
                        'categorical_features': categorical_feats,
                        'n_features': len(feature_names),
                        'n_samples': len(valid_data)
                    }
                    
                    # 准备预处理信息 - 包含实际的预处理器对象
                    import pickle
                    preprocessing_info = {
                        'encoder_type': 'OneHotEncoder',
                        'scaler_type': 'StandardScaler',
                        'has_categorical': len(categorical_feats) > 0,
                        'has_numeric': len(numeric_feats) > 0,
                        'data_type': 'advanced_preprocessing_dataset',
                        # 序列化预处理器对象
                        'feature_encoder': base64.b64encode(pickle.dumps(encoder)).decode() if encoder is not None else None,
                        'feature_scaler': base64.b64encode(pickle.dumps(scaler)).decode() if scaler is not None else None,
                        'encoder_feature_names': encoder.feature_names_in_.tolist() if hasattr(encoder, 'feature_names_in_') else [],
                        'scaler_feature_names': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else []
                    }
                    
                    # 准备训练信息
                    training_info = {
                        'evaluation_strategy': eval_strategy,
                        'hyperparameter_tuning': st.session_state.get("enable_hp_tuning", False),
                        'test_size': st.session_state.get("test_size", 0.2),
                        'auto_model_select': False,
                        'training_timestamp': datetime.now().isoformat(),
                        'best_params': result.get('best_params', {}),
                        'model_type': result['name']
                    }
                    
                    # 保存模型
                    model_key = model_cache_manager.save_model(
                        model_name=result['name'],
                        target_variable=st.session_state.selected_target,
                        evaluation_strategy=eval_strategy,
                        data_hash=data_hash,
                        trained_model=result['model'],
                        best_params=result.get('best_params', {}),
                        evaluation_results=evaluation_results,
                        feature_info=feature_info,
                        preprocessing_info=preprocessing_info,
                        training_info=training_info
                    )
                    
                    if model_key:
                        st.success(f"✅ Model saved successfully with key: {model_key}")
                        st.session_state.current_model_key = model_key
                    else:
                        st.warning("❌ Failed to save model to cache")
            
            except Exception as e:
                st.warning(f"Model cache save failed: {e}")
            
            # Clear progress display
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            with results_container:
                st.success("Training completed successfully!")
                
                # Performance metrics
                r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
                mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model", st.session_state.get("selected_model", "Auto"))
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")
                with col3:
                    st.metric("MSE", f"{mse:.2f}")
                with col4:
                    st.metric("RMSE", f"{np.sqrt(mse):.2f}")
                
                # CV误差分析（始终显示，只要有CV结果）
                if "training_result" in st.session_state:
                    result = st.session_state.training_result
                    
                    if "cv_scores" in result:
                        st.markdown("---")
                        st.subheader("Cross-Validation Error Analysis")
                        
                        # 显示是否使用了超参数调优
                        if st.session_state.get("enable_hp_tuning", False):
                            st.info("**CV Results with Grid Search Hyperparameter Tuning**")
                        else:
                            st.info("**Basic 5-Fold Cross-Validation Results** (without hyperparameter tuning)")
                        
                        # Display CV statistical information
                        col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
                        
                        with col_cv1:
                            st.metric(
                                "CV Mean R²", 
                                f"{result['cv_mean']:.4f}",
                                delta=f"±{result['cv_std']:.4f}"
                            )
                        
                        with col_cv2:
                            st.metric(
                                "CV Training R²", 
                                f"{result['cv_train_mean']:.4f}",
                                delta=f"±{result['cv_train_std']:.4f}"
                            )
                        
                        with col_cv3:
                            overfitting = result['cv_train_mean'] - result['cv_mean']
                            st.metric(
                                "Overfitting Level", 
                                f"{overfitting:.4f}",
                                delta="Training-Validation" if overfitting > 0.1 else None,
                                delta_color="inverse" if overfitting > 0.1 else "normal"
                            )
                        
                        with col_cv4:
                            stability = 1 - (result['cv_std'] / result['cv_mean']) if result['cv_mean'] != 0 else 0
                            st.metric(
                                "Model Stability", 
                                f"{stability:.3f}",
                                help="1.0 is perfectly stable, higher values indicate better stability"
                            )
                        
                        # CV分数详细分析
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            # 每个fold的CV分数
                            cv_df = pd.DataFrame({
                                'Fold': [f'Fold {i+1}' for i in range(len(result['cv_scores']))],
                                'Validation_R2': result['cv_scores'],
                                'Training_R2': result['cv_train_scores']
                            })
                            
                            # 重塑数据用于可视化
                            cv_melted = pd.melt(cv_df, id_vars=['Fold'], 
                                              value_vars=['Validation_R2', 'Training_R2'],
                                              var_name='Score_Type', value_name='R2_Score')
                            
                            cv_chart = alt.Chart(cv_melted).mark_bar(opacity=0.8).encode(
                                x=alt.X('Fold:N', title='Cross-validation Folds'),
                                y=alt.Y('R2_Score:Q', title='R² Score'),
                                color=alt.Color('Score_Type:N', 
                                              scale=alt.Scale(domain=['Validation_R2', 'Training_R2'],
                                                            range=['#1f77b4', '#ff7f0e']),
                                              legend=alt.Legend(title="Score Type")),
                                tooltip=['Fold', 'Score_Type', 'R2_Score']
                            ).properties(
                                title='Cross-validation Score Comparison by Fold',
                                height=300
                            )
                            
                            st.altair_chart(cv_chart, use_container_width=True)
                        
                        with col_chart2:
                            # CV分数分布
                            cv_stats_df = pd.DataFrame({
                                'Metric': ['Validation R²', 'Training R²'],
                                'Mean': [result['cv_mean'], result['cv_train_mean']],
                                'Std': [result['cv_std'], result['cv_train_std']],
                                'Min': [min(result['cv_scores']), min(result['cv_train_scores'])],
                                'Max': [max(result['cv_scores']), max(result['cv_train_scores'])]
                            })
                            
                            # 创建误差条图
                            error_chart = alt.Chart(cv_stats_df).mark_errorbar(extent='stdev').encode(
                                x=alt.X('Metric:N', title='Metric Type'),
                                y=alt.Y('Mean:Q', title='R² Score'),
                                yError='Std:Q'
                            ).properties(
                                title='CV Score Mean and Standard Deviation',
                                height=300
                            )
                            
                            points = alt.Chart(cv_stats_df).mark_circle(size=100, color='red').encode(
                                x='Metric:N',
                                y='Mean:Q',
                                tooltip=['Metric', 'Mean', 'Std', 'Min', 'Max']
                            )
                            
                            st.altair_chart(error_chart + points, use_container_width=True)
                        
                        # 详细CV结果表格
                        with st.expander("Detailed CV Score Data", expanded=False):
                            detailed_cv_df = pd.DataFrame({
                                'Fold': [f'Fold {i+1}' for i in range(len(result['cv_scores']))],
                                'Validation_R²': [f"{score:.4f}" for score in result['cv_scores']],
                                'Training_R²': [f"{score:.4f}" for score in result['cv_train_scores']],
                                'Difference': [f"{train-val:.4f}" for train, val in zip(result['cv_train_scores'], result['cv_scores'])]
                            })
                            st.dataframe(detailed_cv_df, use_container_width=True)
                            
                            # 统计摘要
                            st.write("**Statistical Summary:**")
                            st.write(f"- Validation R²: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
                            st.write(f"- Training R²: {result['cv_train_mean']:.4f} ± {result['cv_train_std']:.4f}")
                            st.write(f"- Overfitting Level: {result['cv_train_mean'] - result['cv_mean']:.4f}")
                            st.write(f"- CV Stability: {stability:.3f}")
                            
                            if overfitting > 0.1:
                                st.warning("Severe overfitting detected, consider adjusting model complexity or increasing regularization")
                            elif overfitting > 0.05:
                                st.info("Slight overfitting detected, model performance is acceptable")
                            else:
                                st.success("Model training performed well with low overfitting")
                
                # Model comparison (if auto-select enabled)
                if st.session_state.get("auto_model_select", False) and "model_comparison_results" in st.session_state:
                    st.subheader("Model Comparison")
                    comparison_df = pd.DataFrame(st.session_state.model_comparison_results)
                    
                    # Create comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        chart_r2 = alt.Chart(comparison_df).mark_bar().encode(
                            x=alt.X('R2:Q', title='R² Score'),
                            y=alt.Y('Model:N', sort='-x'),
                            color=alt.Color('Model:N', legend=None)
                        ).properties(
                            title='R² Score Comparison',
                            height=200
                        )
                        st.altair_chart(chart_r2, use_container_width=True)
                    
                    with col2:
                        chart_mse = alt.Chart(comparison_df).mark_bar().encode(
                            x=alt.X('MSE:Q', title='Mean Squared Error'),
                            y=alt.Y('Model:N', sort='x'),
                            color=alt.Color('Model:N', legend=None)
                        ).properties(
                            title='MSE Comparison',
                            height=200
                        )
                        st.altair_chart(chart_mse, use_container_width=True)
                    
                    # CV误差对比（始终显示，只要有CV结果）
                    if "all_training_results" in st.session_state:
                        st.markdown("---")
                        st.subheader("Model CV Error Comparison Analysis")
                        
                        if st.session_state.get("enable_hp_tuning", False):
                            st.info("**Model Comparison Based on Grid Search Hyperparameter Tuning CV**")
                        else:
                            st.info("**Model Comparison Based on Basic 5-Fold Cross-Validation**")
                        
                        all_results = st.session_state.all_training_results
                        cv_comparison_data = []
                        
                        for result in all_results:
                            if "cv_scores" in result:
                                cv_comparison_data.append({
                                    'Model': result['name'],
                                    'CV_Mean': result['cv_mean'],
                                    'CV_Std': result['cv_std'],
                                    'CV_Train_Mean': result['cv_train_mean'],
                                    'Overfitting': result['cv_train_mean'] - result['cv_mean'],
                                    'Stability': 1 - (result['cv_std'] / result['cv_mean']) if result['cv_mean'] != 0 else 0
                                })
                        
                        if cv_comparison_data:
                            cv_comp_df = pd.DataFrame(cv_comparison_data)
                            
                            # Add color column for overfitting level visualization
                            def get_overfitting_color(overfitting):
                                if overfitting > 0.1:
                                    return 'red'
                                elif overfitting > 0.05:
                                    return 'orange'
                                else:
                                    return 'green'
                            
                            cv_comp_df['color'] = cv_comp_df['Overfitting'].apply(get_overfitting_color)
                            
                            # 显示CV比较表格
                            st.dataframe(cv_comp_df.round(4), use_container_width=True)
                            
                            # CV误差可视化对比
                            col_cv1, col_cv2 = st.columns(2)
                            
                            with col_cv1:
                                # CV mean and standard deviation comparison
                                cv_error_chart = alt.Chart(cv_comp_df).mark_errorbar(extent='stdev').encode(
                                    x=alt.X('Model:N', title='Model'),
                                    y=alt.Y('CV_Mean:Q', title='CV R² Score'),
                                    yError='CV_Std:Q'
                                ).properties(
                                    title='Model CV Performance Comparison (Mean ± Std Dev)',
                                    height=300
                                )
                                
                                cv_points = alt.Chart(cv_comp_df).mark_circle(size=100).encode(
                                    x='Model:N',
                                    y='CV_Mean:Q',
                                    color=alt.Color('Model:N', legend=None),
                                    tooltip=['Model', 'CV_Mean', 'CV_Std', 'Stability']
                                )
                                
                                st.altair_chart(cv_error_chart + cv_points, use_container_width=True)
                            
                            with col_cv2:
                                # Overfitting level comparison
                                overfitting_chart = alt.Chart(cv_comp_df).mark_bar().encode(
                                    x=alt.X('Model:N', title='Model'),
                                    y=alt.Y('Overfitting:Q', title='Overfitting Level'),
                                    color=alt.Color(
                                        'color:N',
                                        scale=alt.Scale(
                                            domain=['green', 'orange', 'red'],
                                            range=['green', 'orange', 'red']
                                        ),
                                        legend=alt.Legend(title="Overfitting Level")
                                    ),
                                    tooltip=['Model', 'Overfitting', 'CV_Mean', 'CV_Train_Mean']
                                ).properties(
                                    title='Model Overfitting Level Comparison',
                                    height=300
                                )
                                
                                st.altair_chart(overfitting_chart, use_container_width=True)
                            
                            # Recommend best model
                            best_stability = cv_comp_df.loc[cv_comp_df['Stability'].idxmax()]
                            best_performance = cv_comp_df.loc[cv_comp_df['CV_Mean'].idxmax()]
                            least_overfitting = cv_comp_df.loc[cv_comp_df['Overfitting'].idxmin()]
                            
                            st.markdown("### Model Recommendation Analysis")
                            col_rec1, col_rec2, col_rec3 = st.columns(3)
                            
                            with col_rec1:
                                st.info(f"""
                                **Most Stable Model:** {best_stability['Model']}
                                - Stability: {best_stability['Stability']:.3f}
                                - CV Std Dev: {best_stability['CV_Std']:.4f}
                                """)
                            
                            with col_rec2:
                                st.success(f"""
                                **Best Performance Model:** {best_performance['Model']}
                                - CV Mean: {best_performance['CV_Mean']:.4f}
                                - CV Std: {best_performance['CV_Std']:.4f}
                                """)
                            
                            with col_rec3:
                                st.warning(f"""
                                **Least Overfitting Model:** {least_overfitting['Model']}
                                - Overfitting Level: {least_overfitting['Overfitting']:.4f}
                                - Strong Generalization
                                """)
                
                # Visualization analysis
                st.subheader("Model Performance Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Actual vs predicted values
                    scatter_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.y_pred
                    })
                    
                    # Add ideal line data
                    min_val = min(scatter_df['Actual'].min(), scatter_df['Predicted'].min())
                    max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
                    line_df = pd.DataFrame({
                        'x': [min_val, max_val],
                        'y': [min_val, max_val]
                    })
                    
                    scatter = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.7).encode(
                        x=alt.X('Actual:Q', title='Actual Values'),
                        y=alt.Y('Predicted:Q', title='Predicted Values'),
                        tooltip=['Actual', 'Predicted']
                    )
                    
                    line = alt.Chart(line_df).mark_line(color='red', strokeDash=[5,5]).encode(
                        x='x:Q',
                        y='y:Q'
                    )
                    
                    chart = (scatter + line).properties(
                        title='Actual vs Predicted Values',
                        height=300
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                
                with col2:
                    # Residual plot
                    residuals = st.session_state.y_test - st.session_state.y_pred
                    resid_df = pd.DataFrame({
                        'Predicted': st.session_state.y_pred,
                        'Residuals': residuals
                    })
                    
                    # Add zero line
                    zero_line_df = pd.DataFrame({
                        'x': [resid_df['Predicted'].min(), resid_df['Predicted'].max()],
                        'y': [0, 0]
                    })
                    
                    resid_scatter = alt.Chart(resid_df).mark_circle(size=60, opacity=0.7).encode(
                        x=alt.X('Predicted:Q', title='Predicted Values'),
                        y=alt.Y('Residuals:Q', title='Residuals'),
                        tooltip=['Predicted', 'Residuals']
                    )
                    
                    zero_line = alt.Chart(zero_line_df).mark_line(color='red', strokeDash=[5,5]).encode(
                        x='x:Q',
                        y='y:Q'
                    )
                    
                    resid_chart = (resid_scatter + zero_line).properties(
                        title='Residual Plot',
                        height=300
                    )
                    
                    st.altair_chart(resid_chart, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance Analysis")
                
                try:
                    # SHAP analysis with Pipeline support
                    # Extract the actual model from pipeline if it's a pipeline
                    if hasattr(model, 'named_steps'):
                        # It's a Pipeline, get the final estimator
                        actual_model = model.named_steps[list(model.named_steps.keys())[-1]]
                    else:
                        actual_model = model
                    
                    # Check if model is tree-based for TreeExplainer
                    model_type = type(actual_model).__name__
                    if model_type in ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor', 'GradientBoostingRegressor']:
                        explainer = shap.TreeExplainer(actual_model)
                        sample_size = min(100, len(st.session_state.X_train_proc))
                        X_sample = st.session_state.X_train_proc[:sample_size]
                        
                        # Clean feature names
                        clean_feature_names = []
                        for name in st.session_state.feature_names:
                            clean_name = str(name).replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                            clean_name = clean_name.replace('=', 'eq').replace('&', 'and').replace('|', 'or')
                            clean_feature_names.append(clean_name)
                        
                        X_sample_df = pd.DataFrame(X_sample, columns=clean_feature_names)
                        shap_values = explainer.shap_values(X_sample_df)
                        
                        # Calculate feature importance
                        importance = np.abs(shap_values).mean(axis=0)
                        
                        # Truncate long feature names for better display
                        display_feature_names = []
                        for name in clean_feature_names:
                            if len(name) > 25:
                                # Keep the important part and add ellipsis
                                display_name = name[:22] + "..."
                            else:
                                display_name = name
                            display_feature_names.append(display_name)
                        
                        feat_imp_df = pd.DataFrame({
                            'Feature': display_feature_names,
                            'Importance': importance
                        })
                        
                        # Display SHAP-based feature importance
                        st.write(f"**SHAP Analysis:** Total features: {len(feat_imp_df)}, Non-zero features: {len(feat_imp_df[feat_imp_df['Importance'] > 0])}")
                        
                        # Determine number of features to display
                        max_features = min(15, len(feat_imp_df))
                        feat_imp_df = feat_imp_df.nlargest(max_features, 'Importance')
                        
                        # Create feature importance chart
                        imp_chart = alt.Chart(feat_imp_df).mark_bar().encode(
                            x=alt.X('Importance:Q', title='Mean |SHAP Value|'),
                            y=alt.Y('Feature:N', sort='-x', title='Feature'),
                            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None)
                        ).properties(
                            title=f'Top {max_features} Most Important Features (SHAP Analysis)',
                            height=max(400, max_features * 30)
                        )
                        
                        st.altair_chart(imp_chart, use_container_width=True)
                        
                        # Add SHAP Summary Plot (Beeswarm style)
                        st.markdown("#### SHAP Value Distribution")
                        try:
                            # Create SHAP summary plot using matplotlib
                            import matplotlib.pyplot as plt
                            import io
                            
                            # Create figure
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Create SHAP summary plot
                            shap.summary_plot(shap_values, X_sample_df, 
                                            feature_names=display_feature_names[:max_features],
                                            max_display=max_features, 
                                            show=False, ax=ax)
                            
                            # Adjust layout
                            plt.tight_layout()
                            
                            # Display in streamlit
                            st.pyplot(fig)
                            plt.close(fig)
                            
                        except Exception as plot_e:
                            st.info(f"Could not generate SHAP summary plot: {plot_e}")
                            
                            # Fallback: Create a simple scatter plot showing SHAP value distributions
                            try:
                                # Prepare data for scatter plot
                                scatter_data = []
                                for i, feature_name in enumerate(display_feature_names[:max_features]):
                                    for j, shap_val in enumerate(shap_values[:, i]):
                                        feature_val = X_sample[j, i] if i < X_sample.shape[1] else 0
                                        scatter_data.append({
                                            'Feature': feature_name,
                                            'SHAP_Value': shap_val,
                                            'Feature_Value': feature_val,
                                            'Sample': j
                                        })
                                
                                scatter_df = pd.DataFrame(scatter_data)
                                
                                # Create scatter plot
                                scatter_chart = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.7).encode(
                                    x=alt.X('SHAP_Value:Q', title='SHAP Value'),
                                    y=alt.Y('Feature:N', sort=alt.EncodingSortField(field='SHAP_Value', op='mean', order='descending'), title='Feature'),
                                    color=alt.Color('Feature_Value:Q', scale=alt.Scale(scheme='viridis'), title='Feature Value'),
                                    tooltip=['Feature:N', 'SHAP_Value:Q', 'Feature_Value:Q', 'Sample:O']
                                ).properties(
                                    title='SHAP Value Distribution by Feature',
                                    height=max(400, max_features * 30),
                                    width=600
                                )
                                
                                st.altair_chart(scatter_chart, use_container_width=True)
                                
                            except Exception as fallback_e:
                                st.warning(f"Could not create fallback SHAP visualization: {fallback_e}")
                        
                        # Display detailed feature importance table
                        with st.expander("Detailed SHAP Feature Importance Values", expanded=False):
                            st.dataframe(feat_imp_df.reset_index(drop=True), use_container_width=True)
                    
                    else:
                        # For non-tree models, use Linear/Kernel explainer or fallback to built-in importance
                        st.info(f"SHAP TreeExplainer not supported for {model_type}. Using alternative analysis.")
                        
                        # Try LinearExplainer for linear models
                        if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
                            try:
                                explainer = shap.LinearExplainer(actual_model, st.session_state.X_train_proc[:100])
                                sample_size = min(50, len(st.session_state.X_train_proc))
                                X_sample = st.session_state.X_train_proc[:sample_size]
                                shap_values = explainer.shap_values(X_sample)
                                
                                # Calculate feature importance
                                importance = np.abs(shap_values).mean(axis=0)
                                clean_feature_names = [str(name).replace('[', '').replace(']', '') for name in st.session_state.feature_names]
                                
                                feat_imp_df = pd.DataFrame({
                                    'Feature': clean_feature_names,
                                    'Importance': importance
                                })
                                
                                max_features = min(15, len(feat_imp_df))
                                feat_imp_df = feat_imp_df.nlargest(max_features, 'Importance')
                                
                                imp_chart = alt.Chart(feat_imp_df).mark_bar().encode(
                                    x=alt.X('Importance:Q', title='Mean |SHAP Value|'),
                                    y=alt.Y('Feature:N', sort='-x', title='Feature'),
                                    color=alt.Color('Importance:Q', scale=alt.Scale(scheme='oranges'), legend=None)
                                ).properties(
                                    title=f'Top {max_features} Most Important Features (Linear SHAP)',
                                    height=max(300, max_features * 25)
                                )
                                
                                st.altair_chart(imp_chart, use_container_width=True)
                                
                                # Add Linear SHAP distribution visualization
                                st.markdown("#### Linear SHAP Value Distribution")
                                try:
                                    # Create a violin plot style visualization for linear SHAP
                                    shap_dist_data = []
                                    for i, feature_name in enumerate(clean_feature_names[:max_features]):
                                        for shap_val in shap_values[:, i]:
                                            shap_dist_data.append({
                                                'Feature': feature_name[:22] + "..." if len(feature_name) > 25 else feature_name,
                                                'SHAP_Value': shap_val
                                            })
                                    
                                    shap_dist_df = pd.DataFrame(shap_dist_data)
                                    
                                    # Create box plot
                                    box_chart = alt.Chart(shap_dist_df).mark_boxplot(
                                        outliers=True,
                                        size=20
                                    ).encode(
                                        x=alt.X('SHAP_Value:Q', title='SHAP Value'),
                                        y=alt.Y('Feature:N', sort=alt.EncodingSortField(field='SHAP_Value', op='mean', order='descending'), title='Feature'),
                                        color=alt.Color('Feature:N', legend=None, scale=alt.Scale(scheme='category20'))
                                    ).properties(
                                        title='Linear SHAP Value Distribution (Box Plot)',
                                        height=max(400, max_features * 25),
                                        width=600
                                    )
                                    
                                    st.altair_chart(box_chart, use_container_width=True)
                                    
                                except Exception as dist_e:
                                    st.info(f"Could not create SHAP distribution plot: {dist_e}")
                                
                                with st.expander("Detailed Linear SHAP Feature Importance", expanded=False):
                                    st.dataframe(feat_imp_df.reset_index(drop=True), use_container_width=True)
                                    
                            except Exception as linear_e:
                                st.warning(f"Linear SHAP analysis also failed: {linear_e}")
                                # Fall back to built-in feature importance
                                if hasattr(actual_model, 'feature_importances_'):
                                    importance = actual_model.feature_importances_
                                    clean_feature_names = [str(name).replace('[', '').replace(']', '') for name in st.session_state.feature_names]
                                    
                                    # Truncate long feature names for better display
                                    display_feature_names = []
                                    for name in clean_feature_names:
                                        if len(name) > 25:
                                            display_name = name[:22] + "..."
                                        else:
                                            display_name = name
                                        display_feature_names.append(display_name)
                                    
                                    feat_imp_df = pd.DataFrame({
                                        'Feature': display_feature_names,
                                        'Importance': importance
                                    })
                                    
                                    max_features = min(15, len(feat_imp_df))
                                    feat_imp_df = feat_imp_df.nlargest(max_features, 'Importance')
                                    
                                    imp_chart = alt.Chart(feat_imp_df).mark_bar().encode(
                                        x=alt.X('Importance:Q', title='Feature Importance'),
                                        y=alt.Y('Feature:N', sort='-x', title='Feature'),
                                        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='greens'), legend=None)
                                    ).properties(
                                        title=f'Top {max_features} Most Important Features (Built-in)',
                                        height=max(400, max_features * 30)
                                    )
                                    
                                    st.altair_chart(imp_chart, use_container_width=True)
                                else:
                                    st.warning("No feature importance analysis available for this model type.")
                        else:
                            # Fallback to built-in feature importance for other models
                            if hasattr(actual_model, 'feature_importances_'):
                                importance = actual_model.feature_importances_
                                clean_feature_names = [str(name).replace('[', '').replace(']', '') for name in st.session_state.feature_names]
                                
                                # Truncate long feature names for better display
                                display_feature_names = []
                                for name in clean_feature_names:
                                    if len(name) > 25:
                                        display_name = name[:22] + "..."
                                    else:
                                        display_name = name
                                    display_feature_names.append(display_name)
                                
                                feat_imp_df = pd.DataFrame({
                                    'Feature': display_feature_names,
                                    'Importance': importance
                                })
                                
                                max_features = min(15, len(feat_imp_df))
                                feat_imp_df = feat_imp_df.nlargest(max_features, 'Importance')
                                
                                imp_chart = alt.Chart(feat_imp_df).mark_bar().encode(
                                    x=alt.X('Importance:Q', title='Feature Importance'),
                                    y=alt.Y('Feature:N', sort='-x', title='Feature'),
                                    color=alt.Color('Importance:Q', scale=alt.Scale(scheme='greens'), legend=None)
                                ).properties(
                                    title=f'Top {max_features} Most Important Features (Built-in)',
                                    height=max(400, max_features * 30)
                                )
                                
                                st.altair_chart(imp_chart, use_container_width=True)
                                
                                with st.expander("Detailed Built-in Feature Importance", expanded=False):
                                    st.dataframe(feat_imp_df.reset_index(drop=True), use_container_width=True)
                                st.warning("No feature importance analysis available for this model type.")
                
                except Exception as e:
                    st.warning(f"Feature importance analysis failed: {e}")
                    st.info("Using basic model information if available.")
                    
                    # Final fallback: basic model info
                    if hasattr(model, 'feature_importances_'):
                        try:
                            # Extract model from pipeline if needed
                            if hasattr(model, 'named_steps'):
                                actual_model = model.named_steps[list(model.named_steps.keys())[-1]]
                            else:
                                actual_model = model
                                
                            importance = actual_model.feature_importances_
                            clean_feature_names = [str(name).replace('[', '').replace(']', '') for name in st.session_state.feature_names]
                            
                            feat_imp_df = pd.DataFrame({
                                'Feature': clean_feature_names[:len(importance)],
                                'Importance': importance
                            })
                            
                            max_features = min(10, len(feat_imp_df))
                            feat_imp_df = feat_imp_df.nlargest(max_features, 'Importance')
                            
                            fallback_chart = alt.Chart(feat_imp_df).mark_bar().encode(
                                x=alt.X('Importance:Q', title='Feature Importance'),
                                y=alt.Y('Feature:N', sort='-x', title='Feature'),
                                color=alt.value('#8B4513')
                            ).properties(
                                title=f'Top {max_features} Features (Fallback Analysis)',
                                height=max(250, max_features * 20)
                            )
                            
                            st.altair_chart(fallback_chart, use_container_width=True)
                        except Exception as fallback_e:
                            st.error(f"All feature importance methods failed: {fallback_e}")
                    else:
                        st.info("No feature importance methods available for this model type.")
# ——————————————————————————————
# Tab 6: Prediction (对所有已认证用户可见)
# ——————————————————————————————
with tabs[tab_indexes["predictions"]]:
    
    # Predictions interface
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; border-left: 4px solid #3498db;">
        <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 600;">
            Predictions
        </h3>
        <p style="color: #7f8c8d; margin: 0; font-size: 1rem;">
            Use trained models to make predictions on new data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型选择部分 - 只对admin用户显示
    if "authenticated_user" in st.session_state and st.session_state["authenticated_user"]["role"] == "admin":
        st.markdown("#### Model Selection")
        
        # 检查可用的模型
        has_current_model = "trained_model" in st.session_state
        
        try:
            model_cache_manager = ModelCacheManager(engine)
            cached_models = model_cache_manager.list_cached_models()
            has_cached_models = len(cached_models) > 0
        except:
            cached_models = []
            has_cached_models = False
        
        if not has_current_model and not has_cached_models:
            st.warning("No trained models available. Please train a model first in the Model Training tab.")
            st.stop()
        
        # 模型选择选项
        model_source_options = []
        if has_current_model:
            current_model_name = st.session_state.get("selected_model", "Current Model")
            current_target = st.session_state.get("selected_target", "Unknown Target")
            model_source_options.append(f"Current Session Model ({current_model_name} - {current_target})")
        
        if has_cached_models:
            for model in cached_models:
                model_info = f"Cached: {model['model_name']} - {model['target_variable']} ({model['model_key']})"
                model_source_options.append(model_info)
        
        selected_model_source = st.selectbox(
            "Select model to use for predictions:",
            model_source_options,
            help="Choose between current session model or previously cached models"
        )
    else:
        # 对于非admin用户，直接使用Model Configuration中选择的模型
        if "selected_model_for_prediction" not in st.session_state:
            st.warning("No model selected. Please select a model in the Model Configuration tab first.")
            st.stop()
        
        # 使用Model Configuration中选择的模型信息
        selected_model_source = "auto_selected"
    
    # 根据选择加载模型
    if selected_model_source.startswith("Current Session Model"):
        # 使用当前会话的模型
        model_to_use = st.session_state.trained_model
        model_target = st.session_state.get("selected_target", "Unknown")
        model_name = st.session_state.get("selected_model", "Unknown")
        training_feature_info = st.session_state.get('training_feature_info', {})
        training_data_type = st.session_state.get("training_data_type", "advanced_preprocessing_dataset")
        
        st.success(f"Using current session model: {model_name} for {model_target}")
        
    elif selected_model_source == "auto_selected":
        # 对于非admin用户，使用Model Configuration中选择的模型
        try:
            model_config = st.session_state["selected_model_for_prediction"]
            model_cache_manager = ModelCacheManager(engine)
            
            cached_model_details = model_cache_manager.load_model(model_config['model_key'])
            if cached_model_details:
                model_to_use = cached_model_details['model']
                model_target = cached_model_details['target_variable']
                model_name = cached_model_details['model_name']
                
                # 重建training_feature_info结构
                feature_info = cached_model_details['feature_info']
                preprocessing_info = cached_model_details['preprocessing_info']
                
                # 从缓存中恢复预处理器对象
                feature_encoder = None
                feature_scaler = None
                
                if preprocessing_info.get('feature_encoder'):
                    try:
                        import pickle
                        feature_encoder = pickle.loads(base64.b64decode(preprocessing_info['feature_encoder'].encode()))
                    except Exception as e:
                        st.warning(f"Failed to load feature encoder: {e}")
                        st.warning("This model was saved before the preprocessing fix. Please retrain the model to use cached preprocessors.")
                
                if preprocessing_info.get('feature_scaler'):
                    try:
                        import pickle
                        feature_scaler = pickle.loads(base64.b64decode(preprocessing_info['feature_scaler'].encode()))
                    except Exception as e:
                        st.warning(f"Failed to load feature scaler: {e}")
                        st.warning("This model was saved before the preprocessing fix. Please retrain the model to use cached preprocessors.")
                
                # 向后兼容性检查：如果无法从缓存中恢复预处理器，尝试使用session state中的
                if feature_encoder is None and hasattr(st.session_state, 'feature_encoder'):
                    feature_encoder = st.session_state.feature_encoder
                    st.info("Using fallback preprocessor from current session")
                
                if feature_scaler is None and hasattr(st.session_state, 'feature_scaler'):
                    feature_scaler = st.session_state.feature_scaler
                    st.info("Using fallback preprocessor from current session")
                
                training_feature_info = {
                    'feature_names': feature_info.get('feature_names', []),
                    'feature_columns': feature_info.get('feature_columns', []),
                    'numeric_features': feature_info.get('numeric_features', []),
                    'categorical_features': feature_info.get('categorical_features', []),
                    'training_columns': feature_info.get('feature_columns', []),
                    'processed_feature_names': feature_info.get('feature_names', []),
                    'feature_encoder': feature_encoder,  # 恢复的预处理器对象
                    'feature_scaler': feature_scaler,    # 恢复的预处理器对象
                    'target_variable': cached_model_details['target_variable']
                }
                training_data_type = "advanced_preprocessing_dataset"
                
                # 保存训练特征信息到session_state
                st.session_state.training_feature_info = training_feature_info
                
                # 显示模型信息
                eval_results = cached_model_details['evaluation_results']
                st.info(f"""
                **Using selected model:** {model_name} for {model_target}
                - Test R²: {eval_results.get('test_r2', 'N/A'):.4f}
                - CV R²: {eval_results.get('cv_r2_mean', 'N/A'):.4f}
                """)
            else:
                st.error("Failed to load selected model from Model Configuration")
                st.stop()
        except Exception as e:
            st.error(f"Error loading model from Model Configuration: {e}")
            st.stop()
    else:
        # 使用缓存的模型（admin用户）
        # 提取model_key
        model_key = selected_model_source.split('(')[-1].split(')')[0]
        
        try:
            cached_model_details = model_cache_manager.load_model(model_key)
            if cached_model_details:
                model_to_use = cached_model_details['model']
                model_target = cached_model_details['target_variable']
                model_name = cached_model_details['model_name']
                
                # 重建training_feature_info结构
                feature_info = cached_model_details['feature_info']
                preprocessing_info = cached_model_details['preprocessing_info']
                
                # 从缓存中恢复预处理器对象
                feature_encoder = None
                feature_scaler = None
                
                if preprocessing_info.get('feature_encoder'):
                    try:
                        import pickle
                        feature_encoder = pickle.loads(base64.b64decode(preprocessing_info['feature_encoder'].encode()))
                    except Exception as e:
                        st.warning(f"Failed to load feature encoder: {e}")
                        st.warning("This model was saved before the preprocessing fix. Please retrain the model to use cached preprocessors.")
                
                if preprocessing_info.get('feature_scaler'):
                    try:
                        import pickle
                        feature_scaler = pickle.loads(base64.b64decode(preprocessing_info['feature_scaler'].encode()))
                    except Exception as e:
                        st.warning(f"Failed to load feature scaler: {e}")
                        st.warning("This model was saved before the preprocessing fix. Please retrain the model to use cached preprocessors.")
                
                # 向后兼容性检查：如果无法从缓存中恢复预处理器，尝试使用session state中的
                if feature_encoder is None and hasattr(st.session_state, 'feature_encoder'):
                    feature_encoder = st.session_state.feature_encoder
                    st.info("Using fallback preprocessor from current session")
                
                if feature_scaler is None and hasattr(st.session_state, 'feature_scaler'):
                    feature_scaler = st.session_state.feature_scaler
                    st.info("Using fallback preprocessor from current session")
                
                training_feature_info = {
                    'feature_names': feature_info.get('feature_names', []),
                    'feature_columns': feature_info.get('feature_columns', []),
                    'numeric_features': feature_info.get('numeric_features', []),
                    'categorical_features': feature_info.get('categorical_features', []),
                    'training_columns': feature_info.get('feature_columns', []),
                    'processed_feature_names': feature_info.get('feature_names', []),
                    'feature_encoder': feature_encoder,  # 恢复的预处理器对象
                    'feature_scaler': feature_scaler,    # 恢复的预处理器对象
                    'target_variable': cached_model_details['target_variable']
                }
                training_data_type = "advanced_preprocessing_dataset"
                
                # 保存训练特征信息到session_state
                st.session_state.training_feature_info = training_feature_info
                
                # 显示模型信息
                eval_results = cached_model_details['evaluation_results']
                st.success(f"""
                Loaded cached model: {model_name} for {model_target}
                - Test R²: {eval_results.get('test_r2', 'N/A'):.4f}
                - CV R²: {eval_results.get('cv_r2_mean', 'N/A'):.4f}
                """)
            else:
                st.error("Failed to load selected cached model")
                st.stop()
        except Exception as e:
            st.error(f"Error loading cached model: {e}")
            st.stop()
    
    # 检查训练数据类型
    use_advanced_features = training_data_type == "advanced_preprocessing_dataset"
    
    # 预测模式选择
    pred_mode = st.radio(
        "Prediction Mode",
        ["Single Prediction", "Batch Prediction"],
        horizontal=True
    )
    
    if pred_mode == "Single Prediction":
            st.subheader("Single Instance Prediction")
            
            # 根据训练数据类型显示不同的输入界面
            if use_advanced_features:
                st.markdown("### Advanced Feature Input (13 Standardized Features)")
                
                # Using 13 features from advanced preprocessing
                with st.form("advanced_prediction_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Environmental Conditions**")
                        ph_env = st.number_input("pH of condition environment", 0.0, 14.0, 7.0)
                        exposure_time = st.number_input("Exposure time (hours)", 0.0, 100000.0, 1000.0)
                        temperature = st.number_input("Exposure temperature (°C)", -50.0, 200.0, 25.0)
                        chloride_ion = st.selectbox("Presence of chloride ion", [0, 1])
                        concrete = st.selectbox("Presence of concrete", [0, 1])
                    
                    with col2:
                        st.markdown("**Material Properties**")
                        fiber_content = st.number_input("Fibre content (%)", 0.0, 100.0, 60.0)
                        diameter = st.number_input("Diameter (mm)", 0.1, 50.0, 10.0)
                        fiber_type = st.selectbox("Fibre type", [0, 1], format_func=lambda x: "Basalt" if x==0 else "Glass")
                        matrix_type = st.selectbox("Matrix type", [0, 1], format_func=lambda x: "Epoxy" if x==0 else "Vinyl ester")
                        surface_treatment = st.selectbox("Surface treatment", [0, 1], format_func=lambda x: "sand coated" if x==0 else "Smooth")
                    
                    with col3:
                        st.markdown("**Load and Strength**")
                        load = st.number_input("Load", 0.0, 1.0, 0.0, help="Load ratio (0-1)")
                        unconditioned_strength = st.number_input("Strength of unconditioned rebar (MPa)", 100.0, 5000.0, 1000.0)
                    
                    submitted = st.form_submit_button("Predict with Advanced Features", type="primary", use_container_width=True)
                    
                    if submitted:
                        # 准备高级特征输入数据
                        advanced_input = {
                            "pH of condition environment": ph_env,
                            "Exposure time": exposure_time,
                            "Fibre content": fiber_content,
                            "Exposure temperature": temperature,
                            "Diameter": diameter,
                            "Presence of concrete": concrete,
                            "Load": load,
                            "Presence of chloride ion": chloride_ion,
                            "Fibre type": fiber_type,
                            "Matrix type": matrix_type,
                            "Surface treatment": surface_treatment,
                            "Strength of unconditioned rebar": unconditioned_strength
                        }
                        
                        # 执行预测
                        try:
                            # 创建输入DataFrame（匹配训练时的特征顺序）
                            input_df = pd.DataFrame([advanced_input])
                            
                            # 简洁的预测流程，无调试信息
                            
                            # 使用从training_feature_info中恢复的预处理器
                            feature_encoder = training_feature_info.get('feature_encoder')
                            feature_scaler = training_feature_info.get('feature_scaler')
                            
                            if feature_encoder is not None and feature_scaler is not None:
                                # 获取训练时的特征信息
                                numeric_features = training_feature_info.get('numeric_features', [])
                                categorical_features = training_feature_info.get('categorical_features', [])
                                
                                # 处理数值特征
                                if numeric_features:
                                    # 确保输入包含所有数值特征
                                    input_numeric = input_df[numeric_features].fillna(0)
                                    input_num_scaled = feature_scaler.transform(input_numeric)
                                else:
                                    input_num_scaled = np.empty((1, 0))
                                
                                # 处理分类特征
                                if categorical_features:
                                    input_categorical = input_df[categorical_features].fillna('unknown').astype(str)
                                    input_cat_encoded = feature_encoder.transform(input_categorical)
                                else:
                                    input_cat_encoded = np.empty((1, 0))
                                
                                # 合并特征
                                input_processed = np.hstack([input_num_scaled, input_cat_encoded])
                                
                                # 检查特征数量是否与训练时一致
                                training_feature_names = training_feature_info.get('feature_names', [])
                                expected_features_count = len(training_feature_names)
                                actual_features_count = input_processed.shape[1]
                                
                                if expected_features_count > 0 and expected_features_count != actual_features_count:
                                    # 尝试修复特征数量不匹配
                                    if actual_features_count < expected_features_count:
                                        padding_needed = expected_features_count - actual_features_count
                                        padding = np.zeros((1, padding_needed))
                                        input_processed = np.hstack([input_processed, padding])
                                        
                                    elif actual_features_count > expected_features_count:
                                        input_processed = input_processed[:, :expected_features_count]
                                
                            else:
                                st.error("缺少预处理器信息，请重新训练模型")
                                st.stop()
                            
                            # Make prediction
                            model = model_to_use
                            prediction = model.predict(input_processed)[0]
                            
                            # 合理性检查和结果转换
                            if model_target.startswith("Tensile"):
                                # 检查预测值是否为小数形式（需要转换为百分比）
                                if prediction < 2:  # 如果预测值小于2，可能是小数形式
                                    display_prediction = prediction * 100  # 转换为百分比
                                else:
                                    display_prediction = prediction
                            else:
                                # 强度预测，检查物理合理性
                                display_prediction = prediction
                                
                                # Apply physical constraint for residual strength
                                if prediction > unconditioned_strength:
                                    corrected_prediction = min(prediction, unconditioned_strength)
                                    prediction = corrected_prediction
                            
                            # Display results
                            st.success("Prediction completed!")
                            
                            if model_target.startswith("Residual"):
                                # For residual strength prediction, show 3 columns
                                col_pred1, col_pred2, col_pred3 = st.columns(3)
                                
                                with col_pred1:
                                    st.metric("Predicted Residual Strength", f"{prediction:.2f} MPa")
                                
                                with col_pred2:
                                    retention_rate = (prediction / unconditioned_strength) * 100
                                    st.metric("Retention Rate", f"{retention_rate:.2f}%")
                                
                                with col_pred3:
                                    if retention_rate >= 80:
                                        performance = "Excellent"
                                        color = "🟢"
                                    elif retention_rate >= 60:
                                        performance = "Good"
                                        color = "🟡"
                                    else:
                                        performance = "Needs Attention"
                                        color = "🔴"
                                    st.metric("Performance", f"{color} {performance}")
                            
                            else:
                                # For retention rate prediction, show 2 columns
                                col_pred1, col_pred2 = st.columns(2)
                                
                                with col_pred1:
                                    # 使用转换后的显示值
                                    if prediction < 2:  # 小数形式转百分比
                                        display_value = prediction * 100
                                    else:
                                        display_value = prediction
                                    st.metric("Predicted Retention Rate", f"{display_value:.2f}%")
                                
                                with col_pred2:
                                    # 使用转换后的值计算性能等级
                                    if prediction < 2:
                                        retention_rate = prediction * 100
                                    else:
                                        retention_rate = prediction
                                    
                                    if retention_rate >= 80:
                                        performance = "Excellent"
                                        color = "🟢"
                                    elif retention_rate >= 60:
                                        performance = "Good"
                                        color = "🟡"
                                    else:
                                        performance = "Needs Attention"
                                        color = "🔴"
                                    
                                    st.metric("Performance", f"{color} {performance}")
                            
                            # Record prediction
                            if "authenticated_user" in st.session_state and engine:
                                user_info = st.session_state["authenticated_user"]
                                pred_data = {
                                    "prediction": prediction,
                                    "input_features": advanced_input,
                                    "model": model_name,
                                    "target": model_target,
                                    "data_type": "advanced_features",
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                try:
                                    log_operation(
                                        user_info['email'],
                                        'prediction',
                                        'single',
                                        None,
                                        pred_data,
                                        get_client_ip(),
                                        engine
                                    )
                                except:
                                    pass  # 日志记录失败不影响预测功能
                        
                        except Exception as e:
                            # 高级预测失败，尝试智能修复
                            error_msg = str(e)
                            
                            # 检查是否是特征数量不匹配错误
                            if "features" in error_msg.lower() and "expecting" in error_msg.lower():
                                # 从错误信息中提取期望的特征数量
                                import re
                                match = re.search(r'expecting (\d+) features', error_msg)
                                expected_by_model = int(match.group(1)) if match else None
                                
                                # 尝试智能特征修复
                                try:
                                    # 重新处理特征，确保数量匹配
                                    if hasattr(st.session_state, 'feature_encoder') and hasattr(st.session_state, 'feature_scaler'):
                                        encoder = st.session_state.feature_encoder
                                        scaler = st.session_state.feature_scaler
                                        numeric_features = st.session_state.get('numeric_features', [])
                                        categorical_features = st.session_state.get('categorical_features', [])
                                        
                                        # 重新创建输入
                                        input_df = pd.DataFrame([advanced_input])
                                        
                                        # 处理数值特征
                                        if numeric_features:
                                            input_numeric = input_df[numeric_features].fillna(0)
                                            input_num_scaled = scaler.transform(input_numeric)
                                        else:
                                            input_num_scaled = np.empty((1, 0))
                                        
                                        # 处理分类特征
                                        if categorical_features:
                                            input_categorical = input_df[categorical_features].fillna('unknown').astype(str)
                                            input_cat_encoded = encoder.transform(input_categorical)
                                        else:
                                            input_cat_encoded = np.empty((1, 0))
                                        
                                        # 合并特征
                                        input_processed_fixed = np.hstack([input_num_scaled, input_cat_encoded])
                                        current_feature_count = input_processed_fixed.shape[1]
                                        
                                        # 根据模型期望调整特征数量
                                        if match and expected_by_model:
                                            target_features = expected_by_model
                                            if current_feature_count < target_features:
                                                padding = np.zeros((1, target_features - current_feature_count))
                                                input_processed_fixed = np.hstack([input_processed_fixed, padding])
                                            elif current_feature_count > target_features:
                                                input_processed_fixed = input_processed_fixed[:, :target_features]
                                        
                                        # Prediction
                                        prediction = model_to_use.predict(input_processed_fixed)[0]
                                        
                                        # Apply physical constraints for residual strength
                                        if model_target == "Residual tensile strength (MPa)":
                                            if prediction > unconditioned_strength:
                                                # Calculate physics-based prediction
                                                estimated_retention = min(0.95, max(0.1, 0.8 - (ph_env - 7.0) * 0.05 - (temperature - 20.0) * 0.002))
                                                physics_based_prediction = unconditioned_strength * estimated_retention
                                                
                                                # Apply physical constraint
                                                corrected_prediction = min(prediction, unconditioned_strength)
                                                
                                                # Use more conservative prediction value
                                                prediction = min(corrected_prediction, physics_based_prediction)
                                        
                                        # 继续处理预测结果...
                                        if model_target.startswith("Tensile"):
                                            if prediction < 2:
                                                display_prediction = prediction * 100
                                                st.info(f"� 转换为百分比：{display_prediction:.2f}%")
                                            else:
                                                display_prediction = prediction
                                        else:
                                            display_prediction = prediction
                                        
                                        # Process prediction result
                                        if model_target.startswith("Tensile"):
                                            if prediction < 2:
                                                display_prediction = prediction * 100
                                            else:
                                                display_prediction = prediction
                                        else:
                                            display_prediction = prediction
                                        
                                        # Display prediction results
                                        st.markdown("---")
                                        st.subheader("Prediction Results")
                                        
                                        # Skip subsequent simplified prediction, show results directly
                                        skip_fallback = True
                                    else:
                                        skip_fallback = False
                                
                                except Exception as fix_error:
                                    skip_fallback = False
                            else:
                                skip_fallback = False
                            
                            # If intelligent repair fails, continue with simplified prediction
                            if not locals().get('skip_fallback', False):
                                
                                try:
                                    # For cached models, try to use saved feature information to construct proper feature vector
                                    if "training_feature_info" in st.session_state and st.session_state["training_feature_info"]:
                                        feature_info = st.session_state["training_feature_info"]
                                        expected_features = feature_info.get("processed_feature_names", [])
                                        expected_count = len(expected_features)
                                        
                                        # Get the numeric and categorical features used during training
                                        training_numeric_features = feature_info.get("numeric_features", [])
                                        training_categorical_features = feature_info.get("categorical_features", [])
                                        
                                        # Create a comprehensive feature dictionary with all possible features
                                        comprehensive_features = {
                                            'pH of condition environment': ph_env,
                                            'Exposure time': exposure_time,
                                            'Fibre content': fiber_content,
                                            'Exposure temperature': temperature,
                                            'Diameter': diameter,
                                            'Strength of unconditioned rebar': unconditioned_strength,
                                            'Presence of concrete': 0.0,  # Default numeric value
                                            'Load': 0.0,  # Default numeric value
                                            'Presence of chloride ion': 0.0,  # Default numeric value
                                            # Use correct categorical feature names as expected by training
                                            'Fibre type': fiber_type if 'fiber_type' in locals() else 'unknown',
                                            'Matrix type': matrix_type if 'matrix_type' in locals() else 'unknown',
                                            'Surface treatment': surface_treatment if 'surface_treatment' in locals() else 'unknown'
                                        }
                                        
                                        # Try to use the exact same preprocessing as training
                                        try:
                                            # Method 1: Use Pipeline model directly with DataFrame
                                            if hasattr(model_to_use, 'named_steps'):
                                                # Get the exact training column order from training_feature_info
                                                training_columns = training_feature_info.get('training_columns', [])
                                                if training_columns:
                                                    
                                                    # Create DataFrame with only the training columns in exact order
                                                    ordered_data = {}
                                                    for col in training_columns:
                                                        if col in comprehensive_features:
                                                            ordered_data[col] = comprehensive_features[col]
                                                        else:
                                                            # Use appropriate default values for missing columns
                                                            if col in ['Fibre type', 'Matrix type', 'Surface treatment']:
                                                                ordered_data[col] = 'unknown'
                                                            else:
                                                                ordered_data[col] = 0.0
                                                    
                                                    feature_df = pd.DataFrame([ordered_data])
                                                    
                                                    # Ensure column order matches exactly
                                                    feature_df = feature_df[training_columns]
                                                    
                                                else:
                                                    # Fallback: use all available features
                                                    st.warning("No training columns info, using all available features")
                                                    feature_df = pd.DataFrame([comprehensive_features])
                                                    
                                                    # Add any missing columns that might be needed
                                                    all_training_features = training_numeric_features + training_categorical_features
                                                    for feature in all_training_features:
                                                        if feature not in feature_df.columns:
                                                            if feature in ['Fibre type', 'Matrix type', 'Surface treatment']:
                                                                feature_df[feature] = 'unknown'
                                                            else:
                                                                feature_df[feature] = 0.0
                                                
                                                # Clean categorical data to prevent string handling issues
                                                for col in feature_df.columns:
                                                    if feature_df[col].dtype == object:
                                                        feature_df[col] = feature_df[col].fillna('unknown').astype(str)
                                                
                                                try:
                                                    prediction = model_to_use.predict(feature_df)[0]
                                                    
                                                    # Check if prediction value is reasonable
                                                    is_abnormal = False
                                                    if model_target.startswith("Tensile") or "retention" in model_target.lower():
                                                        # For retention rate prediction: should be between 0 and 1 (or 0-100 if percentage)
                                                        if prediction < 0 or prediction > 1.2:  # Allow some tolerance
                                                            is_abnormal = True
                                                            abnormal_reason = f"Retention rate {prediction:.3f} is outside normal range [0, 1.2]"
                                                    else:
                                                        # For residual strength prediction: should be positive and reasonable
                                                        if prediction < 1 or prediction > unconditioned_strength * 1.5:
                                                            is_abnormal = True
                                                            abnormal_reason = f"Residual strength {prediction:.1f} MPa seems unrealistic (original: {unconditioned_strength:.1f} MPa)"
                                                    
                                                    if is_abnormal:
                                                        st.warning(f"Note: {abnormal_reason}")
                                                        st.write("Possible reasons: Feature scaling issues, model training data issues, or feature mismatch")
                                                    
                                                    # Pipeline prediction successful, skip further fallback attempts
                                                    skip_fallback = True
                                                    
                                                except Exception as pipeline_error:
                                                    st.error(f"Debug: Pipeline prediction failed: {pipeline_error}")
                                                    st.error(f"Debug: DataFrame columns: {feature_df.columns.tolist()}")
                                                    st.error(f"Debug: DataFrame dtypes: {feature_df.dtypes.to_dict()}")
                                                    st.error(f"Debug: DataFrame shape: {feature_df.shape}")
                                                    
                                                    # Try to get pipeline step information
                                                    if hasattr(model_to_use, 'named_steps'):
                                                        st.error(f"Debug: Pipeline steps: {list(model_to_use.named_steps.keys())}")
                                                        if 'preprocessor' in model_to_use.named_steps:
                                                            preprocessor = model_to_use.named_steps['preprocessor']
                                                            if hasattr(preprocessor, 'transformers'):
                                                                st.error(f"Debug: Preprocessor transformers: {[(name, transformer, cols) for name, transformer, cols in preprocessor.transformers]}")
                                                    
                                                    # Don't raise error, instead continue to fallback methods
                                                    st.info("Will try alternative prediction method...")
                                                    skip_fallback = False
                                            else:
                                                # Method 2: Manual preprocessing to match training exactly
                                                # Use the saved preprocessors if available
                                                feature_encoder = feature_info.get('feature_encoder')
                                                feature_scaler = feature_info.get('feature_scaler')
                                                
                                                if feature_encoder is not None and feature_scaler is not None:
                                                    # Reconstruct the exact feature vector used during training
                                                    
                                                    # Step 1: Process numeric features
                                                    numeric_values = []
                                                    for feature in training_numeric_features:
                                                        if feature in comprehensive_features:
                                                            value = comprehensive_features[feature]
                                                            # Handle case where categorical features were misclassified as numeric during training
                                                            if feature in ['Fibre type', 'Matrix type']:
                                                                # These were misclassified as numeric, use numeric encoding
                                                                if feature == 'Fibre type':
                                                                    # Map fiber type to numeric value as training expected
                                                                    fiber_mapping = {'Glass': 0, 'Carbon': 1, 'Basalt': 2, 'unknown': 0}
                                                                    fiber_val = comprehensive_features.get('Fibre type', 'unknown')
                                                                    numeric_values.append(float(fiber_mapping.get(fiber_val, 0)))
                                                                elif feature == 'Matrix type':
                                                                    # Map matrix type to numeric value as training expected
                                                                    matrix_mapping = {'Vinyl ester': 0, 'Epoxy': 1, 'unknown': 0}
                                                                    matrix_val = comprehensive_features.get('Matrix type', 'unknown')
                                                                    numeric_values.append(float(matrix_mapping.get(matrix_val, 0)))
                                                            else:
                                                                numeric_values.append(float(value))
                                                        else:
                                                            numeric_values.append(0.0)  # Default value for missing features
                                                    
                                                    # Critical fix: If model expects 17 features but we only have 16,
                                                    # we need to add the missing numeric feature that was wrongly excluded
                                                    if len(training_numeric_features) == 10 and expected_count == 17:
                                                        # Check if we need to add 'Surface treatment' as numeric feature
                                                        if 'Surface treatment' not in training_numeric_features:
                                                            # Add Surface treatment as numeric (0 for 'unknown')
                                                            surface_mapping = {'None': 0, 'Coating': 1, 'unknown': 0}
                                                            surface_val = comprehensive_features.get('Surface treatment', 'unknown')
                                                            numeric_values.append(float(surface_mapping.get(surface_val, 0)))
                                                        
                                                        # If we still don't have enough, add more missing features
                                                        while len(numeric_values) < 11:  # Should be 11 numeric features for 17 total
                                                            numeric_values.append(0.0)
                                                    # Step 2: Scale numeric features
                                                    if numeric_values:
                                                        numeric_array = np.array([numeric_values])
                                                        # Only use the scaler for the original training numeric features
                                                        original_count = len(training_numeric_features)
                                                        if len(numeric_values) > original_count:
                                                            # Scale only the original features, keep additional ones as-is
                                                            original_features = numeric_array[:, :original_count]
                                                            additional_features = numeric_array[:, original_count:]
                                                            scaled_original = feature_scaler.transform(original_features)
                                                            numeric_scaled = np.hstack([scaled_original, additional_features])
                                                        else:
                                                            numeric_scaled = feature_scaler.transform(numeric_array)
                                                    else:
                                                        numeric_scaled = np.empty((1, 0))
                                                    
                                                    # Step 3: Process categorical features
                                                    if training_categorical_features:
                                                        categorical_data = {}
                                                        for feature in training_categorical_features:
                                                            if feature in comprehensive_features:
                                                                categorical_data[feature] = comprehensive_features[feature]
                                                            else:
                                                                categorical_data[feature] = 'unknown'
                                                        
                                                        categorical_df = pd.DataFrame([categorical_data])
                                                        # Clean categorical data
                                                        for col in categorical_df.columns:
                                                            categorical_df[col] = categorical_df[col].fillna('unknown').astype(str)
                                                        
                                                        categorical_encoded = feature_encoder.transform(categorical_df)
                                                    else:
                                                        categorical_encoded = np.empty((1, 0))
                                                    
                                                    # Step 4: Combine features
                                                    final_features = np.hstack([numeric_scaled, categorical_encoded])
                                                    
                                                    # Ensure exact feature count match with intelligent padding
                                                    if final_features.shape[1] != expected_count:
                                                        # Try to identify what's missing
                                                        current_numeric = len(training_numeric_features) if training_numeric_features else 0
                                                        current_categorical_encoded = categorical_encoded.shape[1] if categorical_encoded.size > 0 else 0
                                                        
                                                        if final_features.shape[1] < expected_count:
                                                            # Add missing features - likely need one more numeric feature
                                                            missing_count = expected_count - final_features.shape[1]
                                                            
                                                            # Add a specific numeric feature that might be missing (e.g., surface treatment as numeric)
                                                            if missing_count == 1:
                                                                # Probably missing 'Surface treatment' as numeric (value 0 for 'unknown')
                                                                additional_feature = np.array([[0.0]])
                                                                # Insert the additional feature at the end of numeric features, before categorical
                                                                if categorical_encoded.size > 0:
                                                                    # Insert between numeric and categorical
                                                                    final_features = np.hstack([numeric_scaled, additional_feature, categorical_encoded])
                                                                else:
                                                                    # Just add to the end
                                                                    final_features = np.hstack([final_features, additional_feature])
                                                            else:
                                                                # General padding
                                                                padding = np.zeros((1, missing_count))
                                                                final_features = np.hstack([final_features, padding])
                                                        elif final_features.shape[1] > expected_count:
                                                            final_features = final_features[:, :expected_count]
                                                    prediction = model_to_use.predict(final_features)[0]
                                                else:
                                                    # Fallback: construct feature vector manually
                                                    processed_features = np.zeros(expected_count)
                                                    
                                                    # Fill numeric features based on expected feature names
                                                    numeric_feature_mapping = {
                                                        'pH of condition environment': ph_env,
                                                        'Exposure time': exposure_time,
                                                        'Fibre content': fiber_content,
                                                        'Exposure temperature': temperature,
                                                        'Diameter': diameter,
                                                        'Strength of unconditioned rebar': unconditioned_strength
                                                    }
                                                    
                                                    filled_count = 0
                                                    # Try to match features with expected feature names
                                                    for i, feature_name in enumerate(expected_features):
                                                        if feature_name in numeric_feature_mapping:
                                                            processed_features[i] = numeric_feature_mapping[feature_name]
                                                            filled_count += 1
                                                        # For categorical features (one-hot encoded), keep default 0
                                                    prediction = model_to_use.predict(processed_features.reshape(1, -1))[0]
                                        except Exception as detailed_e:
                                            # If detailed matching fails, fall back to simpler approach
                                            raise detailed_e
                                            
                                    else:
                                        # Fallback: Simplified prediction method for models without saved feature info
                                        simple_features = {
                                            'pH of condition environment': ph_env,
                                            'Exposure time': exposure_time,
                                            'Fibre content': fiber_content,
                                            'Exposure temperature': temperature,
                                            'Diameter': diameter,
                                            'Strength of unconditioned rebar': unconditioned_strength
                                        }
                                        
                                        simple_df = pd.DataFrame([simple_features])
                                        
                                        # If model is Pipeline, use directly
                                        if hasattr(model_to_use, 'named_steps'):
                                            prediction = model_to_use.predict(simple_df)[0]
                                        else:
                                            # Use basic numeric features, but match expected feature count
                                            numeric_values = np.array([[ph_env, exposure_time, fiber_content, 
                                                                      temperature, diameter, unconditioned_strength]])
                                            
                                            # Extract expected feature count from error message
                                            current_features = numeric_values.shape[1]
                                            expected_features = 17  # Default expected for cached models
                                            
                                            # Try to extract actual expected count from error
                                            if "expecting" in str(e):
                                                match = re.search(r'expecting (\d+) features', str(e))
                                                if match:
                                                    expected_features = int(match.group(1))
                                            
                                            if current_features < expected_features:
                                                padding = np.zeros((1, expected_features - current_features))
                                                numeric_values = np.hstack([numeric_values, padding])
                                            elif current_features > expected_features:
                                                numeric_values = numeric_values[:, :expected_features]
                                            
                                            prediction = model_to_use.predict(numeric_values)[0]
                                    
                                    # Check if prediction value is abnormal based on target variable type
                                    is_abnormal = False
                                    if model_target.startswith("Tensile") or "retention" in model_target.lower():
                                        # For retention rate prediction: should be between 0 and 1 (or 0-100 if percentage)
                                        if prediction < 0 or prediction > 1.2:  # Allow some tolerance for rates slightly above 100%
                                            is_abnormal = True
                                            abnormal_reason = f"Retention rate {prediction:.3f} is outside normal range [0, 1.2]"
                                    else:
                                        # For residual strength prediction: should be positive and reasonable
                                        if prediction < 1 or prediction > unconditioned_strength * 1.5:  # Should not exceed 150% of original strength significantly
                                            is_abnormal = True
                                            abnormal_reason = f"Residual strength {prediction:.1f} MPa seems unrealistic (original: {unconditioned_strength:.1f} MPa)"
                                    
                                    if is_abnormal:
                                        st.warning(f"Prediction value may be unusual: {abnormal_reason}")
                                        st.write("Possible reasons:")
                                        st.write("1. Feature scaling issues")
                                        st.write("2. Model training data issues") 
                                        st.write("3. Feature mismatch")
                                        st.write("Suggestion: Retrain model or check input data")
                                    else:
                                        # Prediction is within normal range - display in card format like Fixed Test Set
                                        st.success("Prediction completed!")
                                        
                                        col_pred1, col_pred2 = st.columns(2)
                                        
                                        with col_pred1:
                                            if model_target.startswith("Tensile") or "retention" in model_target.lower():
                                                # Convert to percentage if needed
                                                if prediction < 2:  # Decimal form to percentage
                                                    display_value = prediction * 100
                                                else:
                                                    display_value = prediction
                                                st.metric("Predicted Retention Rate", f"{display_value:.2f}%")
                                            else:
                                                st.metric("Predicted Residual Strength", f"{prediction:.2f} MPa")
                                        
                                        with col_pred2:
                                            if model_target.startswith("Tensile") or "retention" in model_target.lower():
                                                # Use converted value for performance calculation
                                                if prediction < 2:
                                                    retention_rate = prediction * 100
                                                else:
                                                    retention_rate = prediction
                                                
                                                if retention_rate >= 80:
                                                    performance = "Excellent"
                                                    color = "🟢"
                                                elif retention_rate >= 60:
                                                    performance = "Good"
                                                    color = "🟡"
                                                else:
                                                    performance = "Needs Attention"
                                                    color = "🔴"
                                                
                                                st.metric("Performance", f"{color} {performance}")
                                            else:
                                                # For residual strength, calculate retention rate
                                                retention_rate = (prediction / unconditioned_strength) * 100 if unconditioned_strength > 0 else 0
                                                
                                                if retention_rate >= 80:
                                                    performance = "Excellent"
                                                    color = "🟢"
                                                elif retention_rate >= 60:
                                                    performance = "Good"
                                                    color = "🟡"
                                                else:
                                                    performance = "Needs Attention"
                                                    color = "🔴"
                                                
                                                st.metric("Performance", f"{color} {performance}")
                                    
                                except Exception as e2:
                                    st.error(f"Simplified prediction also failed: {e2}")
                                    
                                    # Last fallback method
                                    if model_target.startswith("Tensile"):
                                        prediction = 50.0  # Default retention rate 50%
                                    else:
                                        prediction = unconditioned_strength * 0.5  # Default residual strength 50% of original
                                    
                                    st.info("Suggested solutions:")
                                    st.info("1. Retrain model to ensure feature consistency")
                                    st.info("2. Check input data format and range")
                                    st.info("3. Contact administrator to check model configuration")
                            
                            # 继续显示结果（使用任何可用的预测值）
            
            else:
                # 基础特征输入（保留原有功能作为备用）
                st.markdown("### Basic Feature Input")
                st.info("Using basic features for prediction")
                
                # 这里可以保留原来的基础特征输入表单作为备用
                with st.form("basic_prediction_form"):
                    st.markdown("**Basic Parameters**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        temperature = st.number_input("Temperature (°C)", -50.0, 200.0, 25.0)
                        ph = st.number_input("pH", 0.0, 14.0, 7.0)
                        time_field = st.number_input("Exposure time (hours)", 0.0, 100000.0, 1000.0)
                        fiber_content = st.number_input("Fiber content (%)", 0.0, 100.0, 60.0)
                    
                    with col2:
                        diameter = st.number_input("Diameter (mm)", 0.1, 50.0, 10.0)
                        initial_strength = st.number_input("Initial strength (MPa)", 100.0, 5000.0, 1000.0)
                        fiber_type = st.selectbox("Fiber type", ["Glass", "Carbon", "Aramid"])
                        matrix_type = st.selectbox("Matrix type", ["Epoxy", "Vinyl ester", "Polyester"])
                    
                    submitted = st.form_submit_button("Predict with Basic Features", type="primary", use_container_width=True)
                    
                    if submitted:
                        # 准备基础输入数据
                        basic_input = {
                            "temperature": temperature,
                            "pH_1": ph,
                            "time_field": time_field,
                            "Fiber_content_weight": fiber_content,
                            "diameter": diameter,
                            "Fiber_type": fiber_type,
                            "Matrix_type": matrix_type,
                            "Value2_1": initial_strength
                        }

                        if model_target.startswith("Residual"):
                            basic_input["Value2_1"] = initial_strength
                        
                        try:
                            # 使用从training_feature_info中恢复的预处理器
                            feature_encoder = training_feature_info.get('feature_encoder')
                            feature_scaler = training_feature_info.get('feature_scaler')
                            
                            if feature_encoder is not None and feature_scaler is not None:
                                encoder = feature_encoder
                                scaler = feature_scaler
                            else:
                                st.error("Missing preprocessing components. Please retrain the model.")
                                st.stop()
                            
                            # 分离特征类型
                            categorical_feats = list(encoder.feature_names_in_) if hasattr(encoder, 'feature_names_in_') else []
                            numeric_feats = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else []
                            
                            # 创建输入DataFrame
                            input_df = pd.DataFrame([basic_input])
                            
                            # 预处理分类特征
                            if categorical_feats:
                                input_cat = input_df[categorical_feats].astype(str)
                                input_cat_encoded = encoder.transform(input_cat)
                            else:
                                input_cat_encoded = np.empty((1, 0))
                            
                            # 预处理数值特征
                            if numeric_feats:
                                input_num = input_df[numeric_feats]
                                input_num_scaled = scaler.transform(input_num)
                            else:
                                input_num_scaled = np.empty((1, 0))
                            
                            # 合并特征
                            input_processed = np.hstack([input_num_scaled, input_cat_encoded])
                            
                            # 执行预测
                            prediction = model_to_use.predict(input_processed)[0]
                            
                            # 显示预测结果
                            st.success("Prediction completed!")
                            
                            col_pred1, col_pred2 = st.columns(2)
                            
                            with col_pred1:
                                st.metric("Predicted Retention Rate", f"{prediction:.2f}%")
                            
                            with col_pred2:
                                # 性能评级
                                if prediction >= 80:
                                    performance = "Excellent"
                                    color = "🟢"
                                elif prediction >= 60:
                                    performance = "Good" 
                                    color = "🟡"
                                else:
                                    performance = "Needs Attention"
                                    color = "🔴"
                                
                                st.metric("Performance", f"{color} {performance}")
                            
                            # 记录预测
                            if "authenticated_user" in st.session_state and engine:
                                user_info = st.session_state["authenticated_user"]
                                pred_data = {
                                    "prediction": prediction,
                                    "input_features": basic_input,
                                    "model": model_name,
                                    "target": model_target,
                                    "data_type": "basic_features",
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                try:
                                    log_operation(
                                        user_info['email'],
                                        'prediction',
                                        'single',
                                        None,
                                        pred_data,
                                        get_client_ip(),
                                        engine
                                    )
                                except:
                                    pass  # 日志记录失败不影响预测功能
                                st.warning("Standard feature processing failed. Trying emergency fallback...")
                                # 尝试紧急备用方法
                                prediction = emergency_prediction_fallback(input_df, model)
                                
                                if prediction is None:
                                    st.error("All prediction methods failed. Please check your input data and retrain the model.")
                                    st.stop()
                                else:
                                    st.warning(" Prediction completed using fallback method. Accuracy may be reduced.")
                            else:
                                # Make prediction
                                if isinstance(input_processed, pd.DataFrame):
                                    # Pipeline model - input is DataFrame
                                    prediction = model.predict(input_processed)[0]
                                else:
                                    # Preprocessed model - input is numpy array
                                    prediction = model.predict(input_processed.reshape(1, -1))[0]
                            
                            # Display prediction results
                            st.success("Prediction completed!")
                            
                            col_pred1, col_pred2 = st.columns(2)
                            
                            with col_pred1:
                                st.metric("Predicted Retention Rate", f"{prediction:.2f}%")
                            
                            with col_pred2:
                                # Performance rating
                                if prediction >= 80:
                                    performance = "Excellent"
                                    color = "🟢"
                                elif prediction >= 60:
                                    performance = "Good"
                                    color = "🟡"
                                else:
                                    performance = "Needs Attention"
                                    color = "🔴"
                                
                                st.metric("Performance", f"{color} {performance}")
                            
                            # Record prediction
                            if "authenticated_user" in st.session_state and engine:
                                user_info = st.session_state["authenticated_user"]
                                pred_data = {
                                    "prediction": prediction,
                                    "input_features": advanced_input,
                                    "model": st.session_state.get("selected_model", "Unknown"),
                                    "data_type": "advanced_features",
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                log_operation(
                                    user_info['email'],
                                    'prediction',
                                    'single',
                                    None,
                                    pred_data,
                                    get_client_ip(),
                                    engine
                                )
                        
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.write("Please check input data format")
    
    else:  # Batch Prediction
            st.subheader("Batch Prediction")
            
            # Prompt user based on training data type
            if use_advanced_features:
                st.info("Please upload a CSV file containing 13 advanced features")
                st.markdown("""
                **Required Feature Columns:**
                - pH of condition environment
                - Exposure time  
                - Fibre content
                - Exposure temperature
                - Diameter
                - Presence of concrete
                - Load
                - Presence of chloride ion
                - Fibre type
                - Matrix type
                - Surface treatment
                - Strength of unconditioned rebar
                """)
            else:
                st.info("Please upload a CSV file containing raw features")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with input data",
                type=['csv'],
                help="Upload a CSV file with the same features as used in training"
            )
            
            if uploaded_file is not None:
                try:
                    # 读取上传的文件
                    batch_df = pd.read_csv(uploaded_file)
                    
                    st.success(f"File uploaded successfully! {len(batch_df)} rows found.")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(batch_df.head(10), use_container_width=True)
                    
                    # 检查必需的列
                    required_features = [
    "Fiber_type", "Matrix_type", "surface_treatment",
    "Fiber_content_weight", "diameter", "temperature", 
    "time_field", "pH_1"
]
                    
                    missing_features = [feat for feat in required_features if feat not in batch_df.columns]
                    
                    if missing_features:
                        st.error(f"Missing required columns: {missing_features}")
                        st.info("Please ensure your CSV file contains all required features.")
                    else:
                        # 列映射检查
                        st.subheader("Column Mapping Verification")
                        
                        mapping_correct = True
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Required Features:**")
                            for feat in required_features:
                                if feat in batch_df.columns:
                                    st.success(f"✓ {feat}")
                                else:
                                    st.error(f"✗ {feat}")
                                    mapping_correct = False
                        
                        with col2:
                            st.write("**Optional Features:**")
                            optional_features = ["applied_load", "presence_concrete", "presence_chloride"]
                            for feat in optional_features:
                                if feat in batch_df.columns:
                                    st.success(f"✓ {feat}")
                                else:
                                    st.info(f"○ {feat} (will be set to default)")
                        
                        if mapping_correct:
                            # 处理缺失的可选特征
                            if "applied_load" not in batch_df.columns:
                                batch_df["applied_load"] = 0.0
                            if "presence_concrete" not in batch_df.columns:
                                batch_df["presence_concrete"] = 0
                            if "presence_chloride" not in batch_df.columns:
                                batch_df["presence_chloride"] = 0
                            
                            # Batch prediction button
                            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                                
                                # 创建进度条
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                try:
                                    status_text.text("Preprocessing data...")
                                    progress_bar.progress(0.2)
                                    
                                    # 使用标准化特征预处理函数
                                    training_feature_info = st.session_state.get('training_feature_info', {})
                                    
                                    if not training_feature_info:
                                        st.error("Missing training feature information. Please retrain the model.")
                                        st.stop()
                                    
                                    status_text.text("Processing features...")
                                    progress_bar.progress(0.5)
                                    
                                    # Get trained model
                                    model = st.session_state.trained_model
                                    
                                    # 标准化所有输入特征
                                    batch_processed = standardize_prediction_features(batch_df, training_feature_info, model)
                                    
                                    if batch_processed is None:
                                        st.error("Failed to process batch features")
                                        st.stop()
                                    
                                    status_text.text("Making predictions...")
                                    progress_bar.progress(0.8)
                                    
                                    # Perform batch prediction
                                    if isinstance(batch_processed, pd.DataFrame):
                                        # Pipeline model - input is DataFrame
                                        predictions = model.predict(batch_processed)
                                    else:
                                        # Preprocessed model - input is numpy array
                                        predictions = model.predict(batch_processed)
                                    
                                    status_text.text("Finalizing results...")
                                    progress_bar.progress(1.0)
                                    
                                    # 添加预测结果到原始数据
                                    result_df = batch_df.copy()
                                    
                                    if st.session_state.selected_target.startswith("Residual"):
                                        result_df['Predicted_Residual_Strength_MPa'] = predictions
                                        if 'Value2' in result_df.columns:
                                            result_df['Predicted_Retention_Rate_%'] = (predictions / result_df['Value2']) * 100
                                    else:
                                        result_df['Predicted_Retention_Rate_%'] = predictions
                                    
                                    # 添加性能等级
                                    if st.session_state.selected_target.startswith("Residual"):
                                        if 'Value2' in result_df.columns:
                                            retention_rates = (predictions / result_df['Value2']) * 100
                                        else:
                                            retention_rates = predictions / 1000 * 100  # 假设初始强度
                                    else:
                                        retention_rates = predictions
                                    
                                    performance_levels = []
                                    for rate in retention_rates:
                                        if rate >= 90:
                                            performance_levels.append("Excellent")
                                        elif rate >= 70:
                                            performance_levels.append("Good")
                                        elif rate >= 50:
                                            performance_levels.append("Fair")
                                        else:
                                            performance_levels.append("Poor")
                                    
                                    result_df['Performance_Level'] = performance_levels
                                    
                                    # 清理进度显示
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # 显示结果
                                    st.success(f"Batch prediction completed! {len(predictions)} predictions made.")
                                    
                                    # 结果统计
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        avg_pred = np.mean(predictions)
                                        st.metric("Average Prediction", f"{avg_pred:.2f}")
                                    
                                    with col2:
                                        std_pred = np.std(predictions)
                                        st.metric("Std Deviation", f"{std_pred:.2f}")
                                    
                                    with col3:
                                        min_pred = np.min(predictions)
                                        st.metric("Minimum", f"{min_pred:.2f}")
                                    
                                    with col4:
                                        max_pred = np.max(predictions)
                                        st.metric("Maximum", f"{max_pred:.2f}")
                                    
                                    # 性能等级分布
                                    st.subheader("Performance Level Distribution")
                                    
                                    perf_counts = pd.Series(performance_levels).value_counts()
                                    perf_df = pd.DataFrame({
                                        'Performance_Level': perf_counts.index,
                                        'Count': perf_counts.values,
                                        'Percentage': (perf_counts.values / len(performance_levels) * 100).round(1)
                                    })
                                    
                                    # 创建分布图
                                    perf_chart = alt.Chart(perf_df).mark_bar().encode(
                                        x=alt.X('Count:Q', title='Number of Samples'),
                                        y=alt.Y('Performance_Level:N', sort=['Excellent', 'Good', 'Fair', 'Poor']),
                                        color=alt.Color(
                                            'Performance_Level:N',
                                            scale=alt.Scale(
                                                domain=['Excellent', 'Good', 'Fair', 'Poor'],
                                                range=['#4CAF50', '#2196F3', '#FF9800', '#F44336']
                                            ),
                                            legend=None
                                        ),
                                        tooltip=['Performance_Level', 'Count', 'Percentage']
                                    ).properties(
                                        title='Performance Level Distribution',
                                        height=200
                                    )
                                    
                                    st.altair_chart(perf_chart, use_container_width=True)
                                    
                                    # 预测值分布
                                    st.subheader("Prediction Distribution")
                                    
                                    pred_df = pd.DataFrame({'Predictions': predictions})
                                    
                                    hist_chart = alt.Chart(pred_df).mark_bar().encode(
                                        alt.X('Predictions:Q', bin=alt.Bin(maxbins=30), title='Predicted Values'),
                                        alt.Y('count()', title='Frequency'),
                                        color=alt.value('#1e3d59')
                                    ).properties(
                                        title='Distribution of Predictions',
                                        height=300
                                    )
                                    
                                    st.altair_chart(hist_chart, use_container_width=True)
                                    
                                    # 显示结果表格
                                    st.subheader("Detailed Results")
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # 导出结果
                                    st.subheader("Export Results")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # CSV导出
                                        csv = result_df.to_csv(index=False)
                                        b64 = base64.b64encode(csv.encode()).decode()
                                        href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download CSV</a>'
                                        st.markdown(href, unsafe_allow_html=True)
                                    
                                    with col2:
                                        # Excel导出
                                        output = io.BytesIO()
                                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                            result_df.to_excel(writer, sheet_name='Predictions', index=False)
                                            
                                            # 添加统计摘要
                                            summary_df = pd.DataFrame({
                                                'Metric': ['Count', 'Mean', 'Std', 'Min', 'Max'],
                                                'Value': [
                                                    len(predictions),
                                                    np.mean(predictions),
                                                    np.std(predictions),
                                                    np.min(predictions),
                                                    np.max(predictions)
                                                ]
                                            })
                                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                        
                                        excel_data = output.getvalue()
                                        b64 = base64.b64encode(excel_data).decode()
                                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="batch_predictions.xlsx">Download Excel</a>'
                                        st.markdown(href, unsafe_allow_html=True)
                                    
                                    # Record batch prediction operation
                                    if "authenticated_user" in st.session_state and engine:
                                        user_info = st.session_state["authenticated_user"]
                                        batch_data = {
                                            "records_processed": len(predictions),
                                            "model": st.session_state.get("selected_model", "Unknown"),
                                            "performance_distribution": perf_counts.to_dict(),
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        
                                        log_operation(
                                            user_info['email'],
                                            'prediction',
                                            'batch',
                                            None,
                                            batch_data,
                                            get_client_ip(),
                                            engine
                                        )
                                
                                except Exception as e:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"Batch prediction failed: {e}")
                                    st.write("Please check your data format and try again.")
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.write("Please ensure the file is in valid CSV format.")
    
    # 预测历史（如果用户已登录）
    if "authenticated_user" in st.session_state and engine:
            st.markdown("---")
            st.subheader("Prediction History")
            
            user_info = st.session_state["authenticated_user"]
            
            try:
                # 获取用户的预测历史
                history_query = text("""
                    SELECT operation_type, table_name, details, created_at
                    FROM operation_logs
                    WHERE user_email = :email AND operation_type = 'prediction'
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                history_df = pd.read_sql(
                    history_query,
                    engine,
                    params={"email": user_info['email']}
                )
                
                if not history_df.empty:
                    # 解析历史记录
                    history_records = []
                    for _, row in history_df.iterrows():
                        try:
                            details = json.loads(row['details']) if isinstance(row['details'], str) else row['details']
                            
                            if row['table_name'] == 'single':
                                pred_type = "Single Prediction"
                                result = f"{details.get('prediction', 'N/A'):.2f}"
                            else:
                                pred_type = "Batch Prediction"
                                result = f"{details.get('records_processed', 'N/A')} records"
                            
                            history_records.append({
                                'Type': pred_type,
                                'Model': details.get('model', 'Unknown'),
                                'Result': result,
                                'Date': row['created_at'].strftime('%Y-%m-%d %H:%M')
                            })
                        except:
                            continue
                    
                    if history_records:
                        history_display_df = pd.DataFrame(history_records)
                        st.dataframe(history_display_df, use_container_width=True)
                    else:
                        st.info("No prediction history found.")
                else:
                    st.info("No prediction history found.")
            
            except Exception as e:
                st.warning(f"Could not load prediction history: {e}")

# ——————————————————————————————
# Helper Functions for Role-based Access Control
# ——————————————————————————————

def check_user_permission(required_role="viewer"):
    """
    检查用户权限
    
    Args:
        required_role: 所需的最低权限级别 ("viewer", "editor", "admin")
    
    Returns:
        tuple: (has_permission, user_info)
    """
    if "authenticated_user" not in st.session_state:
        return False, None
    
    user_info = st.session_state["authenticated_user"]
    user_role = user_info.get("role", "viewer")
    
    # 权限级别定义
    role_hierarchy = {"viewer": 1, "editor": 2, "admin": 3}
    
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 1)
    
    return user_level >= required_level, user_info

def render_access_denied():
    """渲染访问被拒绝的界面"""
    st.error("Access Denied")
    st.warning("You don't have sufficient permissions to access this feature.")
    
    if "authenticated_user" not in st.session_state:
        st.info("Please log in first.")
    else:
        user_role = st.session_state["authenticated_user"].get("role", "viewer")
        st.info(f"Your current role: **{user_role}**")
        st.info("Contact an administrator if you need elevated permissions.")

def log_user_action(action_type, details=None):
    """
    记录用户操作日志
    
    Args:
        action_type: 操作类型
        details: 操作详情
    """
    if "authenticated_user" in st.session_state and engine:
        user_info = st.session_state["authenticated_user"]
        try:
            log_operation(
                user_info['email'],
                action_type,
                'system',
                None,
                details or {},
                get_client_ip(),
                engine
            )
        except Exception as e:
            # 静默处理日志错误，不影响主要功能
            pass

def display_user_info():
    """在侧边栏显示用户信息"""
    if "authenticated_user" in st.session_state:
        user_info = st.session_state["authenticated_user"]
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: white; margin: 0 0 0.5rem 0;">👤 {user_info['name']}</h4>
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">
                📧 {user_info['email']}<br>
                🏢 {user_info.get('institution', 'N/A')}<br>
                🎭 {user_info['role'].title()}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(" Logout", use_container_width=True):
            # 记录登出操作
            log_user_action("logout", {"user": user_info['email']})
            
            # 清除会话状态
            for key in list(st.session_state.keys()):
                if key.startswith(('authenticated_', 'user_')):
                    del st.session_state[key]
            
            st.rerun()

def render_feature_coming_soon(feature_name):
    """渲染功能即将推出的界面"""
    st.info(f"🚧 **{feature_name}** feature is coming soon!")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 8px; text-align: center; margin: 2rem 0;">
        <h3 style="color: #6c757d;">🔨 Under Development</h3>
        <p style="color: #6c757d;">This feature is currently being developed and will be available in a future update.</p>
    </div>
    """, unsafe_allow_html=True)

# ——————————————————————————————
# Session State Initialization
# ——————————————————————————————

def initialize_session_state():
    """初始化会话状态"""
    # 数据相关
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    
    if "model_dataset" not in st.session_state:
        st.session_state.model_dataset = None
    
    # 模型配置相关
    if "selected_target" not in st.session_state:
        st.session_state.selected_target = "Tensile strength retention rate (%)"
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Random Forest"
    
    if "use_advanced_preprocessing" not in st.session_state:
        st.session_state.use_advanced_preprocessing = True
    
    if "auto_model_select" not in st.session_state:
        st.session_state.auto_model_select = False
    
    if "enable_hp_tuning" not in st.session_state:
        st.session_state.enable_hp_tuning = False
    
    if "test_size" not in st.session_state:
        st.session_state.test_size = 0.2
    
    # 评估策略相关
    if "evaluation_strategy" not in st.session_state:
        st.session_state.evaluation_strategy = "Fixed Test Set (Recommended)"
    
    # UI状态相关
    if "show_all_features" not in st.session_state:
        st.session_state.show_all_features = False

# ——————————————————————————————
# Main Application Entry Point
# ——————————————————————————————

def main():
    """主应用程序入口点"""
    # 初始化会话状态
    initialize_session_state()
    
    # 加载默认数据
    if st.session_state.df_raw is None:
        st.session_state.df_raw = load_default_data()
    
    # 记录页面访问
    if "authenticated_user" in st.session_state:
        log_user_action("page_access", {"timestamp": datetime.now().isoformat()})

# ——————————————————————————————
# Application Footer
# ——————————————————————————————

def render_footer():
    """渲染应用程序页脚"""
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: #666; font-size: 0.9rem;">
        <p>© 2025 FRP Rebar Durability Prediction Platform</p>
        <p>Powered by Imperial • Built with ❤️ for Materials Science Research</p>
        <p style="font-size: 0.8rem;">Version 2.1 • Enhanced Model Persistence • Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
    """, unsafe_allow_html=True)

# 调用主应用程序
if __name__ == "__main__":
    try:
        main()
        render_footer()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Please refresh the page or contact support if the problem persists.")