-- FRP钢筋耐久性预测系统 - 数据库初始化脚本
-- 请在MySQL中执行此脚本

-- 创建数据库
CREATE DATABASE IF NOT EXISTS haigui_database 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE haigui_database;

-- 创建用户表 (如果需要用户管理功能)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'user') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 创建预测记录表
CREATE TABLE IF NOT EXISTS prediction_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    input_parameters JSON NOT NULL,
    prediction_result FLOAT NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 创建实验记录表
CREATE TABLE IF NOT EXISTS experiment_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    parameters JSON,
    results JSON,
    r2_score FLOAT,
    rmse FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建数据上传记录表
CREATE TABLE IF NOT EXISTS data_upload_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size INT,
    records_imported INT,
    import_status ENUM('pending', 'success', 'failed') DEFAULT 'pending',
    error_message TEXT,
    uploaded_by INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
);

-- 插入默认管理员用户 (密码: admin123, 请在生产环境中修改)
INSERT IGNORE INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewgohRl4YhL1p4ma', 'admin');

-- 创建索引以提高查询性能
CREATE INDEX idx_prediction_records_created_at ON prediction_records(created_at);
CREATE INDEX idx_experiment_records_created_at ON experiment_records(created_at);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- 显示创建的表
SHOW TABLES;

-- 显示数据库信息
SELECT 
    SCHEMA_NAME as '数据库名',
    DEFAULT_CHARACTER_SET_NAME as '字符集',
    DEFAULT_COLLATION_NAME as '排序规则'
FROM information_schema.SCHEMATA 
WHERE SCHEMA_NAME = 'haigui_database';

-- 完成提示
SELECT '数据库初始化完成! 现在可以开始导入数据了。' as '状态';