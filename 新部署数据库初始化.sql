-- FRP预测平台 - 新部署数据库初始化脚本
-- 这是为新部署准备的完整数据库结构

-- ====================================
-- 数据库创建和配置
-- ====================================

-- 注意：请根据实际情况修改数据库名称
CREATE DATABASE IF NOT EXISTS frp_platform 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE frp_platform;

-- ====================================
-- 用户管理表
-- ====================================

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- ====================================
-- FRP数据表（主要数据表）
-- ====================================

CREATE TABLE IF NOT EXISTS frp_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- 基本信息
    fiber_type VARCHAR(50),           -- 纤维类型
    matrix_type VARCHAR(50),          -- 基体类型
    treatment VARCHAR(100),           -- 处理方式
    
    -- 环境条件
    temperature DECIMAL(8,2),         -- 温度
    humidity DECIMAL(8,2),           -- 湿度
    ph_value DECIMAL(4,2),           -- pH值
    solution_type VARCHAR(100),       -- 溶液类型
    concentration DECIMAL(10,4),      -- 浓度
    
    -- 时间参数
    duration_days INT,               -- 持续天数
    duration_hours DECIMAL(10,2),    -- 持续小时数
    
    -- 性能参数
    tensile_strength DECIMAL(10,2),   -- 拉伸强度
    retention_ratio DECIMAL(5,2),    -- 保留比率
    
    -- 元数据
    source VARCHAR(200),             -- 数据来源
    reference VARCHAR(500),          -- 参考文献
    notes TEXT,                      -- 备注
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- 索引优化
    INDEX idx_fiber_type (fiber_type),
    INDEX idx_temperature (temperature),
    INDEX idx_duration (duration_days),
    INDEX idx_strength (tensile_strength)
);

-- ====================================
-- 预测记录表
-- ====================================

CREATE TABLE IF NOT EXISTS prediction_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    
    -- 输入参数
    input_parameters JSON,           -- 输入参数（JSON格式）
    model_type VARCHAR(50),          -- 使用的模型类型
    model_name VARCHAR(100),         -- 模型名称
    
    -- 预测结果
    prediction_result DECIMAL(10,4), -- 预测结果
    confidence_score DECIMAL(5,4),   -- 置信度
    
    -- 时间记录
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_model_type (model_type),
    INDEX idx_created_at (created_at)
);

-- ====================================
-- 自定义模型表
-- ====================================

CREATE TABLE IF NOT EXISTS custom_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    
    -- 模型信息
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_file_path VARCHAR(255),
    
    -- 模型描述
    description TEXT,
    input_features JSON,             -- 输入特征列表
    performance_metrics JSON,        -- 性能指标
    
    -- 状态
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_model_type (model_type),
    INDEX idx_active (is_active)
);

-- ====================================
-- 系统配置表
-- ====================================

CREATE TABLE IF NOT EXISTS system_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    description VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_config_key (config_key)
);

-- ====================================
-- 插入默认配置
-- ====================================

INSERT INTO system_config (config_key, config_value, description) VALUES
('platform_name', 'FRP纤维增强聚合物耐久性预测平台', '平台名称'),
('version', '2.0.0', '平台版本'),
('max_upload_size', '50', '最大上传文件大小(MB)'),
('default_model_type', 'RandomForest', '默认模型类型'),
('enable_registration', 'true', '是否允许用户注册'),
('maintenance_mode', 'false', '维护模式开关')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_at = CURRENT_TIMESTAMP;

-- ====================================
-- 创建默认管理员用户（可选 - 请修改密码！）
-- ====================================

-- 注意：这是示例用户，密码是 "admin123" 的哈希值
-- 部署后请立即修改密码或删除此用户
INSERT INTO users (username, password_hash, email, full_name, is_active) VALUES
('admin', 'pbkdf2:sha256:260000$xyz123$abcd...', 'admin@example.com', '系统管理员', TRUE)
ON DUPLICATE KEY UPDATE username = username;

-- ====================================
-- 权限和安全设置
-- ====================================

-- 创建只读用户（用于数据查询）
-- CREATE USER IF NOT EXISTS 'frp_reader'@'%' IDENTIFIED BY 'your_password_here';
-- GRANT SELECT ON frp_platform.* TO 'frp_reader'@'%';

-- 创建应用用户（用于应用程序）
-- CREATE USER IF NOT EXISTS 'frp_app'@'%' IDENTIFIED BY 'your_app_password_here';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON frp_platform.* TO 'frp_app'@'%';

-- 刷新权限
-- FLUSH PRIVILEGES;

-- ====================================
-- 性能优化
-- ====================================

-- 设置字符集和排序规则
ALTER DATABASE frp_platform CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 优化表设置
ALTER TABLE users CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE frp_data CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE prediction_records CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE custom_models CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE system_config CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ====================================
-- 验证和完成
-- ====================================

-- 显示所有创建的表
SHOW TABLES;

-- 显示数据库信息
SELECT 
    SCHEMA_NAME as '数据库名',
    DEFAULT_CHARACTER_SET_NAME as '字符集',
    DEFAULT_COLLATION_NAME as '排序规则'
FROM information_schema.SCHEMATA 
WHERE SCHEMA_NAME = 'frp_platform';

-- 显示表结构统计
SELECT 
    TABLE_NAME as '表名',
    TABLE_ROWS as '行数',
    DATA_LENGTH as '数据大小(字节)',
    INDEX_LENGTH as '索引大小(字节)'
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'frp_platform'
ORDER BY TABLE_NAME;

-- 完成提示
SELECT 
    'FRP预测平台数据库初始化完成！' as '状态',
    COUNT(*) as '创建表数量'
FROM information_schema.tables 
WHERE table_schema = 'frp_platform';

-- ====================================
-- 数据导入说明
-- ====================================

/*
数据导入步骤：
1. 如果您有现有的FRP数据，请使用INSERT语句导入到frp_data表
2. 可以通过平台的Excel导入功能批量导入数据
3. 建议先导入少量测试数据验证系统功能

示例数据插入：
INSERT INTO frp_data (
    fiber_type, matrix_type, temperature, duration_days, 
    tensile_strength, retention_ratio, source
) VALUES (
    'Glass', 'Epoxy', 25.0, 30, 
    1200.5, 95.2, '测试数据'
);
*/