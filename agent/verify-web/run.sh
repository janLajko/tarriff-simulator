#!/bin/bash

# 启动Web审核平台

echo "Starting HTSUS Chapter 99 审核平台..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 启动Flask服务器
echo "Starting Flask server on http://localhost:5000"
python3 app.py