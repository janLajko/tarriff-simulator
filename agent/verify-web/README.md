# Web审核平台

一个简洁的Web界面，用于审核HTSUS Chapter 99 Notes的LLM提取结果差异。

## 功能特性

- 显示Note 33/36/37/38的审核状态
- 逐个展示OpenAI与Grok的差异
- 支持选择接受OpenAI、Grok或手动输入
- 生成统一的JSON文件用于后续处理
- 键盘快捷键支持（1=选择第一个选项，2=选择第二个选项，左右箭头切换）

## 安装和运行

### 方式1：使用启动脚本

```bash
cd agent/verify-web
./run.sh
```

### 方式2：手动运行

```bash
cd agent/verify-web

# 安装依赖
pip install -r requirements.txt

# 启动服务器
python3 app.py
```

服务器启动后，在浏览器访问：http://localhost:5000

## 使用流程

1. **查看Note状态**
   - 红色：有冲突需要审核
   - 绿色：模型结果一致
   - 灰色：未处理

2. **开始审核**
   - 点击有冲突的Note卡片
   - 系统会逐个展示差异项

3. **解决冲突**
   - 措施冲突：选择接受或拒绝
   - Scope冲突：选择使用OpenAI、Grok或并集

4. **生成统一文件**
   - 所有冲突解决后，点击"生成统一文件"
   - 文件保存在：`agent/othercharpter-agent/output/note{X}_unified.json`

## 文件结构

```
verify-web/
├── app.py              # Flask后端服务
├── templates/
│   └── index.html      # 前端界面
├── requirements.txt    # Python依赖
├── run.sh             # 启动脚本
└── README.md          # 本文档
```

## 输出文件

审核完成后会生成：
- `note{X}_unified.json` - 统一后的措施数据，包含所有已解决的冲突

## 注意事项

- 确保已运行过 `othercharpter.py` 生成了比对文件
- 审核数据存储在内存中，刷新页面会丢失进度
- 生成的unified.json可用于后续的数据库入库操作