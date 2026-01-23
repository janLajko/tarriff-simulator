# Web审核平台设计文档

## 一、系统概述

Web审核平台用于解决HTSUS Chapter 99 Notes数据提取过程中，OpenAI和Grok两个LLM模型产生不一致结果时的人工审核需求。平台提供直观的界面让操作员快速决策，并管理从模型执行到数据入库的完整流程。

## 二、系统整体流程

```
开始
  ↓
执行LLM模型（OpenAI + Grok并行）
  ↓
生成比对结果
  ↓
[判断] 是否有差异？
  ├─ 否 → 直接生成unified.json → 进入数据库比对
  ├─ 是 → 进入Web审核界面
  │       ↓
  │     人工解决冲突
  │       ↓
  │     点击“下一步”
  │       ↓
  └───→ 生成unified.json
        ↓
      与数据库比对
        ↓
      [判断] 是否有数据库差异？
        ├─ 否 → 直接入库 → 完成
        ├─ 是 → 显示数据库差异
        │       ↓
        │     人工决策（更新/保持/跳过）
        │       ↓
        │     点击“更新数据库”
        │       ↓
        └───→ 执行数据库操作
              ↓
            完成
```

## 三、数据文件流转

```
初始文件：
├── note33_openai.json     # OpenAI生成的结果
├── note33_grok.json        # Grok生成的结果
├── note33_llm_compare.json # 差异比对结果

审核过程生成：
├── note33_resolutions.json # 冲突解决记录
├── note33_unified.json     # 统一后的数据
├── note33_db_compare.json  # 数据库比对结果

最终输出：
├── note33_final.json       # 最终入库数据
├── note33_audit_log.json   # 审核日志
```

## 四、页面设计

### 4.1 审核启动页

```
页面路径：/review/start

显示内容：
- Note列表（33, 36, 37, 38）
- 每个Note的状态
  * 未处理（灰色）
  * 有冲突待审核（红色）
  * 审核中（黄色）
  * 已完成（绿色）

操作：
- 点击某个Note → 进入审核流程
```

### 4.2 冲突审核页

```
页面路径：/review/note/33/conflicts

布局：
┌─────────────────────────────────────┐
│ 进度条：24/30 已解决                │
├─────────────────────────────────────┤
│ 当前冲突：                          │
│ Heading: 9903.94.40                │
│ ┌─────────┬─────────┬──────────┐   │
│ │ OpenAI  │  Grok   │  Manual  │   │
│ ├─────────┼─────────┼──────────┤   │
│ │日期:null│9/16/2025│ [input]  │   │
│ │潜在:否  │  是     │ [select] │   │
│ └─────────┴─────────┴──────────┘   │
│ [选择OpenAI] [选择Grok] [手动输入]  │
├─────────────────────────────────────┤
│ [上一个] [下一个] [下一步 →]         │
└─────────────────────────────────────┘

操作流程：
1. 用户选择某个选项（OpenAI/Grok/Manual）
2. 系统记录到resolutions中
3. 自动跳转下一个冲突
4. 全部解决后，“下一步”按钮变为可用
```

### 4.3 Scope差异页

```
页面路径：/review/note/33/scopes

显示：
- Heading分组显示
- 每组显示：
  * OpenAI有X个codes
  * Grok有Y个codes
  * 共同有Z个codes

操作选项：
- [接受全部OpenAI]
- [接受全部Grok]
- [接受并集]
- [接受交集]
- [手动选择] → 展开详细列表勾选
```

### 4.4 数据库差异审核页

```
页面路径：/review/note/33/db-conflicts

显示格式：

┌─────────────────────────────────────┐
│ 数据库比对结果                       │
├─────────────────────────────────────┤
│ 新增: 5条                           │
│ 更新: 3条                           │
│ 冲突: 2条                           │
├─────────────────────────────────────┤
│ 冲突1：                             │
│ 9903.94.XX在数据库中存在但审核后删除 │
│ [保留] [删除] [查看详情]            │
├─────────────────────────────────────┤
│ 冲突2：                             │
│ 9903.94.YY税率不一致               │
│ 数据库: 25%  审核后: 30%           │
│ [使用数据库值] [使用审核值]         │
├─────────────────────────────────────┤
│ [Commit to Database →]              │
└─────────────────────────────────────┘
```

### 4.5 完成页面

```
页面路径：/review/completed

显示内容：
┌─────────────────────────────────────┐
│ ✓ Note 33 审核完成                  │
├─────────────────────────────────────┤
│ 审核统计：                          │
│ - 解决冲突: 30个                    │
│ - 新增记录: 5条                     │
│ - 更新记录: 3条                     │
│ - 删除记录: 1条                     │
│ - 用时: 45分钟                      │
├─────────────────────────────────────┤
│ 文件输出：                          │
│ - unified.json ✓                    │
│ - audit_log.json ✓                  │
│ - backup已创建 ✓                    │
├─────────────────────────────────────┤
│ [查看审核日志] [继续下一个Note]     │
└─────────────────────────────────────┘
```

## 五、后端服务设计

### 5.1 “下一步”处理流程

当用户点击“下一步”后：

```
伪代码流程：

函数 handleNextStep(session_id):
    获取当前session状态

    如果 状态 == "冲突审核中":
        检查是否所有冲突都已解决
        如果有未解决的:
            返回错误提示
        否则:
            合并数据生成unified.json
            更新状态为"已生成统一文件"
            自动执行数据库比对

    如果 状态 == "已生成统一文件":
        如果 LLM结果一致 且 noteXX_db_compare.json 存在:
            读取文件作为数据库比对结果
        否则:
            直接连接数据库，比对 unified.json 与数据库
        生成比对结果

        如果有差异:
            返回差异列表，进入数据库差异审核页
        否则:
            直接准备入库操作

    如果 状态 == "数据库差异已解决":
        根据用户决策生成SQL操作列表
        执行数据库事务
        返回执行结果
```

### 5.2 数据库比对逻辑

```
数据库比对流程：

1. 构建索引
   unified_index = {measure_key: measure_data}
   db_index = {measure_key: db_data}

2. 找出差异
   新增的 = unified中有但db中没有的
   可能更新的 = 两边都有但内容不同的
   可能删除的 = db中有但unified中没有的

3. 生成比对报告
   {
     "to_insert": [...],    # 确定要插入的
     "to_update": [...],    # 确定要更新的
     "conflicts": [         # 需要人工决策的
       {
         "type": "exists_in_db_not_in_unified",
         "measure": {...},
         "suggestion": "keep|delete"
       }
     ]
   }
```

### 5.3 Session管理

```
Session数据结构：
{
  session_id: "sess_33_timestamp",
  note_id: 33,
  status: "当前状态",
  reviewer: "审核员",
  start_time: "开始时间",

  # 原始数据
  openai_data: {...},
  grok_data: {...},
  compare_data: {...},

  # 审核结果
  resolutions: {
    measures: {...},
    scopes: {...}
  },

  # 生成的文件
  unified_json: {...},
  db_comparison: {...},

  # 数据库决策
  db_decisions: {...},

  # 最终操作
  final_operations: {...}
}

状态流转：
init → conflict_review → unified_generated →
db_comparison → db_conflict_review (可选) →
ready_to_commit → committed → completed
```

## 六、数据结构转换

### 6.1 冲突解决记录

```
llm_compare.json → resolutions.json

输入：差异列表
输出：{
  conflict_id: {
    resolution: "openai|grok|manual",
    value: {...},
    resolved_at: timestamp
  }
}
```

### 6.2 生成统一数据

```
resolutions.json + 原始数据 → unified.json

合并逻辑：
- 以 measure_key 定位（heading|country|start|end|is_potential）
- 冲突展示归并使用 stable_key = heading（同一 heading 不会有多个 country）
- 字段冲突：按人工选择 OpenAI / Grok 取字段值
- only_in_openai / only_in_grok：按“接受/拒绝”决定是否保留 measure
- scope 冲突：
  - 使用 OpenAI → 取 OpenAI 原始 scopes
  - 使用 Grok → 取 Grok 原始 scopes
  - 使用并集 → scopes = OpenAI ∪ Grok（以 key_norm + relation 去重）
    - key_norm 只保留数字，忽略小数点差异
    - 同一 key_norm+relation 元数据冲突：优先 OpenAI
  - 字段选择与 scope 选择相互独立
```

### 6.3 数据库操作生成

```
unified.json + db_data → db_operations.json

输出：{
  inserts: [SQL语句列表],
  updates: [SQL语句列表],
  deletes: [SQL语句列表]
}
```

## 七、差异类型分析

基于实际的`note33_llm_compare.json`，主要差异类型：

1. **措施级别差异**（missing_in_openai/missing_in_grok）
   - 主要是 `is_potential` 和 `effective_start_date` 的差异
   - 同一个heading，不同的属性组合被识别为不同措施

2. **字段差异**（field_diffs）
   - 税率差异
   - 日期差异
   - 国家代码差异

3. **Scope差异**（scope_diffs）
   - HTS代码包含/排除的差异
   - 关系类型(include/exclude)的差异

## 八、错误处理和回滚

### 8.1 错误处理策略

```
1. 冲突解决阶段错误
   - 保存当前进度
   - 允许继续从断点审核

2. 数据库操作错误
   - 自动创建备份表
   - 失败时回滚
   - 记录错误日志
```

### 8.2 回滚机制

```
备份表命名：otherch_measures_backup_[timestamp]
包含：原始数据快照
操作：失败时从备份恢复
```

### 8.3 审核日志

```
审核日志结构：
{
  session_id: "...",
  actions: [
    {
      timestamp: "...",
      action: "resolve_conflict",
      details: {...},
      user: "..."
    }
  ],
  final_result: "success|failed",
  error_detail: null
}
```

## 九、批量操作优化

### 9.1 预设策略

- "信任OpenAI税率"：所有tax_rate冲突选OpenAI
- "Grok优先scope"：所有scope差异选Grok
- "保守日期"：选择较早的日期

### 9.2 键盘快捷键

- `1`: 选择OpenAI
- `2`: 选择Grok
- `3`: 手动输入
- `Enter`: 下一个
- `Shift+Enter`: 上一个
- `Ctrl+Enter`: 下一步

### 9.3 批量应用

用户可以选择"将此决策应用到所有相似冲突"

## 十、统一JSON格式

```json
{
  "note_number": 33,
  "review_session_id": "session_33_1234567890",
  "reviewed_at": "2024-01-22T10:30:00",
  "reviewer": "admin",
  "measures": [
    {
      "heading": "9903.94.40",
      "country_iso2": "JP",
      "ad_valorem_rate": "25",
      "effective_start_date": "2025-09-16",
      "is_potential": true,
      "scopes": [
        {
          "key": "8407.31.00",
          "relation": "include"
        }
      ]
    }
  ],
  "resolution_summary": {
    "total_conflicts": 30,
    "openai_accepted": 10,
    "grok_accepted": 15,
    "manual_override": 5
  }
}
```

## 十一、API端点设计

### 11.1 核心API

- `POST /api/notes/<note_id>/start-review` - 启动审核会话
- `POST /api/sessions/<session_id>/resolve-measure` - 解决措施冲突
- `POST /api/sessions/<session_id>/resolve-scope` - 解决scope差异
- `POST /api/sessions/<session_id>/next-step` - 生成 unified.json 并执行数据库比对
- `POST /api/sessions/<session_id>/generate-unified` - 生成统一JSON（兼容保留）
- `POST /api/sessions/<session_id>/compare-with-db` - 数据库比对（可选）
- `POST /api/sessions/<session_id>/update-db` - 更新数据库

### 11.2 辅助API

- `GET /api/notes/<note_id>/conflicts` - 获取冲突列表
- `GET /api/sessions/<session_id>/status` - 获取会话状态
- `GET /api/sessions/<session_id>/progress` - 获取进度
- `POST /api/sessions/<session_id>/save-progress` - 保存进度

## 十二、技术栈建议

### 12.1 前端
- Framework: React 或 Vue.js
- UI库: Ant Design 或 Element UI
- 状态管理: Redux 或 Vuex
- HTTP客户端: Axios

### 12.2 后端
- Framework: Flask (与现有Python代码集成)
- 数据库: PostgreSQL (已在使用)
- Session存储: Redis 或内存
- 文件存储: 本地文件系统

### 12.3 部署
- Web服务器: Nginx
- 应用服务器: Gunicorn
- 容器化: Docker
- 编排: Docker Compose

## 十三、优势总结

1. **流程清晰**：每一步都有明确的输入输出
2. **用户友好**：简洁的界面，点击式操作
3. **数据安全**：完善的备份回滚机制
4. **可追溯**：详细的审核日志
5. **高效审核**：批量操作和预设策略
6. **灵活扩展**：易于添加新的Note类型

## 十四、实施建议

### 14.1 第一阶段：基础功能
- 实现冲突审核界面
- 生成unified.json
- 基本的session管理

### 14.2 第二阶段：数据库集成
- 数据库比对功能
- 数据入库操作
- 备份和回滚

### 14.3 第三阶段：优化提升
- 批量操作
- 键盘快捷键
- 审核日志和报表

### 14.4 第四阶段：高级功能
- 历史数据分析
- 模型准确率统计
- 审核效率报告
