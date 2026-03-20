# OtherChapter 双模型抽取与比对设计

本文档描述在 `othercharpter.py` 的现有流程中，引入 Gemini 与 OpenAI 同步抽取，并进行模型比对与数据库比对的设计方案。**不修改既有表结构与入库逻辑**，仅新增比对与输出。

## 1. 目标

- 同一份 note 文本与上下文，分别由 OpenAI 与 Gemini 抽取结构化 `measures`。
- 对两模型结果进行一致性比对：
  - 一致：继续与数据库数据比对。
  - 不一致：输出差异内容，不做 DB 比对。
- 输出文件固定到：`agent/othercharpter-agent/output`

## 2. 范围与约束

- 仅比对 `measures`，**不包含** `extra_measures`。
- 不比对 `value_basis` 与 `notes`。
- 只对以下字段做一致性判断：
  - `heading`
  - `country_iso2`
  - `ad_valorem_rate`（允许数值精度差异）
  - `is_potential`
  - `effective_start_date`
  - `effective_end_date`
  - `scopes`
- `scopes` 只比较 **`key + relation`**，忽略顺序；忽略小数点格式差异。

## 3. 模型与调用

- OpenAI：现有 `OtherChapterLLM` 不变。
- Gemini：新增 Gemini LLM 封装类，模型固定：
  - `gemini-3-flash-preview`
- 两模型使用**同一 LLM_PROMPT** 与 **同一解析器** `_parse_json_payload`。

## 4. 输出文件

所有输出均写入：`agent/othercharpter-agent/output`

- `note{n}_openai.json`：OpenAI 原始抽取结果
- `note{n}_gemini.json`：Gemini 原始抽取结果
- `note{n}_llm_compare.json`：两模型比对结果
- `note{n}_db_compare.json`：DB 比对结果（仅当两模型一致时生成）

## 5. 标准化与比对规则

### 5.1 Measure 规范化

用于对齐并比对两模型输出的 measure 记录。

- `heading`：去空白与尾随标点
- `country_iso2`：大写；空值统一为 `null`
- `effective_*`：解析为 ISO 日期；空值 `null`
- `ad_valorem_rate`：转 Decimal 数值比较（`25` == `25.000`）
- `is_potential`：转 bool；空值 `null`

### 5.2 Measure 对齐主键

```
measure_key = heading + "|" + country_iso2 + "|" +
              effective_start_date + "|" + effective_end_date + "|" +
              is_potential
```

### 5.3 Scope 规范化与比较

`scopes` 只比较 `key + relation`，忽略顺序。

- 展开 `keys` → 多条 `key`
- `relation` 为空时默认 `include`
- `key` 规范化：
  - 保留原值用于输出
  - 同时生成 `key_norm`：仅保留数字（忽略小数点等格式差异）
    - 例如：`9903.94.01` 与 `99039401` 视为一致
- 实际比对使用 `(key_norm, relation)` 的集合

### 5.4 不一致输出（scope diffs）

diff 中保留原始 `key`，并附带规范化值便于定位：

```
{
  "key_raw": "9903.94.01",
  "key_norm": "99039401",
  "relation": "include"
}
```

## 6. 模型比对输出结构

`note{n}_llm_compare.json` 建议结构：

```
{
  "consistent": true|false,
  "summary": {
    "openai_count": 0,
    "gemini_count": 0,
    "matched_count": 0
  },
  "missing_in_openai": [measure_key...],
  "missing_in_gemini": [measure_key...],
  "field_diffs": {
    "<measure_key>": {
      "country_iso2": {"openai": "...", "gemini": "..."},
      "ad_valorem_rate": {"openai": "...", "gemini": "..."},
      "is_potential": {"openai": "...", "gemini": "..."},
      "effective_start_date": {"openai": "...", "gemini": "..."},
      "effective_end_date": {"openai": "...", "gemini": "..."}
    }
  },
  "scope_diffs": {
    "<measure_key>": {
      "only_in_openai": [{key_raw,key_norm,relation}...],
      "only_in_gemini": [{key_raw,key_norm,relation}...]
    }
  }
}
```

## 7. DB 比对设计

仅在两模型一致时执行 DB 比对。

### 7.1 DB 过滤条件

- `notes->>'note_number' = note_number`
- 使用现有表（不变）：
  - `otherch_measures`
  - `otherch_scope`
  - `otherch_scope_measure_map`

### 7.2 DB 还原结构

将 DB 数据还原为与 LLM 输出一致的结构，仅保留本设计要求的字段与 scopes。

### 7.3 DB 对比输出

`note{n}_db_compare.json` 结构与 LLM 比对一致：

- `consistent`
- `summary`
- `missing_in_db`
- `extra_in_db`
- `field_diffs`
- `scope_diffs`

## 8. 处理流程（单 note）

1. 读取 note 文本与 context
2. OpenAI 抽取 → 保存 `note{n}_openai.json`
3. Gemini 抽取 → 保存 `note{n}_gemini.json`
4. 标准化后做 LLM 比对 → 保存 `note{n}_llm_compare.json`
5. 若一致：
   - 从 DB 还原结构 → 比对 → 保存 `note{n}_db_compare.json`
6. 若不一致：
   - 只输出差异，不触发 DB 比对

## 9. 日志与异常

- 任何模型输出解析失败：记录错误并终止本 note 的比对流程
- 输出文件写入失败：记录错误并终止本 note 的后续步骤

## 10. 可扩展项（可选）

- 在比对输出中增加 `normalized_measures` 便于调试
- 允许配置是否包含 `notes` / `value_basis` 的比对开关
