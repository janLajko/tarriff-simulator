# Section 301 Agent 概览

该脚本负责把 HTS Section 301 的 99 系列条目、注释范围与数据库表结构化关联起来，核心由 DB 访问、LLM 解析和流程编排三部分组成。

## 代码框架

- 常量/正则
  - `DEFAULT_HEADINGS`：默认要处理的 9903.88.xx headings。
  - `LLM_MEASURE_PROMPT` / `LLM_NOTE_PROMPT`：两类 LLM 结构化抽取提示词。
  - `NOTE_TOKEN_RE`：用于识别 note 标签 token。
- 数据结构
  - `MeasureAnalysis`：LLM 输出的 include/exclude + 生效期。
  - `ScopeRecord`：`s301_scope` 表的写入对象。
- 工具函数
  - `parse_date()`：多格式日期解析。
  - `normalize_note_label()`：统一 note 标识为 `note(x)(y)`。
  - `classify_code_type()`：判断 heading/hts8。
- 数据库层 `Section301Database`
  - `fetch_hts_code()`：从 `hts_codes` 拉取 heading 行。
  - `ensure_measure()`：插入或复用 `s301_measures`。
  - `ensure_scope()`：插入或复用 `s301_scope`。
  - `ensure_scope_measure_map()`：插入或复用 `s301_scope_measure_map`。
  - `fetch_note_rows()`：按 note label 取整段注释文本。
  - `fetch_scope_relations()`：读取某 measure 已有 scope 关联。
- LLM 层 `Section301LLM`
  - `_post()`：调用 OpenAI Chat Completions。
  - `extract_measure()`：解析 heading description 的 include/exclude 与生效期。
  - `extract_note()`：解析 note 文本生成 scope/except 列表。
- 业务编排 `Section301Agent`
  - `run()` / `process_heading()`：单 heading 全流程。
  - `_derive_rate()`：从描述中推导税率。
  - `_apply_scope_links()`：处理 include/exclude 引用。
  - `_process_note_references()`：读取 note + LLM 抽取并入库。
  - `_handle_non_note_reference()`：处理非 note 的直接 codes。
  - `_convert_note_entries()`：LLM 输出转 `ScopeRecord`。
  - `_link_child_heading()` / `_link_child_heading_reference()`：处理 99 子 heading。
- CLI
  - `parse_args()` / `main()`：读取 DSN、headings、日志级别并启动。

## 实际流程(按代码执行顺序)

1. CLI 参数
   - 输入: `--dsn`、`--headings`、`--log-level`。
   - 默认 headings: `DEFAULT_HEADINGS`。
2. `agent.run()` 遍历 headings
   - 输入: heading 列表。
   - 输出: 对每个 heading 调用 `process_heading()`。
3. `process_heading()` 前置判断
   - 命中 `_measure_cache` 直接复用。
   - `_in_progress` 防止递归死循环。
   - 若 `s301_measures` 已存在同 heading/country 记录，直接复用并返回。
4. 读取 heading 元数据
   - 输入: `hts_codes.hts_number = heading`。
   - 输出: `description`、`status`、`general_rate_of_duty`。
   - 过滤: `status=expired` 或 `description` 为空则跳过。
5. LLM 解析 heading description
   - 输入: `description`。
   - 输出: `include`/`except` + `effective_period`。
6. 税率与 measure 写入
   - `_derive_rate()` 从描述提取税率(含 "plus" 规则与兜底)。
   - `ensure_measure()` 写入/复用 `s301_measures`。
7. include/exclude 拆分
   - note 引用: `note 20(a)` 这类文本。
   - 非 note 引用: 直接 code(如 `8501.10.40`)。
8. note 引用处理流程
   - `fetch_note_rows()` 从 `hts_notes` 拉取 note 及子节点文本。
   - 多个 note 会合并成一个 LLM 输入块。
   - LLM 输出 `scope`/`except` 列表后，由 `_convert_note_entries()` 转成 `ScopeRecord`。
   - 注意: LLM 返回 `keys` 会被拆分成多个单条 key 后写库。
9. 非 note 引用处理流程
   - 直接写入 `s301_scope` + `s301_scope_measure_map`。
   - 如果引用是 99 开头的 heading，目前代码直接跳过(链接逻辑被注释)。
10. 子 heading 处理
   - 如果 note scope/exclude 中出现 99 heading，会递归 `process_heading()`。
   - include: 继承子 heading 的 scope。
   - exclude: 只记录父->子 heading 的排除关系。
11. 事务与缓存
   - 单 heading 成功后 `commit()`，异常则 `rollback()`。
   - `_note_cache` 复用相同 note 组合的 LLM 结果。

## 示例输入/输出片段(按实际流程)

### 示例 1: heading description -> measure 解析

输入: `hts_codes.description`
```text
Products of China, as provided for in note 20(a) and in the subheadings
enumerated in note 20(b). Except as provided in headings 9903.88.05 and
9903.88.58. The duty provided in the applicable subheading plus 25 percent.
Effective with respect to entries on or after June 15, 2024 through
November 29, 2025.
```

LLM 输出(示例):
```json
{
  "except": ["9903.88.05", "9903.88.58"],
  "include": ["note 20(a)", "note 20(b)"],
  "effective_period": {
    "start_date": "June 15, 2024",
    "end_date": "November 29, 2025"
  }
}
```

进入数据库的核心字段(示例):
```text
s301_measures:
  heading=9903.88.01
  country_iso2=CN
  ad_valorem_rate=25.000
  value_basis=customs_value
  effective_start_date=2024-06-15
  effective_end_date=2025-11-29
```

### 示例 2: note 文本 -> scope/exclude 解析

输入: `hts_notes` 合并后的 note 文本块(每行由 label + content 拼接)
```text
note 20(a) For the purposes of heading 9903.88.01, the products of China,
as provided for in subheadings 8501.10.40, 8501.31.40, and 8517.62.00,
except products provided for in statistical reporting number 8517.62.0090.
Effective January 1, 2023 through December 31, 2023.
```

LLM 输出(示例):
```json
{
  "input_htscode": ["9903.88.01"],
  "scope": [
    {
      "keys": "8501.10.40,8501.31.40,8517.62.00",
      "key_type": "hts8",
      "country_iso2": "CN",
      "source_label": "note20(a)",
      "effective_start_date": "2023-01-01",
      "effective_end_date": "2023-12-31"
    }
  ],
  "except": [
    {
      "key": "8517.62.0090",
      "key_type": "hts10",
      "country_iso2": "CN",
      "source_label": "note20(a)-exclusion",
      "effective_start_date": "2023-01-01",
      "effective_end_date": "2023-12-31"
    }
  ]
}
```

代码落表的实际行为:
```text
s301_scope:
  key=8501.10.40 (hts8, CN, note20(a), 2023-01-01..2023-12-31)
  key=8501.31.40 (hts8, CN, note20(a), 2023-01-01..2023-12-31)
  key=8517.62.00 (hts8, CN, note20(a), 2023-01-01..2023-12-31)
  key=8517.62.0090 (hts10, CN, note20(a)-exclusion, 2023-01-01..2023-12-31)

s301_scope_measure_map:
  scope_id -> measure_id, relation=include, note_label=note20(a)
  scope_id -> measure_id, relation=exclude, note_label=note20(a)-exclusion
```

## 关键行为细节(与图示的对应关系)

- 图示里的 "get metadata from csv" 实际来自 `hts_codes` 表。
- LLM 解析分两次: 先解析 heading description，再解析 note 文本。
- note 输出里的 `keys` 会被拆成多条 `s301_scope` 记录。
- note 里未给出日期时，代码会回退到 heading 的生效期。
- `s301_scope_measure_map` 的有效期目前固定为 `NULL`(函数内部覆盖传入日期)。

## 主要输出

- 数据库写入
  - `s301_measures`：每个 heading 的国家、税率、起止日期等。
  - `s301_scope`：scope/exclude 的 heading/hts8/hts10 条目。
  - `s301_scope_measure_map`：scope 与 measure 的 include/exclude 关系。
- LLM 结构化输出
  - measure 级别 JSON：`include`、`except`、`effective_period`。
  - note 级别 JSON：`input_htscode`、`scope`、`except`。
- 日志输出
  - 处理进度、跳过原因、重复插入与异常信息。
