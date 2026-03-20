# Other Chapter Agent (notes 33/36/37/38) 概览

该脚本处理 Chapter 99 的 notes 33/36/37/38。它从本地 note 文本文件读取内容、从 `hts_codes` 拉取 Chapter 99 headings 描述与税率，调用 LLM 结构化抽取 measures + scopes，最后写入 `otherch_*` 三张表并输出额外 measure 记录到 JSON 文件。

## 代码框架

- 常量/路径
  - `NOTE_LABELS` / `NOTE_HEADINGS`：每个 note 对应的 Chapter 99 headings 列表。
  - `NOTE_PDF_DIR` / `NOTE_PDF_FILES`：note 文本来源文件。
  - `NOTE38_SUBVISIONI_PATH`：note 38 subdivision (i) 的扩展文本。
  - `EXTRA_MEASURES_DIR`：LLM 解析出的额外 heading 输出目录。
- 正则/解析工具
  - `HTS_CODE_RE` / `RANGE_SHORT_RE` / `RANGE_FULL_RE`：抽取与展开 HTS code 范围。
  - `_derive_rate()`：从 `general_rate_of_duty` 解析税率。
  - `_parse_date()` / `_parse_rate()` / `_normalize_iso()`：类型与格式归一化。
- 数据结构
  - `MeasureRecord`：measure 写入对象。
  - `ScopeRecord`：scope 写入对象。
  - `ScopeMapEntry`：scope↔measure 关联对象。
- LLM 层 `OtherChapterLLM`
  - `_chat()`：OpenAI Chat Completions。
  - `extract()`：拼装上下文 + note 文本 → JSON。
- DB 层 `OtherChapterDB`
  - `fetch_hts_descriptions()`：批量拉取 `hts_codes` 描述与税率。
  - `insert_measures()` / `insert_scopes()` / `insert_scope_measure_map()`：批量写表。
- 业务编排 `OtherChapterProcessor`
  - `process_note()`：单 note 处理主流程。
  - `_split_measures_by_heading()`：将 LLM 输出分为“主 headings”和“额外 headings”。
  - `_persist()`：将 measures/scopes 落库。

## 实际流程(按代码执行顺序)

1. CLI 参数
   - 输入: `--dsn`, `--note`, `--table-prefix`, `--model`, `--base-url`, `--api-key`。
   - 默认 `--note=all`，依次处理 33/36/37/38。
2. 读取 note 文本
   - `NOTE_PDF_FILES` → 读取 `pdf/note33.txt` 等。
   - note 文本是本地文件，不是 `hts_notes` 表。
3. 收集 Chapter 99 headings 元数据
   - 从 `NOTE_HEADINGS[note_number]` 得到 headings。
   - `fetch_hts_descriptions()` 从 `hts_codes` 获取 description/general_rate_of_duty。
4. 组织 LLM context
   - `note_number`, `note_label`, `chapter99_headings`。
   - `hts_codes[]`: code + description + general_rate_of_duty + ad_valorem_rate(预解析)。
   - 加载 `note38subvisioni.txt` 作为 subdivision (i) 展开来源。
5. LLM 解析
   - 输入: note 文本 + context + subdivision (i) 文本。
   - 输出: `{"measures": [...]}`，每个 measure 含 scopes。
6. 分离“额外 headings”
   - 若 LLM 输出的 heading 不在 `NOTE_HEADINGS` 列表，归为 extra。
   - extra measures 写入 `output/noteXX_extra_measures.json`。
7. 落库
   - `MeasureRecord`: 
     - `effective_start_date` 缺失时回退到 `1900-01-01`。
     - `ad_valorem_rate` 缺失时默认为 `0`。
     - `value_basis` 默认 `total_value`。
   - `ScopeRecord`: 通过 `scopes` 展开 `key`/`keys`，标准化 `key_type`。
   - `ScopeMapEntry`: 将 relation/note_label/text_criteria 写入 map 表。

## 示例输入/输出片段(按实际流程)

### 示例 1: note 文本 → LLM 输出

输入: `pdf/note36.txt` 中的片段(示意)
```text
U.S. note 36. For the purposes of heading 9903.78.01, products of the United Kingdom,
as provided for in subheadings 7208.25.10 and 7208.25.20, are subject to additional duties.
Except for goods classified in 9903.78.02. Effective with respect to entries on or after 2024-01-01.
```

LLM 输出(示例)
```json
{
  "measures": [
    {
      "heading": "9903.78.01",
      "country_iso2": "UK",
      "ad_valorem_rate": 25,
      "value_basis": "customs_value",
      "is_potential": false,
      "notes": {
        "note_number": 36,
        "entry_date_on_or_after": "2024-01-01"
      },
      "effective_start_date": "2024-01-01",
      "effective_end_date": null,
      "scopes": [
        {
          "keys": "7208.25.10,7208.25.20",
          "key_type": "hts8",
          "relation": "include",
          "country_iso2": "UK",
          "source_label": "note(36)",
          "note_label": "note(36)",
          "text_criteria": null,
          "effective_start_date": "2024-01-01",
          "effective_end_date": null
        },
        {
          "key": "9903.78.02",
          "key_type": "heading",
          "relation": "exclude",
          "country_iso2": "UK",
          "source_label": "note(36)-exclusion",
          "note_label": "note(36)-exclusion",
          "text_criteria": null,
          "effective_start_date": "2024-01-01",
          "effective_end_date": null
        }
      ]
    }
  ]
}
```

### 示例 2: 落库后的数据形态

写入 `otherch_measures` (示意)
```text
heading=9903.78.01
country_iso2=UK
ad_valorem_rate=25.000
value_basis=customs_value
effective_start_date=2024-01-01
effective_end_date=NULL
```

写入 `otherch_scope` (示意)
```text
key=7208.25.10 key_type=hts8 country_iso2=UK source_label=note(36)
key=7208.25.20 key_type=hts8 country_iso2=UK source_label=note(36)
key=9903.78.02 key_type=heading country_iso2=UK source_label=note(36)-exclusion
```

写入 `otherch_scope_measure_map` (示意)
```text
scope_id -> measure_id relation=include note_label=note(36)
scope_id -> measure_id relation=exclude note_label=note(36)-exclusion
```

### 示例 3: 额外 heading 输出

当 LLM 解析到 note 文本中出现了不在 `NOTE_HEADINGS` 的 Chapter 99 headings：
```text
output/note36_extra_measures.json
```
```json
{
  "note_number": 36,
  "note_label": "note(36)",
  "measures": [
    {"heading": "9903.90.01", "country_iso2": null, "scopes": []}
  ]
}
```

## 关键行为细节

- note 文本来源于本地 `pdf/*.txt`，不是 `hts_notes` 数据库。
- LLM 只允许输出 note 文本和 context 中出现的 HTS codes；Chapter 99 headings 必须全部输出。
- `scope` 中 `keys` 会被拆成多条 `otherch_scope` 记录。
- `key_type` 由 `_normalize_key_type()` 兜底，避免 LLM 输出 `hts`。
- `effective_start_date` 缺失时统一回退为 `1900-01-01`。
- note 38 subdivision (i) 被引用时，会扩展成多条 HTS codes 并写入 scopes。

