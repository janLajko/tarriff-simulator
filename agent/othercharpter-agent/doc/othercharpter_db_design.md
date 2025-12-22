# Other Chapter Notes (33/36/37/38) – Database Design

本設計**只使用三張表**，結構完全參考 `sieepa_measures`、`sieepa_scope`、`sieepa_scope_measure_map`。不新增關係表，也不設 `alt_heading`。替代情境統一用 `is_potential = true` 表示。

> 參考來源：`agent/sectionieepa/ieepa.sql`

---

## 設計目標

- 與 `sieepa` 結構/索引/唯一性規則一致，降低後續維護成本。
- 每個 note 一次性產出多個 heading，scope 仍透過 map 連到 measure。
- 替代情境不建表、不新增欄位，僅用 `is_potential` 表達。
- 支援同一 heading 在不同國別或期間的差異。

---

## 一、Measure 表（對應 sieepa_measures）

**表名建議：`otherch_measures`（欄位與 `sieepa_measures` 對齊）**

用途：保存每個 Chapter 99 heading 的主要關稅條款與屬性。

**字段（與 sieepa 一致）**

- `id` (bigserial, PK)
- `heading` (text, NOT NULL)     // 9903.xx.xx
- `country_iso2` (text, NULL)    // 適用國家；null 表示通用
- `ad_valorem_rate` (numeric(6,3), NOT NULL)  // 稅率（如 +25%，豁免可存 0）
- `value_basis` (text, NOT NULL) // total_value / us_content / non_us_content / copper_content / non_copper_content
- `melt_pour_origin_iso2` (text, NULL)        // 保留對齊欄位，其他章節通常為 null
- `origin_exclude_iso2` (text[], NULL)        // 排除國家
- `notes` (jsonb, NULL)          // 結構化附註（如 in_lieu_of、FTA/AD/CVD 堆疊、note_number 等）
- `effective_start_date` (date, NOT NULL)
- `effective_end_date` (date, NULL)
- `is_potential` (bool, NULL)    // 替代情境以 true 表示

**索引與唯一性（與 sieepa 對齊）**

- `idx_otherch_measures_heading_country_date`
  - (heading, country_iso2, effective_start_date, effective_end_date)
- `idx_otherch_measures_origin_excl` (gin)
  - (origin_exclude_iso2)
- `idx_otherch_measures_unique`
  - (heading, country_iso2, effective_start_date, COALESCE(effective_end_date, '9999-12-31'))

---

## 二、Scope 表（對應 sieepa_scope）

**表名建議：`otherch_scope`（欄位與 `sieepa_scope` 對齊）**

用途：保存 note 中提到的 HTS 範圍（heading/hts8/hts10/note）。

**字段（與 sieepa 一致）**

- `id` (bigserial, PK)
- `key` (text, NOT NULL)      // 例如 9403.40.9060 或 9903.94.01
- `key_type` (text, NOT NULL) // heading / hts8 / hts10 / note
- `country_iso2` (text, NULL)
- `source_label` (text, NULL) // 例如 note33(k)
- `effective_start_date` (date, NOT NULL)
- `effective_end_date` (date, NULL)

**索引與唯一性（與 sieepa 對齊）**

- `idx_otherch_scope_key`
  - (key, key_type)
- `idx_otherch_scope_dates`
  - (effective_start_date, effective_end_date)
- `idx_otherch_scope_unique`
  - (key, key_type, COALESCE(country_iso2, ''), effective_start_date, COALESCE(effective_end_date, '9999-12-31'))

---

## 三、Scope ↔ Measure Map（對應 sieepa_scope_measure_map）

**表名建議：`otherch_scope_measure_map`（欄位與 `sieepa_scope_measure_map` 對齊）**

用途：將 scope 連到 measure（依 `measure_id`），維持與 IEEPA 相同的查詢路徑。

**字段（與 sieepa 一致）**

- `id` (bigserial, PK)
- `scope_id` (bigint, NOT NULL)   // FK → otherch_scope.id
- `measure_id` (bigint, NOT NULL) // FK → otherch_measures.id
- `relation` (text, NOT NULL)     // include / exclude
- `note_label` (text, NULL)
- `text_criteria` (text, NULL)
- `effective_start_date` (date, NULL)
- `effective_end_date` (date, NULL)

**索引與唯一性（與 sieepa 對齊）**

- `idx_otherch_map_measure`
  - (measure_id, relation)
- `idx_otherch_map_scope`
  - (scope_id)
- `idx_otherch_map_dates`
  - (effective_start_date, effective_end_date)
- `idx_otherch_map_unique`
  - (scope_id, measure_id, relation, effective_start_date, COALESCE(effective_end_date, '9999-12-31'))

---

## 四、落庫與查詢關係摘要

- **Heading → Measures**：`otherch_measures.heading`
- **Measure → Scope**：`otherch_scope_measure_map`（透過 measure_id）
- **Scope → 明細**：`otherch_scope.id`

---

## 五、注意事項

- 同一 heading 在不同國別/期間可能多筆，唯一性需包含 `country_iso2 + effective_date`。
- Note36 的配對關係、Note37 的 in_lieu_of 規則，請放入 `notes` JSONB（不新增欄位）。
- EU 國家建議在 `country_iso2` 用 `EU`，並保留 `notes` 的成員國列表。
- **處理 Chapter 99 heading 互相排除的情境**（例：`9903.00.01 exclude 9903.00.02`）：
  - 在 `otherch_scope` 新增 `key=9903.00.02`、`key_type=heading`
  - 在 `otherch_scope_measure_map` 用 `measure_id=9903.00.01` 連到該 scope 並標記 `relation=exclude`
  - 不需要 9903.00.02 的 measure_id

---

## 六、統計流程設計（每個 note 一次性處理）

- **資料讀取**
  - 從 `hts_notes` 取得 `note(33)`/`note(36)`/`note(37)`/`note(38)` 的完整子樹內容。
  - 從 `hts_codes` 取得：
    - Chapter 99 headings 的 description（如 9903.94.xx、9903.78.xx、9903.76.xx、9903.74.xx）。
    - note 中 subdivision 列出的 HTS codes 之 description。
- **單 note 單 prompt**
  - 每個 note 一次性統計所有 headings 與 scope。
  - 輸出資料包含：measure（heading 條目）、scope（覆蓋 HTS 範圍）、map（include/exclude）。
- **落庫流程**
  - 先 upsert `otherch_measures`（依 heading + country + 生效期）。
  - 再 upsert `otherch_scope`（依 key + key_type + country + 生效期）。
  - 最後 upsert `otherch_scope_measure_map`（依 scope_id + measure_id + relation + 生效期）。

---

## 七、LLM Prompt 產出規範（不輸出程式碼）

- **輸入內容**
  - note 文本（完整子樹，包含 subdivisions）。
  - Chapter 99 headings 清單與 description。
  - subdivision 內 HTS codes 清單與 description。
- **輸出內容（結構化資料）**
  - **Measures**：每個 Chapter 99 heading 一筆。
    - heading
    - country_iso2（若為國別條件）
    - ad_valorem_rate（無稅或豁免存 0）
    - value_basis（例如 us_content / non_us_content / copper_content / non_copper_content）
    - is_potential（替代條款為 true）
    - notes（存放 in_lieu_of、堆疊規則、配對規則等）
  - **Scopes**：每個引用的 HTS 範圍一筆（含 heading 類型）。
    - key / key_type / country_iso2 / source_label / 生效期
  - **Maps**：measure_id 與 scope 的 include/exclude 關係。
    - relation = include / exclude
    - note_label / text_criteria（原文條件摘要）

---

## 八、關鍵抽取規則

- **替代條款**：不記錄 alt_heading，僅將該 heading 的 `is_potential=true`，並在 `notes` 保留條件摘要。
- **配對條款**：不新增欄位，配對信息寫入 `notes`（例如 pair_group=US_CONTENT / NON_US_CONTENT）。
- **in_lieu_of**：寫入 `notes`，避免誤判為疊加。
- **FTA/AD/CVD 堆疊**：寫入 `notes`，不影響 scope 邊界。
- **Chapter 99 heading 作為排除項**：允許出現在 scope（key_type=heading），即使未在 measures 中落庫。

---

## 九、每個 note 的落庫重點

- **Note33（車輛與零件）**
  - 多國家條件（UK/JP/EU/KR）。
  - USMCA 與認證用途條件 → `is_potential=true` 並寫入 `notes`。
  - 25 年老爺車條款 → `is_potential=true`。
- **Note36（銅產品）**
  - 9903.78.01/9903.78.02 配對拆分 → `notes` 記錄 pair_group。
  - scope 來自 subdivision (b) 的 HTS codes。
- **Note37（木製品）**
  - 國家條件（UK/JP/EU/KR）與 in_lieu_of 規則 → `notes`。
  - 非成品豁免（9903.76.04） → `ad_valorem_rate=0`。
- **Note38（中重型車輛）**
  - USMCA 拆分（9903.74.03/9903.74.06）→ `notes` + `is_potential`。
  - 25 年老爺車 → `is_potential=true`。
  - 排除 72/73/76 章 → 作為 scope exclude。
