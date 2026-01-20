现在要利用openai filesearch功能实现以下需求
- 根据存到数据库中的数据，做检索，获取以下模块的hts关系
 - 301模块 9903.88.69,9903.88.01,9903.88.02,9903.88.70, 9903.88.03,9903.88.04,9903.88.15
 - 232模块 9903.81.87,9903.81.89,9903.81.90,9903.81.91,9903.81.92,9903.81.93,9903.81.94,9903.81.95,9903.81.96,9903.81.97,9903.81.98,9903.81.99,9903.85.02,9903.85.04,9903.85.07,9903.85.08,9903.85.09,9903.85.12,9903.85.13,9903.85.14,9903.85.15,9903.85.67,9903.85.69,9903.85.68,9903.85.70
 - ieepa模块 9903.02.02,9903.01.32,9903.02.78,9903.01.30,9903.01.31,9903.01.33,9903.01.34,9903.02.01,9903.01.10,9903.01.11,9903.01.12,9903.01.13,9903.01.14,9903.01.15,9903.01.16,9903.01.24,9903.01.21,9903.01.22,9903.01.23,9903.01.26,9903.01.27,9903.01.28,9903.01.29,9903.02.03,9903.02.04,9903.02.05,9903.02.06,9903.02.07,9903.02.08,9903.02.09,9903.02.10,9903.02.11,9903.02.12,9903.02.13,9903.02.14,9903.02.15,9903.02.16,9903.02.17,9903.02.18,9903.02.72,9903.02.73,9903.02.79,9903.02.80,9903.02.82,9903.02.84,9903.02.85,9903.02.86,9903.02.83,9903.02.87,9903.02.89,9903.02.90,9903.02.91,9903.02.88,9903.02.81,9903.96.02,9903.02.74,9903.02.75,9903.02.76,9903.02.77,9903.02.21,9903.02.22,9903.02.23,9903.02.24,9903.02.25,9903.02.26,9903.02.27,9903.02.28,9903.02.29,9903.02.31,9903.02.32,9903.02.33,9903.02.34,9903.02.35,9903.02.37,9903.02.38,9903.02.39,9903.02.40,9903.02.41,9903.02.42,9903.02.43,9903.02.44,9903.02.45,9903.02.46,9903.02.47,9903.02.48,9903.02.49,9903.02.50,9903.02.51,9903.02.52,9903.02.53,9903.02.54,9903.02.55,9903.02.57,9903.02.59,9903.02.60,9903.01.77,9903.01.81,9903.01.82,9903.01.90,9903.01.78,9903.01.79,9903.01.80,9903.01.83,9903.01.84,9903.01.86,9903.01.85,9903.96.01,9903.01.25,9903.02.19,9903.02.20,9903.02.61,9903.02.62,9903.02.63,9903.02.64,9903.02.65,9903.02.66,9903.02.67,9903.02.68,9903.02.69,9903.02.70,9903.02.71,9903.01.01,9903.01.02,9903.01.03,9903.01.04,9903.01.05,9903.01.87,9903.01.88,9903.01.89
 - 其他模块：9903.94.01,9903.94.02,9903.94.03,9903.94.04,9903.94.05,9903.94.06,9903.94.07,9903.94.31,9903.94.32,9903.94.33,9903.94.40,9903.94.41,9903.94.42,9903.94.43,9903.94.44,9903.94.45,9903.94.50,9903.94.51,9903.94.52,9903.94.53,9903.94.54,9903.94.55,9903.94.60,9903.94.61,9903.94.62,9903.94.63,9903.94.64,9903.94.65,9903.78.01,9903.78.02,9903.76.01,9903.76.02,9903.76.03,9903.76.04,9903.76.20,9903.76.21,9903.76.22,9903.76.23,9903.74.01,9903.74.02,9903.74.03,9903.74.05,9903.74.06,9903.74.07,9903.74.08,9903.74.09,9903.74.10,9903.74.11

这是入库的表字段（postgreSQL）
table measures
"id" int8 NOT NULL DEFAULT nextval('sieepa_measures_id_seq'::regclass),
"heading" text COLLATE "pg_catalog"."default" NOT NULL,
"country_iso2" text COLLATE "pg_catalog"."default",
"ad_valorem_rate" numeric(6,3) NOT NULL,
"melt_pour_origin_iso2" text COLLATE "pg_catalog"."default",
"origin_exclude_iso2" text[] COLLATE "pg_catalog"."default",
"notes" jsonb,
"effective_start_date" date NOT NULL,
"effective_end_date" date,
"is_potential" bool,
"date_of_loading" date,
"entry_date" date,

关于字段，给出一些解释和例子
heading 就是99章节关税例子，例如：9903.88.69
country_iso2 生效国家，例如：
- 这一段9903.81.94 and 9903.81.95的 country_iso2就是 UK
Except as provided in heading 9903.96.02, 9903.02.76, and 9903.02.81, headings 9903.81.87 and 9903.81.88 provide the ordinary customs duty treatment of iron or steel products, as enumerated in subdivision (j) of this note, of all countries, other than iron or steel products of the United Kingdom, as provided in subdivisions (p) and (q) of this note and headings 9903.81.94 and 9903.81.95 of this chapter
- 这一段表名9903.88.69的country_iso2是 CN
9903.88.69 Effective with respect to entries on or after June 15, 2024 and through November 9, 2026, articles the product of China, as provided for in U.S. note 20(vvv) to this subchapter, each covered by an exclusion granted by the U.S. Trade Representative
- ad_valorem_rate 这个表达的是当前heading的税率，The duty provided in the applicable subheading + 15%就是 15%，The duty provided in the applicable subheading就是0
- melt_pour_origin_iso2是section 232特有的，表达的是钢铁的熔铸地，例如 9903.81.92	  1/	Derivative iron or steel products provided for in the tariff subheadings enumerated in subdivision subdivisions (m), (n), (t) or (u) of note 16 to this subchapter, where the derivative iron or steel product was processed in another country from steel articles that were melted and poured in the United States. 9903.81.92的melt_pour_origin_iso2就是 US
- origin_exclude_iso2 表达的是排除国家，例如 232模块针对英国有特定的关税项目，那么对于使用其他国家的关税例如 9903.81.92的origin_exclude_iso2应该包含UK
- effective_start_date,effective_end_date 表达的是heading的生效和过期时间
- date_of_loading是装货时间，例如 9903.02.02 Except for goods loaded onto a vessel at the port of loading and in transit on the final mode of transit before 12:01 a.m. eastern daylight time on August 7, 2025, and entered for consumption or withdrawn from warehouse for consumption before 12:01 a.m. eastern daylight time on October 5, 2025, except for products described in headings 9903.01.30-9903.01.33 and 9903.02.78, and except as provided for in headings 9903.01.34 and 9903.02.01, articles the product of Afghanistan, as provided for in subdivision (v) of U.S. note 2 to this subchapter，9903.02.02的 date_of_loading是2025-08-07，entry_date是2025-10-05
- is_potential是满足某一些情况下可能会有效
例子1：Products of iron or steel of the United Kingdom provided for in the tariff headings or subheadings enumerated in subdivision (q) of note 16 to this subchapter, admitted to a U.S. foreign trade zone under “privileged foreign status” as defined by 19 CFR 146.41, prior to 12:01 a.m. eastern daylight time on June 4, 2025 （依据：privileged foreign status）
例子2：Articles that are entered free of duty under the terms of general note 11 to the HTSUS, including any treatment set forth in subchapter XXIII of chapter 98 and subchapter XXII of chapter 99 of the HTS, as related to the USMCA（依据：USMCA）
例子3：Heading 9903.74.05 applies to entries of articles that are classifiable under provisions of the HTSUS enumerated in
subdivision (b) of this note but that are not medium- and heavy-duty vehicles（依据： 模糊概念 that are not medium- and heavy-duty vehicles）
例子4：(b) The rates of duty set forth in headings 9903.94.01, 9903.94.02, 9903.94.03, 9903.94.04, 9903.94.40, 9903.94.41,
9903.94.50, 9903.94.51, 9903.94.60, 9903.94.61, and certain entries under 9903.94.31, apply to all imported products
classifiable in the provisions of the HTSUS enumerated in this subdivision:
8703.24.018703.23.018703.22.01
8703.33.018703.32.018703.31.01
8703.60.008703.50.008703.40.00
8703.90.018703.80.008703.70.00
8704.41.008704.31.018704.21.01
8704.60.008704.51.00
(c) Heading 9903.94.02 applies to:
(i) all entries of articles classifiable under provisions of the HTSUS enumerated in subdivision (b) of this note, but that
are not passenger vehicles (sedans, sport utility vehicles, crossover utility vehicles, minivans, and cargo vans) and
light trucks; as well as
(ii) the U.S. content of passenger vehicles and light trucks described in subdivision (d) of this note, upon approval from
the Secretary of Commerce（依据： 9903.94.01, 9903.94.02, 9903.94.03, 9903.94.04, 9903.94.40, 9903.94.41,
9903.94.50, 9903.94.51, 9903.94.60, 9903.94.61, and certain entries under 9903.94.31都有相同的scope，但是条件不一样，例如9903.94.02只有在特定的条件下，才有效）


table scope
  "id" int8 NOT NULL DEFAULT nextval('otherch_scope_id_seq'::regclass),
  "key" text COLLATE "pg_catalog"."default" NOT NULL,
  "key_type" text COLLATE "pg_catalog"."default" NOT NULL,
  "country_iso2" text COLLATE "pg_catalog"."default",
  "source_label" text COLLATE "pg_catalog"."default",
  "effective_start_date" date NOT NULL,
  "effective_end_date" date,


table 
"id" int8 NOT NULL DEFAULT nextval('otherch_scope_measure_map_id_seq'::regclass),
  "scope_id" int8 NOT NULL,
  "measure_id" int8 NOT NULL,
  "relation" text COLLATE "pg_catalog"."default" NOT NULL,
  "note_label" text COLLATE "pg_catalog"."default",
  "text_criteria" text COLLATE "pg_catalog"."default",
  "effective_start_date" date,
  "effective_end_date" date,

输出
{{
  "input_htscode": ["9903.88.01"],
  "scope": [
    {{ "keys": "0203.29.20,0203.29.40,0206.10.00,0208.10.00,0208.90.20,0208.90.25,0210.19.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }},
    {{ "key": "2845.30.00", "key_type": "hts8", "country_iso2": "CN", "source_label": "note20(b)", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ],
  "except": [
    {{ "key": "9903.88.05", "key_type": "heading", "country_iso2": "CN", "source_label": "note20(a)-exclusion", "effective_start_date": "1900-01-01", "effective_end_date": null }}
  ]
}}

scope表示哪些会缴纳input_htscode的hts code
except表示哪些heading出现就不在缴纳input_htscode

