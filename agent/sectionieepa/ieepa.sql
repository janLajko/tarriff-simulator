CREATE SEQUENCE "public"."sieepa_measures_id_seq"
    AS bigint
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE "public"."sieepa_measures_id_seq"
    OWNER TO "postgres";

CREATE TABLE "public"."sieepa_measures" (
  "id" int8 NOT NULL DEFAULT nextval('sieepa_measures_id_seq'::regclass),
  "heading" text COLLATE "pg_catalog"."default" NOT NULL,
  "country_iso2" text COLLATE "pg_catalog"."default",
  "ad_valorem_rate" numeric(6,3) NOT NULL,
  "value_basis" text COLLATE "pg_catalog"."default" NOT NULL,
  "melt_pour_origin_iso2" text COLLATE "pg_catalog"."default",
  "origin_exclude_iso2" text[] COLLATE "pg_catalog"."default",
  "notes" jsonb,
  "effective_start_date" date NOT NULL,
  "effective_end_date" date,
  "is_potential" bool,
  CONSTRAINT "sieepa_measures_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "sieepa_measures_melt_pour_iso2_len_chk" CHECK (melt_pour_origin_iso2 IS NULL OR length(melt_pour_origin_iso2) = 2)
)
;

ALTER TABLE "public"."sieepa_measures" 
  OWNER TO "postgres";

ALTER SEQUENCE "public"."sieepa_measures_id_seq"
    OWNED BY "public"."sieepa_measures"."id";

CREATE INDEX "idx_sieepa_measures_heading_country_date" ON "public"."sieepa_measures" USING btree (
  "heading" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "country_iso2" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  "effective_end_date" "pg_catalog"."date_ops" ASC NULLS LAST
);

CREATE INDEX "idx_sieepa_measures_melt_pour" ON "public"."sieepa_measures" USING btree (
  "melt_pour_origin_iso2" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

CREATE INDEX "idx_sieepa_measures_origin_excl" ON "public"."sieepa_measures" USING gin (
  "origin_exclude_iso2" COLLATE "pg_catalog"."default" "pg_catalog"."array_ops"
);

CREATE UNIQUE INDEX "idx_sieepa_measures_unique" ON "public"."sieepa_measures" USING btree (
  "heading" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "country_iso2" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  COALESCE(effective_end_date, '9999-12-31'::date) "pg_catalog"."date_ops" ASC NULLS LAST
);

CREATE SEQUENCE "public"."sieepa_scope_id_seq"
    AS bigint
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE "public"."sieepa_scope_id_seq"
    OWNER TO "postgres";

CREATE TABLE "public"."sieepa_scope" (
  "id" int8 NOT NULL DEFAULT nextval('sieepa_scope_id_seq'::regclass),
  "key" text COLLATE "pg_catalog"."default" NOT NULL,
  "key_type" text COLLATE "pg_catalog"."default" NOT NULL,
  "country_iso2" text COLLATE "pg_catalog"."default",
  "source_label" text COLLATE "pg_catalog"."default",
  "effective_start_date" date NOT NULL,
  "effective_end_date" date,
  CONSTRAINT "sieepa_scope_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "sieepa_scope_key_type_check" CHECK (key_type = ANY (ARRAY['hts8'::text, 'hts10'::text, 'heading'::text, 'note'::text]))
)
;

ALTER TABLE "public"."sieepa_scope" 
  OWNER TO "postgres";

ALTER SEQUENCE "public"."sieepa_scope_id_seq"
    OWNED BY "public"."sieepa_scope"."id";

CREATE INDEX "idx_sieepa_scope_dates" ON "public"."sieepa_scope" USING btree (
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  "effective_end_date" "pg_catalog"."date_ops" ASC NULLS LAST
);

CREATE INDEX "idx_sieepa_scope_key" ON "public"."sieepa_scope" USING btree (
  "key" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "key_type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

CREATE UNIQUE INDEX "idx_sieepa_scope_unique" ON "public"."sieepa_scope" USING btree (
  "key" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "key_type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  COALESCE(country_iso2, ''::text) COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  COALESCE(effective_end_date, '9999-12-31'::date) "pg_catalog"."date_ops" ASC NULLS LAST
);

CREATE SEQUENCE "public"."sieepa_scope_measure_map_id_seq"
    AS bigint
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE "public"."sieepa_scope_measure_map_id_seq"
    OWNER TO "postgres";

CREATE TABLE "public"."sieepa_scope_measure_map" (
  "id" int8 NOT NULL DEFAULT nextval('sieepa_scope_measure_map_id_seq'::regclass),
  "scope_id" int8 NOT NULL,
  "measure_id" int8 NOT NULL,
  "relation" text COLLATE "pg_catalog"."default" NOT NULL,
  "note_label" text COLLATE "pg_catalog"."default",
  "text_criteria" text COLLATE "pg_catalog"."default",
  "effective_start_date" date,
  "effective_end_date" date,
  CONSTRAINT "sieepa_scope_measure_map_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "sieepa_map_measure_fk" FOREIGN KEY ("measure_id") REFERENCES "public"."sieepa_measures" ("id") ON DELETE CASCADE ON UPDATE NO ACTION,
  CONSTRAINT "sieepa_map_scope_fk" FOREIGN KEY ("scope_id") REFERENCES "public"."sieepa_scope" ("id") ON DELETE CASCADE ON UPDATE NO ACTION,
  CONSTRAINT "sieepa_scope_measure_map_relation_check" CHECK (relation = ANY (ARRAY['include'::text, 'exclude'::text]))
)
;

ALTER TABLE "public"."sieepa_scope_measure_map" 
  OWNER TO "postgres";

ALTER SEQUENCE "public"."sieepa_scope_measure_map_id_seq"
    OWNED BY "public"."sieepa_scope_measure_map"."id";

CREATE INDEX "idx_sieepa_map_dates" ON "public"."sieepa_scope_measure_map" USING btree (
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  "effective_end_date" "pg_catalog"."date_ops" ASC NULLS LAST
);

CREATE INDEX "idx_sieepa_map_measure" ON "public"."sieepa_scope_measure_map" USING btree (
  "measure_id" "pg_catalog"."int8_ops" ASC NULLS LAST,
  "relation" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

CREATE INDEX "idx_sieepa_map_scope" ON "public"."sieepa_scope_measure_map" USING btree (
  "scope_id" "pg_catalog"."int8_ops" ASC NULLS LAST
);

CREATE UNIQUE INDEX "idx_sieepa_map_unique" ON "public"."sieepa_scope_measure_map" USING btree (
  "scope_id" "pg_catalog"."int8_ops" ASC NULLS LAST,
  "measure_id" "pg_catalog"."int8_ops" ASC NULLS LAST,
  "relation" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "effective_start_date" "pg_catalog"."date_ops" ASC NULLS LAST,
  COALESCE(effective_end_date, '9999-12-31'::date) "pg_catalog"."date_ops" ASC NULLS LAST
);
