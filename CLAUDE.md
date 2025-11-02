# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a tariff simulation system that imports and manages HTS (Harmonized Tariff Schedule) data from the USITC API into PostgreSQL. The system consists of specialized agents that fetch, parse, and normalize tariff data for analysis.

## Database Configuration

All agents require a PostgreSQL connection string via the `--dsn` flag or `DATABASE_DSN` environment variable:

```
postgresql://user:password@host:port/database
```

Example from script.txt:
```
postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate
```

## Agent Architecture

The codebase is organized around autonomous data import agents in the `agent/` directory. Each agent is responsible for a specific data domain:

### 1. Basic HTS Agent (`agent/basic-hts-agent/`)

**Purpose**: Imports HTS code data from CSV files into the `hts_codes` table.

**Key Features**:
- Normalizes HTS numbers (pads leading zeros to 4 digits)
- Maintains hierarchical relationships via indent levels
- Inherits duty rates from parent rows when missing
- Fetches real-time status from USITC API endpoints
- Handles both standard HTS codes and Chapter 99 codes

**Database Schema**:
- Table: `hts_codes`
- Key columns: `hts_number`, `indent`, `description`, `general_rate_of_duty`, `column_2_rate_of_duty`, `status`, `parent_hts_number`, `row_order`, `parent_row_order`

**Running**:
```bash
# Full import (truncates existing data)
python agent/basic-hts-agent/basic_hts_agent.py \
  --dsn postgresql://... \
  --data-dir agent/basic-hts-agent/

# Append mode (keeps existing data)
python agent/basic-hts-agent/basic_hts_agent.py \
  --dsn postgresql://... \
  --data-dir agent/basic-hts-agent/ \
  --keep-existing
```

**Data Processing Pipeline**:
1. Reads all CSV files in data directory
2. Normalizes HTS numbers and inherits duty rates
3. Fetches status from USITC API (with chapter-based caching)
4. Maintains hierarchy via `row_order` and `parent_row_order`
5. Bulk inserts with parent ID resolution

### 2. Chapter 99 Notes Agent (`agent/chapter99note-agent/`)

**Purpose**: Scrapes and parses Chapter 99 notes from the USITC API into the `hts_notes` table.

**Key Features**:
- Fetches from `https://hts.usitc.gov/reststop/getChapterNotes?doc=99`
- Parses complex nested HTML structure (SUBCHAPTER → note N → (a)/(1)/(A)/(i) etc.)
- Stores hierarchical path as PostgreSQL array for efficient querying
- Handles arbitrary nesting depth

**Database Schema**:
- Table: `hts_notes`
- Key columns: `chapter`, `subchapter`, `label`, `path` (text array), `parent_label`, `content`, `raw_html`
- Indexes: GIN index on `path` array, B-tree on `label`

**Running**:
```bash
python agent/chapter99note-agent/notes_agent.py --dsn postgresql://...
```

**Label Format**:
- Normalized format: `note(16)(a)(ii)`
- Path array: `['note', '16', 'a', 'ii']`
- Query function: `get_note(conn, "note(16)(a)")` returns all matching and descendant rows

## Python Environment

**Setup**:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

**Key Dependencies**:
- `psycopg2-binary`: PostgreSQL adapter
- `Flask` + `Flask-CORS`: Web API framework (not yet implemented)
- `selenium`: Browser automation for scraping
- `openpyxl`: Excel file handling
- `beautifulsoup4`: HTML parsing (used by notes agent)
- `requests`: HTTP client

## Development Patterns

### Adding New Agents

Follow the established agent pattern:

1. Create directory under `agent/` with descriptive name
2. Implement standalone Python script with:
   - `--dsn` argument for database connection
   - `--data-dir` or equivalent for input data location
   - DDL creation/migration in `ensure_table()` or `db_init()` function
   - Idempotent upsert logic when possible
3. Use descriptive docstrings explaining data fixes and transformations
4. Add example command to script.txt

### Database Schema Management

- Each agent manages its own table(s) via DDL in the script
- Use `IF NOT EXISTS` for idempotent table creation
- Add columns with `ADD COLUMN IF NOT EXISTS` for schema evolution
- Consider `TRUNCATE ... RESTART IDENTITY` vs upsert based on data volatility

### API Integration

- USITC endpoints require proper User-Agent headers
- Implement chapter-based caching for API calls (see `_HTS_STATUS_CACHE` in basic_hts_agent.py)
- Handle both standard and Chapter 99 endpoints
- Use fallback from `requests` to `urllib` for maximum compatibility

## Data Characteristics

- HTS numbers may have missing leading zeros (e.g., "410" should be "0410")
- Hierarchy is indicated by indent levels (0-based integer)
- Parent-child relationships span indentation gaps
- Duty rates inherit from ancestors when blank
- Chapter 99 notes have deeply nested structure requiring recursive parsing
