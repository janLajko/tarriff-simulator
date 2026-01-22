# Section 301 Batch Agent - Database Query Update

## Changes Made

The Section 301 batch agent has been updated to automatically fetch headings from the database instead of requiring them as command-line arguments.

### Key Updates

1. **New Database Method**: `fetch_section301_headings()`
   - Executes SQL query to find all Section 301 headings
   - Query criteria:
     - `hts_number LIKE '9903.88%'` - All Section 301 headings
     - `status IS NULL` - Active headings only
     - `description NOT LIKE '%provision suspended%'` - Exclude suspended provisions

2. **Modified `run()` Method**
   - Now accepts optional `headings` parameter
   - If no headings provided, automatically fetches from database
   - Logs the number of headings found

3. **Removed `--headings` CLI Argument**
   - No longer needed since headings are fetched from database
   - Simplifies command-line usage

## SQL Query Used

```sql
SELECT hts_number
FROM hts_codes
WHERE hts_number LIKE '9903.88%'
  AND status IS NULL
  AND description NOT LIKE '%provision suspended%'
ORDER BY hts_number
```

## Usage

### Before (with --headings)
```bash
python section301_batch_agent.py \
  --dsn postgresql://... \
  --headings 9903.88.01 9903.88.02 9903.88.03
```

### Now (automatic database query)
```bash
# Automatically fetches all Section 301 headings
python section301_batch_agent.py \
  --dsn postgresql://...

# Optional: use permissive mode if LLMs don't match
python section301_batch_agent.py \
  --dsn postgresql://... \
  --no-require-match
```

## Testing

Use the provided test script to verify the database query:

```bash
# Set environment variable
export DATABASE_DSN="postgresql://user:pass@host:port/database"

# Run test
python test_db_query.py
```

Expected output:
```
2024-01-20 10:30:00 - INFO - Connected to database
2024-01-20 10:30:01 - INFO - Found 15 Section 301 headings:
2024-01-20 10:30:01 - INFO -   - 9903.88.01
2024-01-20 10:30:01 - INFO -   - 9903.88.02
2024-01-20 10:30:01 - INFO -   - 9903.88.03
...
```

## Benefits

1. **Automatic Discovery**: No need to manually specify headings
2. **Always Current**: Automatically picks up new Section 301 headings added to database
3. **Filters Out Inactive**: Excludes expired or suspended provisions
4. **Simpler Usage**: One less parameter to specify
5. **Consistent**: Always processes all relevant headings

## Environment Variables

- `DATABASE_DSN`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key
- `GEMINI_API_KEY`: Google Gemini API key
- `OPENAI_MODEL`: (Optional) OpenAI model name (default: gpt-4o)
- `GEMINI_MODEL`: (Optional) Gemini model name (default: gemini-2.0-flash-exp)

## Error Handling

- If no Section 301 headings found, logs warning and exits gracefully
- If database connection fails, appropriate error message is shown
- Transaction rollback on any processing errors