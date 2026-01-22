# Section 301 Batch Agent - Note 20 Full Text Processing

## Overview

The Section 301 batch agent has been updated to implement a simplified batch processing approach where the entire Note 20 content is provided to the LLM, rather than extracting specific subsections.

## Key Changes

### 1. **Simplified Note Processing**

**Before (Complex):**
- Parse HTS descriptions to extract note references (e.g., "note 20(a)")
- Query database for specific note subsections
- Extract and send only relevant subsections to LLM

**After (Simplified):**
- Load complete Note 20 text from file
- Send full Note 20 content to LLM with all HTS headings
- Let LLM match references to subsections

### 2. **New File Loading Method**

Added `load_note20_from_file()` method that:
- Checks multiple possible file locations
- Falls back to database if file not found
- Supports flexible directory structures

```python
possible_paths = [
    Path(__file__).parent.parent / "chapter-pdf-agent" / "chapter-data" / "note20.txt",
    Path(__file__).parent.parent / "charpter-pdf-agent" / "charpter-data" / "note20.txt",
    Path(__file__).parent / "data" / "note20.txt",
]
```

### 3. **Updated LLM Prompt**

The prompt now includes:
- Complete Note 20 text as context
- Instructions for LLM to match note references to subsections
- Request for actual HTS codes from subsections, not just "note20(a)"

```python
LLM_BATCH_PROMPT = """
Here is the complete Note 20 to Chapter 99 of the HTS:
===== NOTE 20 COMPLETE TEXT =====
{note20_content}
===== END OF NOTE 20 =====

Now analyze these HTS headings and their descriptions:
{headings_json}

For each heading:
1. Identify which subsection(s) of Note 20 it references
2. Based on the Note 20 content above, list all HTS codes in those subsections as "includes"
3. Identify any exclusions mentioned
...
"""
```

### 4. **Modified Processing Flow**

```
1. Fetch Section 301 headings from database (9903.88%)
2. Fetch HTS descriptions for all headings
3. Load complete Note 20 from file
4. Send all data to LLMs in single batch
5. LLMs extract HTS codes from referenced subsections
6. Compare results and insert to database
```

## File Structure

```
agent/
├── section301-agent/
│   ├── section301_batch_agent.py  # Main agent (updated)
│   ├── data/
│   │   └── note20.txt             # Note 20 content (primary location)
│   └── test_note20_load.py        # Test script
└── chapter-pdf-agent/
    └── chapter-data/
        └── note20.txt              # Alternative location for Note 20
```

## Benefits

1. **Simplicity**: No complex note parsing logic
2. **Accuracy**: LLM sees full context for better understanding
3. **Flexibility**: Easy to update Note 20 content
4. **Performance**: Single LLM call processes all headings
5. **Token Efficiency**: Note 20 is ~5-10K tokens, acceptable for modern LLMs

## Usage

### Setup Note 20 File

Place the complete Note 20 text in one of these locations:
- `agent/section301-agent/data/note20.txt` (recommended)
- `agent/chapter-pdf-agent/chapter-data/note20.txt`
- `agent/charpter-pdf-agent/charpter-data/note20.txt`

### Run the Agent

```bash
# Set environment variables
export DATABASE_DSN="postgresql://..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."

# Run agent (automatically fetches headings and Note 20)
python agent/section301-agent/section301_batch_agent.py
```

### Test Note 20 Loading

```bash
python agent/section301-agent/test_note20_load.py
```

Expected output:
```
Successfully loaded Note 20 (XXXX characters)
Found subsections: (a), (b), (c), (d), ... (vvv)
Found XXX HTS codes in Note 20
```

## Note 20 Format

The Note 20 file should follow this format:

```
Note 20.

(a) For the purposes of heading 9903.88.01, products of China classified in the following HTS subheadings:
0203.29.20    0203.29.40    0206.10.00    ...

(b) For the purposes of heading 9903.88.02, products of China classified in the following HTS subheadings:
0304.63.00    0304.71.50    0304.82.50    ...

...

(vvv) For the purposes of heading 9903.88.69, products of China classified in the following HTS subheadings:
8501.10.20    8501.10.40    8501.10.60    ...
```

## Token Usage Estimation

- Note 20 full text: ~5,000-10,000 tokens
- HTS headings (15-20): ~2,000 tokens
- Prompt framework: ~500 tokens
- **Total per LLM call**: ~7,500-12,500 tokens

This is well within the context window of modern LLMs (GPT-4o: 128K, Gemini: 1M+).

## Migration from Database

If Note 20 is currently in the database, export it:

```sql
SELECT string_agg(content, E'\n' ORDER BY id) as full_content
FROM hts_notes
WHERE chapter = 99
  AND label LIKE 'note20%'
GROUP BY chapter;
```

Save the output to `agent/section301-agent/data/note20.txt`.

## Error Handling

- If Note 20 file not found → Falls back to database query
- If database query fails → Logs error and aborts processing
- If LLM fails to match subsections → Logged in comparison report

## Future Enhancements

1. **Cache Note 20**: Load once and reuse across multiple runs
2. **Support Multiple Notes**: Extend to handle Notes 21, 22, etc.
3. **Auto-update**: Periodically check for Note 20 updates
4. **Validation**: Verify extracted HTS codes against known patterns