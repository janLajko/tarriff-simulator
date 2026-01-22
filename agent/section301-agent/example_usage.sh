#!/bin/bash
# Example usage scripts for section301_batch_agent.py with dual LLM verification

# Set your environment variables first
export DATABASE_DSN="postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate"
export OPENAI_API_KEY="your-openai-key-here"
export GEMINI_API_KEY="your-gemini-key-here"

# Example 1: Basic usage with strict mode (default)
# Only inserts to database if OpenAI and Gemini results match exactly
echo "Example 1: Strict mode (require match)"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN"

# Example 2: Permissive mode (insert even if results differ)
echo ""
echo "Example 2: Permissive mode (no require match)"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --no-require-match

# Example 3: Process specific headings only
echo ""
echo "Example 3: Specific headings"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --headings 9903.88.01 9903.88.02 9903.88.03

# Example 4: Use custom models
echo ""
echo "Example 4: Custom models"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --openai-model gpt-4-turbo \
  --gemini-model gemini-1.5-pro

# Example 5: Debug mode with verbose logging
echo ""
echo "Example 5: Debug mode"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --log-level DEBUG

# Example 6: All headings with custom models in permissive mode
echo ""
echo "Example 6: Complete example"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --headings 9903.88.01 9903.88.02 9903.88.03 9903.88.04 9903.88.15 9903.88.69 9903.88.70 \
  --openai-model gpt-4o \
  --gemini-model gemini-2.0-flash-exp \
  --log-level INFO \
  --no-require-match

# Example 7: Using environment variable for DSN
echo ""
echo "Example 7: Using environment variable"
export DATABASE_DSN="postgresql://postgres:password@localhost:5432/tariff-simulate"
python section301_batch_agent.py  # DSN will be read from env var

# Example 8: Quick test with single heading
echo ""
echo "Example 8: Quick single heading test"
python section301_batch_agent.py \
  --dsn "$DATABASE_DSN" \
  --headings 9903.88.01 \
  --log-level INFO
