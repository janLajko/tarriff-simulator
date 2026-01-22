#!/bin/bash

# Test script for running section301_batch_agent.py

echo "Testing Section 301 Batch Agent"
echo "================================"

# Set environment variables
export DATABASE_DSN="postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate"
export OPENAI_API_KEY="${OPENAI_API_KEY:-your_openai_key_here}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-your_gemini_key_here}"

# Change to project root directory
cd /Users/jan/Documents/7788/tariff-simulate-v2

# Check if Note 20 file exists
NOTE20_FILE="agent/charpter-pdf-agent/charpter-data-txt/SubchapterIII_USNote_20.txt"
if [ -f "$NOTE20_FILE" ]; then
    echo "✓ Note 20 file found at: $NOTE20_FILE"
    echo "  File size: $(wc -c < "$NOTE20_FILE") bytes"
else
    echo "✗ Note 20 file not found at: $NOTE20_FILE"
fi

# Run the agent
echo ""
echo "Running Section 301 agent..."
python3 agent/section301-agent/section301_batch_agent.py --log-level INFO

echo ""
echo "Test complete!"