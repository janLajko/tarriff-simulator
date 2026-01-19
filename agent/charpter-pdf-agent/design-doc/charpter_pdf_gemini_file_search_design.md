Design: Upload Chapter PDF Notes to Gemini File Search

Goal
- Upload all PDFs in agent/charpter-pdf-agent/charpter-data to the Gemini File Search store named tarriff-simulate.
- Attach metadata with two keys: charpter and note.

Inputs
- Directory: agent/charpter-pdf-agent/charpter-data
- File naming format: Subchapter<roman>_USNote_<nn>.pdf

Metadata Rules
- charpter: "Subchapter <roman>" (insert a space before the roman numerals, e.g., SubchapterIII -> Subchapter III).
- note: "note(n)" where n is the numeric suffix without leading zeros (e.g., 01 -> note(1)).

Processing Flow
- Scan the directory and collect files ending in .pdf.
- Sort filenames to keep processing stable and repeatable.
- For each file:
  - Parse the roman numeral from the Subchapter portion.
  - Parse the numeric note suffix.
  - Build metadata with keys charpter and note.
  - Upload to the Gemini File Search store "tarriff-simulate" with display_name set to the filename.

Assumptions and Constraints
- All filenames conform to the required pattern.
- No extra metadata fields are added beyond charpter and note.

Validation
- Optionally list the store documents after upload and spot-check that metadata matches expectations.
