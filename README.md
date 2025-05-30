# bn-llm-benchmark

This repository contains scripts for evaluating and translating of differengt NLP datasets into Bangla.

## Repository structure

- **📂** = folder  
- **📄** = file

A highlevel overview of the structure

```plaintext
├── 📂 src
│   ├── 📄 file1.py
│   └── 📂 translation
│       ├── 📄 translate.py (Translation script)
│       └── 📄 parse_errors.py (Post translation error fix script)
├── 📄 README.md
└── 📄 .gitignore
```


## `src/translation/`

The `translation/` directory houses all code used to translate various English NLP datasets into Bangla.  
Each dataset subfolder typically includes:

- **Translation**  
  `translate.py` scripts that:
  - Read a JSONL of examples.
  - Call an API (e.g. OpenAI) with rate-limits, retries, and backoff.
  - Save outputs to `<dataset>_translated.jsonl`.

- **Error handling & post-processing**  
  `parse_errors.py` files to:
  - Detect JSON decode errors in the translated output.
  - Retry translations with adjusted prompts or temperature.
  - Fix and escape malformed JSON fields.