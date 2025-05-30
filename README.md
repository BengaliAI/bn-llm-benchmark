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

- **Data preparation**  
  Scripts like `create_jsonl.py` ([MATH](src/translation/MATH/test/create_jsonl.py)) to combine raw files into a single JSONL.

- **Translation**  
  `translate.py` scripts (e.g. [src/translation/MATH/train/counting_and_probability_jsonl/translate.py](src/translation/MATH/train/counting_and_probability_jsonl/translate.py)) that:
  - Read a JSONL of examples.
  - Call an API (e.g. OpenAI) with rate-limits, retries, and backoff.
  - Save outputs to `<dataset>_translated.jsonl`.

- **Error handling & post-processing**  
  `parse_errors.py` files (e.g. [src/translation/MATH/test/geometry_jsonl/parse_errors.py](src/translation/MATH/test/geometry_jsonl/parse_errors.py)) to:
  - Detect JSON decode errors in the translated output.
  - Retry translations with adjusted prompts or temperature.
  - Fix and escape malformed JSON fields.

### Special cases

- **AlpacaEval** uses a HuggingFace Dataset builder:  
  see [`AlpacaFarmDataset`](src/translation/AlpacaEval/alpaca_eval/alpaca_eval.py) for loading, splitting and defining features.

- **BigBenchHard** provides a simple script to dump HF datasets to JSONL:  
  [src/translation/BigBenchHard/create.py](src/translation/BigBenchHard/create.py)