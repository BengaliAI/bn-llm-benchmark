# bn-llm-benchmark

This repository contains scripts for evaluating and translating of differengt NLP datasets into Bangla.

## Repository structure

- **📂** = folder  
- **📄** = file

A highlevel overview of the structure

```plaintext
├── 📂 src
│   ├── 📄 infer.py (Main inference script)
    ├── 📄 metrics.py (Script related to the calculation of metrics used in this project)
    ├── 📄 prompt_types.py (Different prompt types based on dataset and language)
    ├── 📄 score.py (Calculation model performance based on inference by using metrics)
    ├──
    ├──
│   └── 📂 translation
│       ├── 📄 translate.py (Translation script)
│       └── 📄 parse_errors.py (Post translation error fix script)
├── 📄 README.md
└── 📄 .gitignore
```


## `src/translation/` Directory

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

## Usage Information

After translation run the following scripts in succession: 


## Inference

In order to perform inference:

```bash
python src/infer.py --dataset_name openbookqa --dataset_path /home/LargeFiles/datasets_v1/openbookqa/test/openbookqa_test_gpt4omini.jsonl --dir_save /home/$USER/Projects/bengali-llm/output --model llama3.1:8b
```

## Scoring

After running inference, execute the scoring script with:

```bash
python src/score.py --inference_output_directory inference-outputs/
```
Be sure to update the directories you want to avoid in the `avoid_dirs` variable in the script.