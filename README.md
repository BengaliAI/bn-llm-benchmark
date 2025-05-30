# bn-llm-benchmark

This repository contains scripts for evaluating and translating of differengt NLP datasets into Bangla.

## Repository structure

- **📂** = folder  
- **📄** = file

A highlevel overview of the structure

```plaintext
├── 📂 src
│   ├── 📄 infer.py (Main inference script)
│   ├── 📄 metrics.py (Script related to the calculation of metrics used in this project)
│   ├── 📄 prompt_types.py (Different prompt types based on dataset and language)
│   ├── 📄 score.py (Calculation model performance based on inference by using metrics)
│   ├── 📄 llm_judge_eval.py (Conducts LLM as a Judge Evaluation)
│   ├── 📄 score_aggregator.py (Helper script to organize results)
│   ├── 📄 tokenizer.py (Script to generate tokenizer counts for different datasets)
│   └── 📄 tokenization_summarizer.py (Summarizes tokenization findings)
├── 📂 tokenization-results (Contains results of tokenization)
├── 📂 translation
│   ├── 📄 translate.py (Translation script)
│   └── 📄 parse_errors.py (Post translation error fix script)
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

## LLM Eval Scoring

In order to have an llm evaluate the results, we introduced an `llm_judge_eval.py` script. We are using the `gpt-4o-mini-2024-07-18` as the model to evaluate our results.

After filling in your open ai api key credentials in a .env file run : 

```bash
python src/llm_eval_judge.py inference-outputs/openbookqa-en/
```

in order to run the script on the inferences found from English OpenbookQA dataset.

## Result Aggregation

To make sure all of the results are organized properly run : 

```bash
python src/score_aggregator.py --input_dir inference-outputs/
```

This will create a centralized result folder which will look like:

```plaintext
├── 📂 results
│   ├── 📂 bn (Contains aggregated results for Bangla)
│   ├── 📄 accuracy.csv
│   ├── 📄 llm_eval.csv
│   ├── 📄 rer.csv
│   ├── 📂 en (Contains aggregated results for English)
│   ├── 📄 accuracy.csv
│   ├── 📄 llm_eval.csv
│   └── 📄 rer.csv
```

## Tokenization

To generate tokenization for different datasets run : 

```bash
python src/tokenization.py --input_dir inference-outputs/
```

This generates tokenization based on per row and per word. From that we also want to summarize findings from tokenization for which we run the script : 

```bash
python src/tokenization_summarizer.py
```

This creates new metrics like `avg_bytes_per_token` and `avg_normalized_seq_len` which provide more insight on tokenization. These metrics are are the same across counts-per-word and counts-per-column.


