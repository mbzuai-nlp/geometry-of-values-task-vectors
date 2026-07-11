# Geometry of Values

Code and final data for **Geometry of Values: Task Vector Composition for Ethical Preference Alignment in Language Models**.

The project studies ethical preference alignment in LLMs over three value conflicts:

- `AB`: Honesty over Justice
- `BC`: Justice over Autonomy
- `CA`: Autonomy over Honesty
- `BA`, `CB`, `AC`: reverse-preference variants

Languages: English (`en`), Hindi (`hi`), Arabic (`ar`), Spanish (`es`), and Chinese (`zh-cn`).

## Repository Contents

Final data:

- `datav2/originalAB/`: final 12,000 English dilemma dataset.
- `datav2/translated_responses/`: final translated dilemma data.
- `datav2/prompt_response/`: train/dev/test prompt-response format used for LoRA fine-tuning, baseline evaluation, and task-vector evaluation.
- `datav2/dpo_format/`: train/dev/test preference-pair format used for DPO.
- `goldv2/`: final human-authored gold evaluation set and prompt-response variants.

Prompt sources:

- `prompts/prompt0.md`, `prompts/prompt1.md`, `prompts/prompt2.md`: prompt templates used for raw dilemma generation.
- `prompts/scenarios.md`: scenario list used by the generation script.
- `prompts/arif/`, `prompts/settings/`: additional prompt drafts/settings kept for provenance.

Final aggregate results:

- `baseline_results_datav2.csv`
- `finetuning_results_v2_1B_5epochs.csv`
- `finetuning_results_v2_3B_5epochs.csv`
- `dpo_training_results_1B_datav2.csv`
- `dpo_training_results_3B_datav2.csv`
- `task_vector_results_1B_all_datav2.csv`
- `task_vector_results_3B_all_datav2.csv`
- `performance_transfer_eval_results_v2.csv`
- `results_tv/combined_task_vector_results_datav2.csv`

Plot scripts are included under `src/plotting/`, but generated plot files are not part of the release.

## Code Layout

Common utilities:

- `src/common/model_utils.py`

Data generation:

- `src/data_generation/generate_data.py`: OpenAI-based raw dilemma generation using `prompts/`.
- `src/data_generation/validate_dilemmas_gemini3.py`: Gemini validation and repair of generated dilemmas.
- `src/data_generation/convert_validated_to_original.py`: flatten validated Gemini outputs into `datav2/originalAB/`.

Data preparation:

- `src/data_preparation/data.py`, `data_ar.py`, `data_cn.py`, `data_es.py`, `data_hi.py`: build prompt-response data from final source/translated data.
- `src/data_preparation/create_dev_split.py`: split training data into train/dev.
- `src/data_preparation/convert_to_dpo_data.py`: convert prompt-response files into DPO preference-pair format.
- `src/data_preparation/data_goldv2.py`: build goldv2 prompt-response data.
- `src/data_preparation/reorder_translated_responses.py`: align translated files with the English source ordering.
- `src/data_preparation/translate_data*.py`: translation helpers.

Training:

- `src/training/finetune_1B_master.py`, `src/training/finetune_3B_master.py`: LoRA fine-tuning.
- `src/training/train_dpo_all_1B.py`, `src/training/train_dpo_all_3B.py`: DPO training.

Evaluation:

- `src/evaluation/evaluate_baseline_all.py`: baseline Llama-3.2-1B/3B evaluation.
- `src/evaluation/evaluate_goldv2_models.py`: goldv2 evaluation.
- `src/evaluation/performance_transfer_eval.py`: cross-lingual transfer evaluation.
- `src/evaluation/gpt5mini/`: GPT-5-mini in-context policy evaluation scripts.

Task vectors:

- `src/task_vectors/task_vec_1B.py`, `src/task_vectors/task_vec_3B.py`: task-vector composition and grid search.

Plotting:

- `src/plotting/graph.py`
- `src/plotting/graph_dpo.py`
- `src/plotting/compare_tv_plots.py`
- `src/plotting/plot_gamma_scatter_tv.py`

Run module-style commands from the repository root, for example:

```bash
python -m src.evaluation.evaluate_baseline_all
python -m src.training.train_dpo_all_1B
python -m src.task_vectors.task_vec_1B
```

## Setup

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece matplotlib pandas numpy scikit-learn tqdm peft trl bitsandbytes openai google-genai googletrans==4.0.0-rc1 jsonlines tenacity
```

The training and task-vector scripts expect local model checkpoints under:

```text
models/meta-llama/Llama-3.2-1B
models/meta-llama/Llama-3.2-3B
```

Model checkpoints and generated model outputs are intentionally ignored by Git.

For GPT-5-mini evaluation scripts, set:

```bash
export OPENAI_API_KEY=...
```

For data generation and Gemini validation, set:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python -m src.data_generation.generate_data
python -m src.data_generation.validate_dilemmas_gemini3
python -m src.data_generation.convert_validated_to_original
```

The raw generation outputs (`openai_responses/`) and Gemini validation outputs (`datav2/validated_responses_g3/`) are generated intermediates and are ignored by Git.

## Notes

The release keeps final datasets, final aggregate result tables, and reusable code. Working notebooks, logs, raw API responses, old `data/` artifacts, model generations, and generated plots were moved out of the release tree into local `discard/` during cleanup.
