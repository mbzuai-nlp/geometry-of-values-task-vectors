# Setup

Create a Python environment with GPU-enabled PyTorch appropriate for your CUDA/runtime setup, then install the project dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece matplotlib pandas numpy scikit-learn tqdm peft trl bitsandbytes openai google-genai googletrans==4.0.0-rc1 jsonlines tenacity
```

The training and evaluation scripts expect Llama checkpoints at:

```text
models/meta-llama/Llama-3.2-1B
models/meta-llama/Llama-3.2-3B
```

For GPT-5-mini evaluation scripts:

```bash
export OPENAI_API_KEY=...
```

For the data generation and Gemini validation pipeline:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python -m src.data_generation.generate_data
python -m src.data_generation.validate_dilemmas_gemini3
python -m src.data_generation.convert_validated_to_original
```

The generated raw responses (`openai_responses/`) and Gemini validation outputs (`datav2/validated_responses_g3/`) are ignored by Git.

Run scripts from the repository root using module paths when possible:

```bash
python -m src.evaluation.evaluate_baseline_all
python -m src.training.train_dpo_all_1B
python -m src.task_vectors.task_vec_1B
```
