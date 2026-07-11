
import torch
import gc
import json
import copy
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # TrainingArguments,
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
import os, random
import numpy as np
from transformers import set_seed
from src.common.model_utils import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define all languages and tasks to loop over
# languages = ["en", "hi", "ar", "es", "zh-cn"]
languages = ["hi", "ar", "es", "zh-cn"]
tasks = ["AB", "BC", "CA"]
# seeds = [0, 1, 2]
seeds = [1,2]
size = "3B"
model_name = f'models/meta-llama/Llama-3.2-{size}'

# Single CSV file for results
results_csv = f'dpo_training_results_{size}_datav2.csv'

# Create headers for CSV file if it doesn't exist
def initialize_csv_file():
    columns = ['seed', 'model_size', 'language', 'task', 'accuracy', 'correct_count', 'incorrect_count', 'option_a_count', 'option_b_count']
    
    if not os.path.exists(results_csv):
        pd.DataFrame(columns=columns).to_csv(results_csv, index=False)

def save_result(result_entry):
    """Save result to CSV file"""
    # Add to results
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        new_df = pd.concat([existing_df, pd.DataFrame([result_entry])], ignore_index=True)
    else:
        new_df = pd.DataFrame([result_entry])
    new_df.to_csv(results_csv, index=False)
    
    print(f"Result saved to {results_csv} for seed={result_entry['seed']}, {result_entry['language']}-{result_entry['task']}")

def experiment_exists(seed, lang, task):
    """Check if experiment result already exists in CSV"""
    if not os.path.exists(results_csv):
        return False
    
    existing_df = pd.read_csv(results_csv)
    mask = (existing_df['seed'] == seed) & (existing_df['language'] == lang) & (existing_df['task'] == task)
    return mask.any()

# Initialize CSV file
initialize_csv_file()

# Define helper functions
def prepare_inputs(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='pt')
    return inputs

def get_answer(row):
    for char in row["model_response"]:
        if char == "A":
            return "A"
        elif char == "B":
            return "B"
    return ""

def get_accuracy(row):
    if row["model_answer"] == row["response"]:
        return 1    # chose correct
    elif row["model_answer"] in ["A", "B"]:
        return 0    # chose incorrect
    return -1       # invalid response

def infer_run_eval(model, tokenizer, experiment, modelname, input_file, inputdata, lang):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not os.path.exists(f"datav2/model_test_generations/{experiment}/{lang}/{modelname}"):
        os.makedirs(f"datav2/model_test_generations/{experiment}/{lang}/{modelname}")

    with open(input_file, "r") as file:
        data = json.load(file)

    model.to("cuda")
    out_jsons = []
    for entry in tqdm(data):
        out_obj = copy.deepcopy(entry)

        inputs = prepare_inputs([entry["prompt"]], tokenizer).to('cuda')
        output = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        out_text = tokenizer.decode(output[0], skip_special_tokens=False)
        
        out_obj["out_text"] = out_text
        if lang=="en":
            out_obj["model_response"] = out_text.split("The correct option:")[1]
        elif lang=="hi":
            out_obj["model_response"] = out_text.split("सही विकल्प:")[1]
        elif lang=="es":
            out_obj["model_response"] = out_text.split("La opción correcta:")[1]
        elif lang=="ar":
            out_obj["model_response"] = out_text.split("الخيار الصحيح:")[1]
        elif lang=="zh-cn":
            out_obj["model_response"] = out_text.split("正确选项:")[1]
        out_jsons.append(out_obj)

    with open(f"datav2/model_test_generations/{experiment}/{lang}/{modelname}/{inputdata}.json", "w", encoding="utf-8") as f:
        json.dump(out_jsons, f, indent=4, ensure_ascii=False)

# Main training loop over all seeds, languages and tasks
experiment_counter = 0

for lang in languages:
    for task in tasks:
        for seed in seeds:
            # Check if this experiment already exists
            if experiment_exists(seed, lang, task):
                print(f"Skipping seed={seed}, language={lang}, task={task} - already completed")
                continue
            
            experiment_counter += 1
            os.environ["PYTHONHASHSEED"] = str(seed)
            # Set seed at the start of each iteration for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            set_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            print(f"\n{'='*60}")
            print(f"Training DPO model [Exp: {experiment_counter}]: Seed={seed}, Language={lang}, Task={task}")
            print(f"{'='*60}")
            
            # Load converted DPO data
            trainpath = f"datav2/dpo_format/{lang}/train/{task}.json"
            devpath = f"datav2/dpo_format/{lang}/dev/{task}.json"
            testpath = f"datav2/dpo_format/{lang}/test/{task}.json"

            with open(trainpath, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            with open(devpath, 'r', encoding='utf-8') as f:
                dev_data = json.load(f)
            with open(testpath, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            # Create Hugging Face DatasetDict
            dataset = DatasetDict({
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(dev_data),
                'test': Dataset.from_list(test_data)
            })

            print(f"Train dataset size: {len(dataset['train'])}")
            print(f"Dev dataset size: {len(dataset['validation'])}")
            print(f"Test dataset size: {len(dataset['test'])}")

            # Load model and tokenizer
            print("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to("cuda")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            # Configure LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            print("Applying LoRA...")
            model = get_peft_model(model, lora_config)

            # Output directory
            out_dir = f"models/v2/dpo_finetuned/{lang}/{size}_{task}_{seed}"
            os.makedirs(out_dir, exist_ok=True)

            # Training arguments
            if lang == "hi" or lang == "ar":
                train_batch_size = 4
                grad_accum_steps = 2
            else:
                train_batch_size = 8
                grad_accum_steps = 1
            training_args = DPOConfig(
                seed=seed,
                output_dir=out_dir,
                overwrite_output_dir=True,
                num_train_epochs=5,
                per_device_train_batch_size=train_batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                per_device_eval_batch_size=8,
                evaluation_strategy='steps',
                eval_steps=200,
                save_steps=500,
                logging_steps=100,
                learning_rate=2e-5,
                warmup_ratio=0.05,
                weight_decay=0.01,
                save_total_limit=1,
                fp16=True,
                remove_unused_columns=False,
                report_to=None,
                # DPO-specific parameters
                beta=0.1,  # DPO regularization parameter
                max_prompt_length=400,
                max_length=512,
            )

            # Initialize DPO Trainer
            print("Initializing DPO trainer...")
            dpo_trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],  # Use dev set for validation during training
                processing_class=tokenizer,
            )

            print("Starting DPO training...")
            dpo_trainer.train()

            print("Evaluating model after training...")
            eval_results = dpo_trainer.evaluate()
            print(f"Evaluation results: {eval_results}")

            print("Saving LoRA model...")
            dpo_trainer.save_model(f"models/v2/lora_dpo_finetuned/{lang}/{size}_{task}_{seed}")
            tokenizer.save_pretrained(f"models/v2/lora_dpo_finetuned/{lang}/{size}_{task}_{seed}")

            # Clear memory after training
            del model, dpo_trainer
            gc.collect()
            torch.cuda.empty_cache()

            # Load the trained model for evaluation
            print("Loading trained model for evaluation...")
            modelname = f"models/v2/lora_dpo_finetuned/{lang}/{size}_{task}_{seed}"
            model, tokenizer = load_model(modelname)

            print("Running evaluation...")
            experiment = "dpo_finetuned"
            modelname = f"{size}_{task}_{seed}"
            input_file = f"datav2/prompt_response/{lang}/test/{task}.json"
            infer_run_eval(model, tokenizer, experiment, modelname, input_file, task, lang)

            # Calculate accuracy and collect results
            filepath = f"datav2/model_test_generations/{experiment}/{lang}/{modelname}/{task}.json"
            df = pd.read_json(filepath)
            df['model_answer'] = df.apply(get_answer, axis=1)
            df['model_acc'] = df.apply(get_accuracy, axis=1)

            correct_count = (df["model_acc"] == 1).sum()
            incorrect_count = (df["model_acc"] == 0).sum()
            option_a_count = (df["model_answer"] == "A").sum()
            option_b_count = (df["model_answer"] == "B").sum()
            total_samples = len(df)
            accuracy = correct_count / total_samples if total_samples > 0 else 0

            print(f"Number of times chose correct: {correct_count}")
            print(f"Number of times chose incorrect: {incorrect_count}")
            print(f"Number of times model chose first option: {option_a_count}")
            print(f"Number of times model chose second option: {option_b_count}")
            print(f"Accuracy: {accuracy:.4f}")

            # Store results for CSV export
            result_entry = {
                'seed': seed,
                'model_size': size,
                'language': lang,
                'task': task,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'option_a_count': option_a_count,
                'option_b_count': option_b_count
            }
            
            # Save result to CSV
            save_result(result_entry)
            
            # Clean up memory
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Completed training and evaluation [{experiment_counter}] for seed={seed}, {lang}-{task}")

# Final summary
print("\n" + "="*60)
print("All iterations completed! Final summary...")
print("="*60)

# Read the final results for summary
if os.path.exists(results_csv):
    final_results_df = pd.read_csv(results_csv)
    
    print(f"Results available at: {results_csv}")
    
    # Print summary statistics
    print("\nOverall Summary:")
    print(f"Total experiments completed: {len(final_results_df)}")
    print(f"Seeds tested: {final_results_df['seed'].nunique()}")
    print(f"Languages tested: {final_results_df['language'].nunique()}")
    print(f"Tasks tested: {final_results_df['task'].nunique()}")
    print(f"Average accuracy across all experiments: {final_results_df['accuracy'].mean():.4f}")
    
    if len(final_results_df) > 0:
        best_idx = final_results_df['accuracy'].idxmax()
        worst_idx = final_results_df['accuracy'].idxmin()
        best_row = final_results_df.loc[best_idx]
        worst_row = final_results_df.loc[worst_idx]
        print(f"Best performing combination: seed={best_row['seed']}, {best_row['language']}-{best_row['task']} (Accuracy: {final_results_df['accuracy'].max():.4f})")
        print(f"Worst performing combination: seed={worst_row['seed']}, {worst_row['language']}-{worst_row['task']} (Accuracy: {final_results_df['accuracy'].min():.4f})")
else:
    print("No results found. Check if any iterations completed successfully.")

print("\nAll training and evaluation completed successfully!")
