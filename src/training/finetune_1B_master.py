import gc
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import random
import json
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel


SEED = 42
size = "1B"
tasks = ["AB", "BC", "CA", "BA", "CB", "AC"]
# tasks = ["CA", "AC"]
langs = ["hi", "ar", "es", "zh-cn"]
# langs = ["en"]
print("\n\n\nSTARTING UP!!!!!!")


# Initialize CSV file for results
csv_filename = f"finetuning_results_v2_{size}_5epochs.csv"
csv_headers = ['model_size', 'language', 'task', 'accuracy', 'correct_count', 'incorrect_count', 'option_a_count', 'option_b_count']

# Create CSV file with headers if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

def save_results_to_csv(model_size, language, task, accuracy, correct_count, incorrect_count, option_a_count, option_b_count):
    """Save results to CSV file"""
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model_size, language, task, accuracy, correct_count, incorrect_count, option_a_count, option_b_count])
    print(f"Results saved to {csv_filename}")


# Preprocessing function for tokenization
def preprocess_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    max_length = 512  # Adjust based on your prompt and response length
    max_response_tokens = 5

    for prompt, response in zip(examples['prompt'], examples['response']):
        # Tokenize prompt and response
        tokenized_response = tokenizer(
            response, truncation=True, max_length=max_response_tokens, add_special_tokens=False
        )
        space_for_prompt = max_length - len(tokenized_response['input_ids'])
        tokenized_prompt = tokenizer(prompt)

        tokenized_prompt['input_ids'] = tokenized_prompt['input_ids'][-space_for_prompt:]
        tokenized_prompt['attention_mask'] = tokenized_prompt['attention_mask'][-space_for_prompt:]

        # Concatenate prompt and response
        input_ids = tokenized_prompt['input_ids'] + tokenized_response['input_ids']
        attention_mask = tokenized_prompt['attention_mask'] + tokenized_response['attention_mask']

        # Create labels (mask the prompt part)
        labels = [-100]*len(tokenized_prompt['input_ids']) + tokenized_response['input_ids']

        # Pad sequences to max_length
        padding_length = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id]*padding_length
        attention_mask += [0]*padding_length
        labels += [-100]*padding_length

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }

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


for lang in langs:
    print("Starting Language: ", lang)
    for task in tasks:
        print("\nStarting new tas:", task)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        set_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        trainpath = f"datav2/prompt_response/{lang}/train/{task}.json"
        devpath = f"datav2/prompt_response/{lang}/dev/{task}.json"

        with open(trainpath, 'r') as f:
            train_data = json.load(f)
        with open(devpath, 'r') as f:
            dev_data = json.load(f)

        # Create Hugging Face DatasetDict
        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'dev': Dataset.from_list(dev_data)
        })

        # Specify the model name
        model_name = f'models/meta-llama/Llama-3.2-{size}'

        print("load model")
        model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                )
        model = model.to('cuda')

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Configure LoRA with PEFT
        lora_config = LoraConfig(
            r=4,  # Low-rank dimension
            lora_alpha=8,
            lora_dropout=0.1,
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )

        print("peft model")
        model = get_peft_model(model, lora_config)

        # Apply the preprocessing function
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=['prompt', 'response']
        )

        out_dir = f"models/v2/llama_finetuned/{lang}/{size}_{task}"

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=8,
            evaluation_strategy='steps',
            eval_steps=200,
            save_steps=500,
            logging_steps=100,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            save_total_limit=1,
            fp16=True,
            push_to_hub=False
        )


        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['dev'],
            tokenizer=tokenizer,
        )

        print("evaluate model before training")
        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")

        print("start training")
        # Train the model
        trainer.train()

        print("evaluate model after training")
        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")


        # Save the fine-tuned model
        print("save lora model")
        trainer.save_model(f"models/v2/lora_llama_finetuned/{lang}/{size}_{task}")
        tokenizer.save_pretrained(f"models/v2/lora_llama_finetuned/{lang}/{size}_{task}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(f"models/meta-llama/Llama-3.2-{size}")
        m1 = PeftModel.from_pretrained(base_model, f"models/v2/lora_llama_finetuned/{lang}/{size}_{task}")

        print("merge model")
        m1 = m1.merge_and_unload()

        print("save merged model")
        # m1.save_pretrained(f"models/llama_finetuned/{lang}/{size}_{task}")
        # tokenizer.save_pretrained(f"models/llama_finetuned/{lang}/{size}_{task}")

        print("Task done:", task, "for language:", lang)
        print("Start evaluation on test set for task:", task, "and language:", lang)

        print("Running evaluation...")
        experiment = "llama_finetuned"
        modelname = f"{size}_{task}"
        input_file = f"datav2/prompt_response/{lang}/test/{task}.json"
        infer_run_eval(m1, tokenizer, experiment, modelname, input_file, task, lang)

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

        # Save results to CSV
        save_results_to_csv(
            model_size=size,
            language=lang,
            task=task,
            accuracy=round(accuracy, 4),
            correct_count=int(correct_count),
            incorrect_count=int(incorrect_count),
            option_a_count=int(option_a_count),
            option_b_count=int(option_b_count)
        )

        del m1, base_model
        gc.collect()
        torch.cuda.empty_cache()

print("\nTraining hyperparameters:")
for key, value in training_args.to_dict().items():
    print(f"{key}: {value}")