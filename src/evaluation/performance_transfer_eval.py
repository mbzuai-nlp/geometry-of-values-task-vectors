import torch
import json
import copy
import os
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

source_langs = ["en", "hi", "es", "ar", "zh-cn"]
langs = ["en", "hi", "es", "ar", "zh-cn"]
modelsize = "3B"

def load_peft_model(base_model_name, peft_model_path, device='cuda'):
    """Load a PEFT model with its base model."""
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    
    print(f"Loading PEFT adapter: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_inputs(texts, tokenizer, max_length=512):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    return inputs

def run_inference(model, tokenizer, input_file, inputfile, modelname, test_lang, src_lang):
    with open(input_file, "r") as file:
        data = json.load(file)

    model.to("cuda")
    model.eval()
    
    out_jsons = []
    batch_size = 16
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {src_lang}->{test_lang} {inputfile}"):
        batch_data = data[i:i+batch_size]
        prompts = [entry["prompt"] for entry in batch_data]

        inputs = prepare_inputs(prompts, tokenizer).to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0
            )
            
        out_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        for j, out_text in enumerate(out_texts):
            out_obj = copy.deepcopy(batch_data[j])
            out_obj["out_text"] = out_text
            
            if test_lang=="hi":
                try:
                    out_obj["model_response"] = out_text.split("सही विकल्प:")[1]
                except IndexError:
                    out_obj["model_response"] = ""
            elif test_lang=="es":
                try:
                    out_obj["model_response"] = out_text.split("La opción correcta:")[1]
                except IndexError:
                    out_obj["model_response"] = ""
            elif test_lang=="ar":
                try:
                    out_obj["model_response"] = out_text.split("الخيار الصحيح:")[1]
                except IndexError:
                    out_obj["model_response"] = ""
            elif test_lang=="zh-cn":
                try:
                    out_obj["model_response"] = out_text.split("正确选项:")[1]
                except IndexError:
                    out_obj["model_response"] = ""
            elif test_lang=="en":
                try:
                    out_obj["model_response"] = out_text.split("The correct option:")[1]
                except IndexError:
                    out_obj["model_response"] = ""
            
            out_jsons.append(out_obj)

    with open(f"datav2/model_responses/cross/{src_lang}/{test_lang}/{modelname}/{inputfile}.json", "w", encoding="utf-8") as f:
        json.dump(out_jsons, f, indent=4, ensure_ascii=False)
    
    return out_jsons

def get_answer(model_response):
    """Extract A or B answer from model response."""
    for char in model_response:
        if char == "A":
            return "A"
        elif char == "B":
            return "B"
    return ""

def get_accuracy(row):
    """Calculate accuracy for a single row."""
    if row["model_answer"] == row["response"]:
        return 1    # chose correct
    elif row["model_answer"] in ["A", "B"]:
        return 0    # chose incorrect
    return -1       # invalid response

def calculate_and_save_accuracy(results, modelname, src_lang, test_lang, task, output_csv="performance_transfer_eval_results_v2.csv"):
    """Calculate accuracy and save to CSV."""
    df = pd.DataFrame(results)
    df['model_answer'] = df['model_response'].apply(get_answer)
    df['model_acc'] = df.apply(get_accuracy, axis=1)
    
    total_samples = len(df)
    correct_answers = (df["model_acc"] == 1).sum()
    incorrect_answers = (df["model_acc"] == 0).sum()
    invalid_answers = (df["model_acc"] == -1).sum()
    option_a_count = (df["model_answer"] == "A").sum()
    option_b_count = (df["model_answer"] == "B").sum()
    
    accuracy = correct_answers / total_samples if total_samples > 0 else 0
    
    print(f"Results for {modelname} on {src_lang} -> {test_lang} {task}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct: {correct_answers} ({correct_answers/total_samples*100:.1f}%)")
    print(f"  Incorrect: {incorrect_answers} ({incorrect_answers/total_samples*100:.1f}%)")
    print(f"  Invalid: {invalid_answers} ({invalid_answers/total_samples*100:.1f}%)")
    print(f"  Option A chosen: {option_a_count} ({option_a_count/total_samples*100:.1f}%)")
    print(f"  Option B chosen: {option_b_count} ({option_b_count/total_samples*100:.1f}%)")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Prepare CSV data
    csv_data = {
        'model_size': modelsize,
        'train_language': src_lang,
        'test_language': test_lang,
        'task': task,
        'total_samples': total_samples,
        'correct_answers': correct_answers,
        'incorrect_answers': incorrect_answers,
        'invalid_answers': invalid_answers,
        'option_a_chosen': option_a_count,
        'option_b_chosen': option_b_count,
        'accuracy': accuracy
    }
    
    # Save to CSV
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        new_df = pd.concat([existing_df, pd.DataFrame([csv_data])], ignore_index=True)
    else:
        new_df = pd.DataFrame([csv_data])
    
    new_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return accuracy


if __name__ == "__main__":
    base_model_name = f"models/meta-llama/Llama-3.2-{modelsize}"
    tasks = ["AB", "BC", "CA"]

    for src_lang in source_langs:
        print(f"\n{'='*60}")
        print(f"PROCESSING SOURCE LANGUAGE (TRAIN): {src_lang}")
        print(f"{'='*60}")
        
        for task in tasks:
            print(f"\nRunning task: {task}")
            modeltask = task
            inputdata = task
            
            # Use correct PEFT model path based on the task_vec patterns
            peft_model_path = f"models/v2/lora_llama_finetuned/{src_lang}/{modelsize}_{modeltask}"
            
            # Check if PEFT model exists, fallback to regular model loading if not
            if os.path.exists(peft_model_path):
                try:
                    model, tokenizer = load_peft_model(base_model_name, peft_model_path)
                    print(f"Loaded PEFT model from: {peft_model_path}")
                except Exception as e:
                    print(f"Failed to load PEFT model: {e}")
                    print("Falling back to regular model loading...")
                    # Fallback to old model path structure
                    modelpath = f"models/v2/llama_finetuned/{src_lang}/{modelsize}_{modeltask}"
                    from src.common.model_utils import load_model
                    model, tokenizer = load_model(modelpath, use_float16=True)
            else:
                print(f"PEFT model path does not exist: {peft_model_path}")
                print("Using regular model loading...")
                # Fallback to old model path structure  
                modelpath = f"models/v2/llama_finetuned/{src_lang}/{modelsize}_{modeltask}"
                from src.common.model_utils import load_model
                model, tokenizer = load_model(modelpath, use_float16=True)

            print(f"Model dtype: {model.parameters().__next__().dtype}")
            tokenizer.pad_token_id = tokenizer.eos_token_id

            for test_lang in langs:
                if test_lang != src_lang:  # Skip same language evaluation
                    print(f"\n{'='*40}")
                    print(f"Testing {src_lang} -> {test_lang}")
                    print(f"{'='*40}")
                    
                    input_file = f"datav2/prompt_response/{test_lang}/test/{inputdata}.json"
                    inputfile = inputdata
                    modelname = f"{modelsize}_{modeltask}_{src_lang}_to_{test_lang}"

                    print(f"Running model: {modelname}")
                    print(f"Input file: {input_file}")

                    savefilepath = f"datav2/model_responses/cross/{src_lang}/{test_lang}/{modelname}"
                    if not os.path.exists(savefilepath):
                        os.makedirs(savefilepath)

                    # Run inference and get results
                    results = run_inference(model, tokenizer, input_file, inputfile, modelname, test_lang, src_lang)

                    # Calculate accuracy and save to CSV
                    accuracy = calculate_and_save_accuracy(results, modelname, src_lang, test_lang, task)
                    
                    print(f"Completed {modelname} with accuracy: {accuracy:.4f}")

            # Clean up memory after processing all test languages for this model
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Completed task {task} for source language {src_lang}")

    print(f"\n{'='*60}")
    print("ALL CROSS-LINGUAL EVALUATIONS COMPLETED")
    print(f"Results saved to: performance_transfer_eval_results_v2.csv")
    print(f"{'='*60}")


