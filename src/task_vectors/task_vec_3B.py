import torch, copy, os, gc, json
from src.common.model_utils import load_model
import pandas as pd
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
import sys
from peft import PeftModel
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
import warnings
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings("ignore", category=FutureWarning)


torch.set_grad_enabled(False)

device = "cpu"
langs = ["en", "hi", "ar", "es", "zh-cn"]
# langs = ["zh-cn"]

size = "3B"
tasks = ["AB", "BC", "CA"]  # Updated to only include these three tasks
# tasks = ["AB", "BC"]
output_csv_path = f"task_vector_results_{size}_all_datav2.csv"

# GPU management
available_gpus = ["cuda:0", "cuda:1"]
gpu_locks = {gpu: threading.Lock() for gpu in available_gpus}
gpu_queue = queue.Queue()
for gpu in available_gpus:
    gpu_queue.put(gpu)


# -------- Hyperparameters --------
alpha_floor = 1e-3
eps = 1e-12
use_early_stopping = False

base_path = f"models/meta-llama/Llama-3.2-{size}"

# gamma_instrs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# gamma_prefs  = [0.5, 0.6, 0.7, 0.8, 0.9]
# gamma_instrs = [1.3]
# gamma_prefs  = [0.7]

def prepare_inputs(texts, tokenizer, max_length=512):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    return inputs

def infer_run_eval(model, tokenizer, experiment, modelname, input_file, inputdata, lang, gpu_device="cuda"):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not os.path.exists(f"datav2/model_test_generations/{experiment}/{lang}/{modelname}"):
        os.makedirs(f"datav2/model_test_generations/{experiment}/{lang}/{modelname}")

    with open(input_file, "r") as file:
        data = json.load(file)

    model.to(gpu_device)
    out_jsons = []
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size), disable=True):
        batch_data = data[i:i+batch_size]
        prompts = [entry["prompt"] for entry in batch_data]

        inputs = prepare_inputs(prompts, tokenizer).to(gpu_device)
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        out_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        for j, out_text in enumerate(out_texts):
            out_obj = copy.deepcopy(batch_data[j])
            out_obj["out_text"] = out_text
            # print(f"Generated text for prompt {j}:")
            # print(out_text)
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

# -------- CSV Logging Functions --------
def initialize_results_csv():
    """Initialize or load existing results CSV file"""
    csv_path = output_csv_path
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Create empty DataFrame with required columns
        columns = ['size', 'lang', 'task', 'gamma_instr', 'gamma_pref', 'dev_accuracy', 'test_accuracy']
        return pd.DataFrame(columns=columns)

def save_result_to_csv(size, lang, task, instruction_models, gamma_instr, gamma_pref, dev_accuracy, test_accuracy=None):
    """Save a single result to CSV file"""
    csv_path = output_csv_path
    
    # Load existing results
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        columns = ['size', 'lang', 'task', 'instruction_models', 'gamma_instr', 'gamma_pref', 'dev_accuracy', 'test_accuracy']
        df = pd.DataFrame(columns=columns)
    
    # Create new row
    new_row = {
        'size': size,
        'lang': lang,
        'task': task,
        'instruction_models': instruction_models,
        'gamma_instr': gamma_instr,
        'gamma_pref': gamma_pref,
        'dev_accuracy': dev_accuracy,
        'test_accuracy': test_accuracy
    }
    
    # Append new row and save
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def evaluate_gammas_threaded(gamma_instr, gamma_pref, base_model, tokenizer, delta_instr, pref_vector, task, instruction_models, lang, split='dev'):
    """Thread-safe version that gets a GPU from queue, evaluates, then releases GPU"""
    
    # Get available GPU
    gpu_device = gpu_queue.get()
    
    try:
        with gpu_locks[gpu_device]:  # Ensure exclusive access to this GPU
            start_time = time.time()  # Start timing after acquiring GPU lock
            print(f"Evaluating with gamma_instr={gamma_instr:.2f}, gamma_pref={gamma_pref:.2f} on {split} set using {gpu_device}")
            
            # Create model copy on CPU first
            new_model = copy.deepcopy(base_model)
            
            # Apply deltas on CPU
            with torch.no_grad():
                for (name, p_new), instr, pref in zip(
                        new_model.named_parameters(),
                        delta_instr.values(),
                        pref_vector.values()):
                    p_new.add_(gamma_instr * instr + gamma_pref * pref)

            modelname = f"{size}_rev{task}_{instruction_models}_gi{gamma_instr:.2f}_gp{gamma_pref:.2f}"
            experiment = f"task_vec_{size}_grid_search"
            inputdata = task[1] + task[0]
            input_file = f"datav2/prompt_response/{lang}/{split}/{inputdata}.json"
            
            # Run inference on assigned GPU
            infer_run_eval(new_model, tokenizer, experiment, modelname, input_file, f"{inputdata}_{split}", lang, gpu_device)

            # Clean up GPU memory
            del new_model
            torch.cuda.empty_cache()
            gc.collect()

            # Process results (CPU-bound work)
            filepath = f"datav2/model_test_generations/{experiment}/{lang}/{modelname}/{inputdata}_{split}.json"
            df = pd.read_json(filepath)
            df['model_answer'] = df.apply(get_answer, axis=1)
            df['model_acc'] = df.apply(get_accuracy, axis=1)
            
            accuracy = (df["model_acc"] == 1).sum()
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Accuracy on {split} set: {accuracy} (GPU: {gpu_device}, Time: {elapsed_time:.2f}s)")
            return accuracy
            
    finally:
        # Always return GPU to queue
        gpu_queue.put(gpu_device)



# def evaluate_gammas(gamma_instr, gamma_pref, base_model, tokenizer, delta_instr, pref_vector, task, instruction_models, lang, split='dev'):
#     print(f"Evaluating with gamma_instr={gamma_instr:.2f}, gamma_pref={gamma_pref:.2f} on {split} set")
#     new_model = copy.deepcopy(base_model)
#     with torch.no_grad():
#         for (name, p_new), instr, pref in zip(
#                 new_model.named_parameters(),
#                 delta_instr.values(),
#                 pref_vector.values()):
#             p_new.add_(gamma_instr * instr + gamma_pref * pref)

#     modelname = f"{size}_rev{task}_{instruction_models}_gi{gamma_instr:.2f}_gp{gamma_pref:.2f}"
#     experiment = f"task_vec_{size}_grid_search"
#     inputdata = task[1] + task[0]
#     input_file = f"datav2/prompt_response/{lang}/{split}/{inputdata}.json"
    
#     infer_run_eval(new_model, tokenizer, experiment, modelname, input_file, f"{inputdata}_{split}", lang)

#     del new_model
#     gc.collect()

#     filepath = f"datav2/model_test_generations/{experiment}/{lang}/{modelname}/{inputdata}_{split}.json"
#     df = pd.read_json(filepath)
#     df['model_answer'] = df.apply(get_answer, axis=1)
#     df['model_acc'] = df.apply(get_accuracy, axis=1)
    
#     accuracy = (df["model_acc"] == 1).sum()
#     print(f"Accuracy on {split} set: {accuracy}")
#     return accuracy

def load_peft_model(peft_path, base_model_path):
    bm = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map=device)
    m1 = PeftModel.from_pretrained(bm, peft_path)
    m1 = m1.merge_and_unload()
    m1 = m1.to(torch.float16)
    m1.eval()
    return m1


# -------- Main Loop --------
for lang in langs:
    print(f"\n{'='*60}")
    print(f"PROCESSING LANGUAGE: {lang}")
    print(f"{'='*60}")
    
    # Define paths for current language
    bc_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_BC"
    cb_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_CB"
    ab_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_AB"
    ba_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_BA"
    ca_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_CA"
    ac_path = f"models/v2/lora_llama_finetuned/{lang}/{size}_AC"
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"PROCESSING TASK: {task} (Language: {lang})")
        print(f"{'='*50}")
        
        # Determine which instruction models to use for this task
        if task == "AB":
            instruction_models_list = ["BC_CB", "CA_AC"]  # Use CA_AC for task AB
        elif task == "BC":
            instruction_models_list = ["CA_AC", "AB_BA"]  # Use CA_AC for task BC
        elif task == "CA":
            instruction_models_list = ["AB_BA", "BC_CB"]  # Use both for CA
    
        for instruction_models in instruction_models_list:
            print(f"\n{'-'*30}")
            print(f"Task: {task}, Instruction Models: {instruction_models}")
            print(f"{'-'*30}")
            
            # -------- 1. Load base model and precompute vectors --------
            print("Loading base model...")
            base_model_path = f"models/meta-llama/Llama-3.2-{size}"
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
            tokenizer.pad_token_id = tokenizer.eos_token_id

            # --- Precompute instruction vector
            print(f"Precomputing instruction vector using {instruction_models}...")
            delta_instr = {k: torch.zeros_like(p) for k, p in base_model.named_parameters()}

            def accumulate_instruction(delta_instr, peft_path, weight=0.5):
                ft_model  = load_peft_model(peft_path, base_model_path)
                with torch.no_grad():
                    for (name, p_base), p_ft in zip(base_model.named_parameters(),
                                                    ft_model.parameters()):
                        delta_instr[name].add_(weight * (p_ft - p_base))
                del ft_model
                gc.collect()

            if instruction_models == "BC_CB":
                accumulate_instruction(delta_instr, bc_path)
                print("Instruction vector accumulated: BC [1/2]")
                accumulate_instruction(delta_instr, cb_path)
                print("Instruction vector accumulated: CB [2/2]")
            elif instruction_models == "AB_BA":
                accumulate_instruction(delta_instr, ab_path)
                print("Instruction vector accumulated: AB [1/2]")
                accumulate_instruction(delta_instr, ba_path)
                print("Instruction vector accumulated: BA [2/2]")
            elif instruction_models == "CA_AC":
                accumulate_instruction(delta_instr, ca_path)
                print("Instruction vector accumulated: CA [1/2]")
                accumulate_instruction(delta_instr, ac_path)
                print("Instruction vector accumulated: AC [2/2]")
            else:
                raise ValueError(f"Invalid instruction_models: {instruction_models}. Choose 'BC_CB', 'AB_BA', or 'CA_AC'")

            # --- Precompute preference vector (flipped & orthogonalized)
            print(f"Computing preference vector for task {task}...")
            if task == "AB":
                pref_model = load_peft_model(ab_path, base_model_path)
            elif task == "BC":
                pref_model = load_peft_model(bc_path, base_model_path)
            elif task == "CA":
                pref_model = load_peft_model(ca_path, base_model_path)
            else:
                raise ValueError(f"Invalid task: {task}. Choose from AB, BC, CA")

            print("Computing preference vector...")
            with torch.no_grad():
                # 1. Flatten the instruction delta once (fp32 for accuracy)
                vec_instr = parameters_to_vector(
                    [p.float() for p in delta_instr.values()]
                )
                # instr_norm_sq = vec_instr.dot(vec_instr) + eps          # scalar
                instr_norm_sq = torch.sum(vec_instr * vec_instr) + eps  # scalar

                # 2. Flatten the raw preference delta once
                vec_pref = parameters_to_vector([
                    (p_pref - p_base).float()
                    for p_pref, p_base in zip(pref_model.parameters(),
                                              base_model.parameters())
                ])

                # 3. Single global projection coefficient α
                alpha = torch.clamp(
                    torch.sum(vec_pref * vec_instr) / instr_norm_sq,
                    min = -1 / alpha_floor,
                    max =  1 / alpha_floor
                )

                # 4. Orthogonalise and flip (A>B  →  B>A)
                vec_pref_orth = -(vec_pref - alpha * vec_instr)

                # 5. Convert back into a {name: tensor} dict (keep fp16 to save RAM)
                pref_tensors = [torch.empty_like(p)            # create writable shells
                                for p in base_model.parameters()]
                vector_to_parameters(vec_pref_orth.to(torch.float16), pref_tensors)  # in‑place

                pref_vector = {
                    name: tensor
                    for (name, _), tensor in zip(base_model.named_parameters(),
                                                pref_tensors)
                }


            print("Preference vector computed.")
            del pref_model, vec_instr, vec_pref, vec_pref_orth, pref_tensors
            gc.collect()

            # -------- 2. Grid search efficiently --------
            best_accuracy = -1
            best_gamma_instr = -1
            best_gamma_pref = -1

            # Initialize CSV
            print("Initializing results CSV...")
            results_df = initialize_results_csv()

            # Coarse Grid Search
            print("\n--- Starting Coarse Grid Search ---")
            coarse_gamma_instrs = np.arange(0.3, 3.1, 0.3)
            coarse_gamma_prefs = np.arange(0.3, 2.1, 0.3)

            # Create all parameter combinations
            param_combinations = []
            
            for gamma_instr in coarse_gamma_instrs:
                # prev_accuracy = -1
                for gamma_pref in coarse_gamma_prefs:
                    param_combinations.append((gamma_instr, gamma_pref))

            # Use ThreadPoolExecutor with max 2 workers (one per GPU)
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit all tasks
                future_to_params = {
                    executor.submit(
                        evaluate_gammas_threaded, 
                        gamma_instr, gamma_pref, base_model, tokenizer, 
                        delta_instr, pref_vector, task, instruction_models, lang
                    ): (gamma_instr, gamma_pref)
                    for gamma_instr, gamma_pref in param_combinations
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    gamma_instr, gamma_pref = future_to_params[future]
                    try:
                        accuracy = future.result()
                        
                        # Save result to CSV immediately
                        save_result_to_csv(size, lang, task, instruction_models, gamma_instr, gamma_pref, accuracy)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_gamma_instr = gamma_instr
                            best_gamma_pref = gamma_pref
                            
                    except Exception as exc:
                        print(f'Evaluation for gamma_instr={gamma_instr}, gamma_pref={gamma_pref} generated an exception: {exc}')

                    # accuracy = evaluate_gammas(gamma_instr, gamma_pref, base_model, tokenizer, delta_instr, pref_vector, task, instruction_models, lang)
                    
                    # # Save result to CSV immediately
                    # save_result_to_csv(size, lang, task, instruction_models, gamma_instr, gamma_pref, accuracy)
                    
                    # if accuracy > best_accuracy:
                    #     best_accuracy = accuracy
                    #     best_gamma_instr = gamma_instr
                    #     best_gamma_pref = gamma_pref
                    
                    # # Early stopping: if accuracy decreases and is below 280, move to next gamma_instr
                    # if use_early_stopping and prev_accuracy != -1 and accuracy < prev_accuracy and accuracy < 300:
                    #     print(f"Early stopping for gamma_instr={gamma_instr:.2f} at gamma_pref={gamma_pref:.2f} (accuracy={accuracy} < prev={prev_accuracy})")
                    #     break
                    
                    # prev_accuracy = accuracy

            print(f"\n--- Coarse Search Complete ---")
            print(f"Best accuracy: {best_accuracy}")
            print(f"Best gamma_instr: {best_gamma_instr:.2f}")
            print(f"Best gamma_pref: {best_gamma_pref:.2f}")


            # Fine Grid Search
            print("\n--- Starting Fine Grid Search ---")
            fine_gamma_instrs = np.arange(best_gamma_instr - 0.2, best_gamma_instr + 0.21, 0.1)
            fine_gamma_prefs = np.arange(best_gamma_pref - 0.2, best_gamma_pref + 0.21, 0.1)

            fine_param_combinations = []
            for gamma_instr in fine_gamma_instrs:
                for gamma_pref in fine_gamma_prefs:
                    # Skip re-evaluating the best point from the coarse search
                    if not (np.isclose(gamma_instr, best_gamma_instr) and np.isclose(gamma_pref, best_gamma_pref)):
                        fine_param_combinations.append((gamma_instr, gamma_pref))

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_params = {
                    executor.submit(
                        evaluate_gammas_threaded,
                        gamma_instr, gamma_pref, base_model, tokenizer,
                        delta_instr, pref_vector, task, instruction_models, lang
                    ): (gamma_instr, gamma_pref)
                    for gamma_instr, gamma_pref in fine_param_combinations
                }
                
                for future in as_completed(future_to_params):
                    gamma_instr, gamma_pref = future_to_params[future]
                    try:
                        accuracy = future.result()
                        
                        save_result_to_csv(size, lang, task, instruction_models, gamma_instr, gamma_pref, accuracy)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_gamma_instr = gamma_instr
                            best_gamma_pref = gamma_pref
                            
                    except Exception as exc:
                        print(f'Fine evaluation for gamma_instr={gamma_instr}, gamma_pref={gamma_pref} generated an exception: {exc}')
                    
                    # # Save result to CSV immediately
                    # save_result_to_csv(size, lang, task, instruction_models, gamma_instr, gamma_pref, accuracy)
                    
                    # if accuracy > best_accuracy:
                    #     best_accuracy = accuracy
                    #     best_gamma_instr = gamma_instr
                    #     best_gamma_pref = gamma_pref

            print(f"\n--- Fine Search Complete ---")
            print(f"Final best accuracy on dev set: {best_accuracy}")
            print(f"Final best gamma_instr: {best_gamma_instr:.2f}")
            print(f"Final best gamma_pref: {best_gamma_pref:.2f}")


            # -------- 3. Final Evaluation on Test Set --------
            print("\n--- Evaluating on Test Set with Best Gammas ---")
            test_accuracy = evaluate_gammas_threaded(best_gamma_instr, best_gamma_pref, base_model, tokenizer, delta_instr, pref_vector, task, instruction_models, lang, split='test')

            # Update the CSV with test accuracy for the best parameters
            csv_path = output_csv_path
            df = pd.read_csv(csv_path)
            # Find the row with best gamma values and update test_accuracy
            mask = (df['size'] == size) & (df['lang'] == lang) & (df['task'] == task) & \
                   (df['instruction_models'] == instruction_models) & \
                   (np.isclose(df['gamma_instr'], best_gamma_instr)) & \
                   (np.isclose(df['gamma_pref'], best_gamma_pref))
            df.loc[mask, 'test_accuracy'] = test_accuracy
            df.to_csv(csv_path, index=False)

            print(f"\n--- Test Set Evaluation Complete ---")
            print(f"Task: {task}, Instruction Models: {instruction_models}")
            print(f"Best gamma_instr: {best_gamma_instr:.2f}")
            print(f"Best gamma_pref: {best_gamma_pref:.2f}")
            print(f"Final accuracy on test set: {test_accuracy}")
            
            # Clean up models before next iteration
            del base_model, delta_instr, pref_vector
            gc.collect()

print(f"\n{'='*60}")
print("ALL LANGUAGES AND TASKS COMPLETED")
print(f"{'='*60}")
