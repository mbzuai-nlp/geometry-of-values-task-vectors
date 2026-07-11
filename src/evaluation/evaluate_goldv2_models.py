import argparse
import copy
import csv
import gc
import json
import os

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CSV_HEADERS = [
    "model_size",
    "language",
    "task",
    "accuracy",
    "correct_count",
    "incorrect_count",
    "option_a_count",
    "option_b_count",
]


def get_languages(prompt_root):
    if not os.path.isdir(prompt_root):
        return []
    return sorted(
        name for name in os.listdir(prompt_root)
        if os.path.isdir(os.path.join(prompt_root, name))
    )


def get_tasks_for_language(prompt_root, lang):
    test_dir = os.path.join(prompt_root, lang, "test")
    if not os.path.isdir(test_dir):
        return []
    tasks = []
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(".json"):
            tasks.append(os.path.splitext(filename)[0])
    return tasks


def get_models_for_language(lora_root, lang):
    lang_dir = os.path.join(lora_root, lang)
    if not os.path.isdir(lang_dir):
        return {}

    models = {}
    for name in sorted(os.listdir(lang_dir)):
        full_path = os.path.join(lang_dir, name)
        if not os.path.isdir(full_path) or "_" not in name:
            continue
        size, task = name.split("_", 1)
        models.setdefault(task, []).append((size, full_path))

    for task in models:
        models[task] = sorted(models[task], key=lambda item: item[0])

    return models


def prepare_inputs(texts, tokenizer):
    return tokenizer(texts, return_tensors="pt")


def get_answer(row):
    for char in row["model_response"]:
        if char == "A":
            return "A"
        if char == "B":
            return "B"
    return ""


def get_accuracy(row):
    if row["model_answer"] == row["response"]:
        return 1
    if row["model_answer"] in ["A", "B"]:
        return 0
    return -1


def infer_run_eval(model, tokenizer, experiment, modelname, input_file, inputdata, lang, output_root, max_new_tokens, device):
    tokenizer.pad_token_id = tokenizer.eos_token_id

    output_dir = os.path.join(output_root, experiment, lang, modelname)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    model.to(device)
    model.eval()

    out_jsons = []
    for entry in tqdm(data, desc=f"{lang} {modelname} {inputdata}"):
        out_obj = copy.deepcopy(entry)

        inputs = prepare_inputs([entry["prompt"]], tokenizer).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        out_text = tokenizer.decode(output[0], skip_special_tokens=False)

        out_obj["out_text"] = out_text
        if lang == "en":
            marker = "The correct option:"
        elif lang == "hi":
            marker = "सही विकल्प:"
        elif lang == "es":
            marker = "La opción correcta:"
        elif lang == "ar":
            marker = "الخيار الصحيح:"
        elif lang == "zh-cn":
            marker = "正确选项:"
        else:
            marker = "The correct option:"

        if marker in out_text:
            out_obj["model_response"] = out_text.split(marker)[1]
        else:
            out_obj["model_response"] = ""

        out_jsons.append(out_obj)

    output_path = os.path.join(output_dir, f"{inputdata}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_jsons, f, indent=4, ensure_ascii=False)


def save_results_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(CSV_HEADERS)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned models on goldv2 prompt_response test sets.")
    parser.add_argument("--prompt-root", default="goldv2/prompt_response")
    parser.add_argument("--lora-root", default="models/v2/lora_llama_finetuned")
    parser.add_argument("--base-root", default="models/meta-llama")
    parser.add_argument("--output", default="goldv2_eval_results.csv")
    parser.add_argument("--output-root", default="goldv2/model_test_generations")
    parser.add_argument("--experiment", default="llama_finetuned")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    languages = get_languages(args.prompt_root)
    if not languages:
        raise SystemExit(f"No languages found under {args.prompt_root}")

    for lang in languages:
        tasks = get_tasks_for_language(args.prompt_root, lang)
        if not tasks:
            continue

        models_by_task = get_models_for_language(args.lora_root, lang)
        if not models_by_task:
            print(f"No models found for language: {lang}")
            continue

        for task in tasks:
            if task not in models_by_task:
                print(f"No model found for {lang} task {task}")
                continue

            input_file = os.path.join(args.prompt_root, lang, "test", f"{task}.json")

            for size, model_path in models_by_task[task]:
                base_model_path = os.path.join(args.base_root, f"Llama-3.2-{size}")
                dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=dtype,
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()

                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                modelname = f"{size}_{task}"
                infer_run_eval(
                    model,
                    tokenizer,
                    args.experiment,
                    modelname,
                    input_file,
                    task,
                    lang,
                    args.output_root,
                    args.max_new_tokens,
                    args.device,
                )

                filepath = os.path.join(args.output_root, args.experiment, lang, modelname, f"{task}.json")
                df = pd.read_json(filepath)
                df["model_answer"] = df.apply(get_answer, axis=1)
                df["model_acc"] = df.apply(get_accuracy, axis=1)

                correct_count = (df["model_acc"] == 1).sum()
                incorrect_count = (df["model_acc"] == 0).sum()
                option_a_count = (df["model_answer"] == "A").sum()
                option_b_count = (df["model_answer"] == "B").sum()
                total_samples = len(df)
                accuracy = correct_count / total_samples if total_samples > 0 else 0

                save_results_to_csv(
                    args.output,
                    [
                        size,
                        lang,
                        task,
                        round(accuracy, 4),
                        int(correct_count),
                        int(incorrect_count),
                        int(option_a_count),
                        int(option_b_count),
                    ],
                )

                del model, base_model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
