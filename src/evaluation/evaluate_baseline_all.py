from src.common.model_utils import *
import json
import copy
import os
import argparse
import csv
from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore') # setting ignore as a parameter
# warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:None for open-end generation")

lang = "en"
DEFAULT_CSV_PATH = "baseline_results_datav2.csv"
CSV_HEADERS = [
    "model_name",
    "model_size",
    "language",
    "task",
    "total_samples",
    "correct_answers",
    "incorrect_answers",
    "invalid_answers",
    "option_a_chosen",
    "option_b_chosen",
    "accuracy",
]

def initialize_results_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)

def save_result_to_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def get_answer(model_response):
    for char in model_response:
        if char == "A":
            return "A"
        if char == "B":
            return "B"
    return ""

def summarize_results(out_jsons):
    total_samples = len(out_jsons)
    correct_answers = 0
    incorrect_answers = 0
    invalid_answers = 0
    option_a_chosen = 0
    option_b_chosen = 0

    for entry in out_jsons:
        model_answer = get_answer(entry.get("model_response", ""))
        if model_answer == "A":
            option_a_chosen += 1
        elif model_answer == "B":
            option_b_chosen += 1

        if model_answer == entry.get("response"):
            correct_answers += 1
        elif model_answer in ["A", "B"]:
            incorrect_answers += 1
        else:
            invalid_answers += 1

    accuracy = correct_answers / total_samples if total_samples else 0.0
    return {
        "total_samples": total_samples,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "invalid_answers": invalid_answers,
        "option_a_chosen": option_a_chosen,
        "option_b_chosen": option_b_chosen,
        "accuracy": accuracy,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline models across languages.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Output CSV path.")
    return parser.parse_args()

def prepare_inputs(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='pt')
    return inputs

def run_inference(model, tokenizer, input_file, inputfile, modelname, lang, write_csv=False, csv_path=None):
    with open(input_file, "r") as file:
        data = json.load(file)

    out_jsons = []
    for entry in tqdm(data):
        out_obj = copy.deepcopy(entry)

        inputs = prepare_inputs([entry["prompt"]], tokenizer).to('cuda')
        output = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        out_text = tokenizer.decode(output[0], skip_special_tokens=False)
        
        out_obj["out_text"] = out_text
        # print(out_text)
        if lang == "hi":
            out_obj["model_response"] = out_text.split("सही विकल्प:")[1]
        if lang == "es":
            out_obj["model_response"] = out_text.split("La opción correcta:")[1]
        if lang == "ar":
            out_obj["model_response"] = out_text.split("الخيار الصحيح:")[1]
        if lang == "zh-cn":
            out_obj["model_response"] = out_text.split("正确选项:")[1]
        if lang == "en":
            out_obj["model_response"] = out_text.split("The correct option:")[1]
        out_jsons.append(out_obj)

    with open(f"datav2/model_responses/{lang}/{modelname}/{inputfile}.json", "w", encoding="utf-8") as f:
        json.dump(out_jsons, f, indent=4, ensure_ascii=False)

    if write_csv:
        results = summarize_results(out_jsons)
        model_size = modelname.split("-")[-1]
        row = [
            modelname,
            model_size,
            lang,
            inputfile,
            results["total_samples"],
            results["correct_answers"],
            results["incorrect_answers"],
            results["invalid_answers"],
            results["option_a_chosen"],
            results["option_b_chosen"],
            results["accuracy"],
        ]
        save_result_to_csv(csv_path, row)
        print(f"Results saved to {csv_path}")

def run_inference_all(lang, write_csv=False, csv_path=None):
    for modelpath in ["models/meta-llama/Llama-3.2-1B", "models/meta-llama/Llama-3.2-3B"]:
        model, tokenizer = load_model(modelpath)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        for input_file in [f"datav2/prompt_response/{lang}/test/AB.json", f"datav2/prompt_response/{lang}/test/BC.json", f"datav2/prompt_response/{lang}/test/CA.json"]:
            inputfile = input_file.split("/")[-1][:2]
            modelname = modelpath.split("/")[-1]

            print(modelname, lang, inputfile)
            if not os.path.exists(f"datav2/model_responses/{lang}/{modelname}"):
                os.makedirs(f"datav2/model_responses/{lang}/{modelname}")
            run_inference(model, tokenizer, input_file, inputfile, modelname, lang, write_csv, csv_path)


if __name__ == "__main__":
    # continue
    langs = ["en", "hi", "es", "ar", "zh-cn"]

    args = parse_args()
    # if args.write_csv:
    initialize_results_csv(args.csv_path)

    for lang in langs:
        run_inference_all(lang, write_csv=True, csv_path=args.csv_path)
