import os
import random
import json
from sklearn.model_selection import train_test_split

from src.data_preparation.io_utils import load_json_or_jsonl


def create_prompt_and_response(entry):
    """
    Creates a prompt and response from a given JSON object in the specified format for the Instruct model.
    Args:
        entry (dict): JSON object with keys "story", "question", and "options".
    Returns:
        dict: A dictionary with keys "prompt" and "response".
    """
    index = entry["index"]
    story = entry["story"]
    question = entry["question"]
    options = entry["options"]

    # Randomize order of options
    options_order = random.sample(range(2), 2)
    options_text = [
        f"A. {options[options_order[0]]}",
        f"B. {options[options_order[1]]}"
    ]

    # Construct the prompt
    prompt = (
        f"Here is a situation that needs to be analysed. The story:\n"
        f"{story}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Options:\n"
        f"{options_text[0]}\n{options_text[1]}\n\n"
        f"The correct option: "
    )

    # True response corresponds to the first option in the JSON object
    response_AB = "A" if options_order[0] == 0 else "B"
    response_BA = "B" if options_order[0] == 0 else "A"

    swap = options_order[0] == 1

    return {"index": index, "swapped": swap, "prompt": prompt, "response": response_AB}, {"index": index, "swapped": swap, "prompt": prompt, "response": response_BA}

def load_data_from_txt_files(txt_files):
    """
    Loads and combines data from multiple .txt files containing JSON arrays.

    Args:
        txt_files (list): List of file paths to .txt files.

    Returns:
        list: List of dictionaries with "prompt" and "response".
    """
    combined_data = []
    for file in txt_files:
        json_objects = json_from_file(file)
        for entry in json_objects:
            if "story" in entry and "question" in entry and "options" in entry:
                combined_data.append(create_prompt_and_response(entry))
    return combined_data


def combine_data_from_txt_files(txt_files):
    """
    Loads and combines data from multiple .txt files containing JSON arrays.

    Args:
        txt_files (list): List of file paths to .txt files.
    """
    combined_data = []
    for file in txt_files:
        json_objects = json_from_file(file)
        for entry in json_objects:
            entry["index"] = file.split("/")[-1][:-4]+"_i"+str(entry["index"])
            if "story" in entry and "question" in entry and "options" in entry:
                combined_data.append(entry)
    return combined_data

def json_from_file(filepath):
    """
    Extracts a list of JSON objects from the specific file format.

    Args:
        filepath (str): Path to the file.

    Returns:
        list: List of dictionaries parsed from the file.
    """
    with open(filepath, 'r') as file:
        raw = file.read()
        raw_json = raw[8:-4]  # Adjust slicing to your specific file format
        data = json.loads(raw_json)
    return data

def get_txt_files_from_folder(folder_path):
    """
    Gets all files from the given folder.

    Args:
        folder_path (str): Path to the folder containing .jsonl files.

    Returns:
        list: List of file paths to all files.
    """
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path)]


if __name__ == "__main__":
    # divide in train/test and put in prompt response format

    for lang in ["en"]:
        for pair in ["AB", "BC", "CA"]:
            if lang == "en":
                input_file = f"datav2/originalAB/{pair}.json"
            else:
                input_file = f"datav2/translated_responses/{lang}/{pair}.json"
            filename1 = pair
            filename2 = pair[::-1]

            data = load_json_or_jsonl(input_file)
            
            train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

            os.makedirs(f"datav2/prompt_response/{lang}/train", exist_ok=True)
            os.makedirs(f"datav2/prompt_response/{lang}/test", exist_ok=True)

            # AB, BA
            prompts_jsonAB = []
            prompts_jsonBA = []
            random.seed(42)
            for train_data_entry in train_data:
                prompt_jsonAB, prompt_jsonBA = create_prompt_and_response(train_data_entry)
                prompts_jsonAB.append(prompt_jsonAB)
                prompts_jsonBA.append(prompt_jsonBA)

            with open(f"datav2/prompt_response/{lang}/train/{filename1}.json", "w") as f:
                json.dump(prompts_jsonAB, f, indent=4)

            with open(f"datav2/prompt_response/{lang}/train/{filename2}.json", "w") as f:
                json.dump(prompts_jsonBA, f, indent=4)

            random.seed(42)
            prompts_jsonAB = []
            prompts_jsonBA = []
            for test_data_entry in test_data:
                prompt_jsonAB, prompt_jsonBA = create_prompt_and_response(test_data_entry)
                prompts_jsonAB.append(prompt_jsonAB)
                prompts_jsonBA.append(prompt_jsonBA)

            with open(f"datav2/prompt_response/{lang}/test/{filename1}.json", "w") as f:
                json.dump(prompts_jsonAB, f, indent=4)

            with open(f"datav2/prompt_response/{lang}/test/{filename2}.json", "w") as f:
                json.dump(prompts_jsonBA, f, indent=4)
