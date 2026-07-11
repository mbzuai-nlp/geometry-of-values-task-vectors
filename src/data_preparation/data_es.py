import os
import random
import json
from sklearn.model_selection import train_test_split

from src.data_preparation.io_utils import load_json_or_jsonl

lang = "es"

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
        f"Aquí hay una situación que necesita ser analizada. La historia:\n"
        f"{story}\n\n"
        f"Pregunta:\n"
        f"{question}\n\n"
        f"Opciones:\n"
        f"{options_text[0]}\n{options_text[1]}\n\n"
        f"La opción correcta: "
    )

    # True response corresponds to the first option in the JSON object
    response_AB = "A" if options_order[0] == 0 else "B"
    response_BA = "B" if options_order[0] == 0 else "A"

    swap = options_order[0] == 1

    return {"index": index, "swapped": swap, "prompt": prompt, "response": response_AB}, {"index": index, "swapped": swap, "prompt": prompt, "response": response_BA}



if __name__ == "__main__":
    # divide in train/test and put in prompt response format
    # input_file = "data/originalAB/AB.json"
    # filename1 = "AB"
    # filename2 = "BA"

    lang = "es"
    for pair in ["AB", "BC", "CA"]:
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

        with open(f"datav2/prompt_response/{lang}/train/{filename1}.json", "w", encoding="utf-8") as f:
            json.dump(prompts_jsonAB, f, indent=4, ensure_ascii=False)

        with open(f"datav2/prompt_response/{lang}/train/{filename2}.json", "w", encoding="utf-8") as f:
            json.dump(prompts_jsonBA, f, indent=4, ensure_ascii=False)

        random.seed(42)
        prompts_jsonAB = []
        prompts_jsonBA = []
        for test_data_entry in test_data:
            prompt_jsonAB, prompt_jsonBA = create_prompt_and_response(test_data_entry)
            prompts_jsonAB.append(prompt_jsonAB)
            prompts_jsonBA.append(prompt_jsonBA)

        with open(f"datav2/prompt_response/{lang}/test/{filename1}.json", "w", encoding="utf-8") as f:
            json.dump(prompts_jsonAB, f, indent=4, ensure_ascii=False)

        with open(f"datav2/prompt_response/{lang}/test/{filename2}.json", "w", encoding="utf-8") as f:
            json.dump(prompts_jsonBA, f, indent=4, ensure_ascii=False)
