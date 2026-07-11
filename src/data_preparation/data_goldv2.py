import os
import random
import json

from src.data_preparation.io_utils import load_json_or_jsonl


PROMPT_TEMPLATES = {
    "en": (
        "Here is a situation that needs to be analysed. The story:\n"
        "{story}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Options:\n"
        "{opt_a}\n{opt_b}\n\n"
        "The correct option: "
    ),
    "ar": (
        "هنا موقف يحتاج إلى التحليل. القصة:\n"
        "{story}\n\n"
        "سؤال:\n"
        "{question}\n\n"
        "خيارات:\n"
        "{opt_a}\n{opt_b}\n\n"
        "الخيار الصحيح: "
    ),
    "es": (
        "Aquí hay una situación que necesita ser analizada. La historia:\n"
        "{story}\n\n"
        "Pregunta:\n"
        "{question}\n\n"
        "Opciones:\n"
        "{opt_a}\n{opt_b}\n\n"
        "La opción correcta: "
    ),
    "hi": (
        "यहाँ एक स्थिति है जिसे विश्लेषण करने की आवश्यकता है। कहानी इस प्रकार है:\n"
        "{story}\n\n"
        "प्रश्न:\n"
        "{question}\n\n"
        "विकल्प:\n"
        "{opt_a}\n{opt_b}\n\n"
        "सही विकल्प: "
    ),
    "zh-cn": (
        "这里有一个需要分析的情境。故事如下：\n"
        "{story}\n\n"
        "问题:\n"
        "{question}\n\n"
        "选项:\n"
        "{opt_a}\n{opt_b}\n\n"
        "正确选项: "
    ),
}


def create_prompt_and_response(entry, lang):
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

    template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])
    prompt = template.format(
        story=story,
        question=question,
        opt_a=options_text[0],
        opt_b=options_text[1],
    )

    # True response corresponds to the first option in the JSON object
    response_AB = "A" if options_order[0] == 0 else "B"
    response_BA = "B" if options_order[0] == 0 else "A"

    swap = options_order[0] == 1

    return {"index": index, "swapped": swap, "prompt": prompt, "response": response_AB}, {"index": index, "swapped": swap, "prompt": prompt, "response": response_BA}


def load_json_entries(filepath, is_jsonl):
    return load_json_or_jsonl(filepath)


def get_translated_languages(translated_root):
    if not os.path.isdir(translated_root):
        return []
    return sorted(
        name for name in os.listdir(translated_root)
        if os.path.isdir(os.path.join(translated_root, name))
    )


if __name__ == "__main__":
    pairs = ["AB", "BC", "CA"]
    original_dir = "goldv2/original/en"
    translated_dir = "goldv2/translated_responses"
    output_dir = "goldv2/prompt_response"

    languages = ["en"] + get_translated_languages(translated_dir)
    seen = set()
    languages = [lang for lang in languages if not (lang in seen or seen.add(lang))]

    for lang in languages:
        for pair in pairs:
            if lang == "en":
                input_file = os.path.join(original_dir, f"{pair}.json")
                is_jsonl = False
            else:
                input_file = os.path.join(translated_dir, lang, f"{pair}.json")
                is_jsonl = True

            if not os.path.exists(input_file):
                continue

            data = load_json_entries(input_file, is_jsonl)

            filename1 = pair
            filename2 = pair[::-1]
            output_lang_dir = os.path.join(output_dir, lang, "test")
            os.makedirs(output_lang_dir, exist_ok=True)

            prompts_jsonAB = []
            prompts_jsonBA = []
            random.seed(42)
            for entry in data:
                prompt_jsonAB, prompt_jsonBA = create_prompt_and_response(entry, lang)
                prompts_jsonAB.append(prompt_jsonAB)
                prompts_jsonBA.append(prompt_jsonBA)

            with open(os.path.join(output_lang_dir, f"{filename1}.json"), "w", encoding="utf-8") as f:
                json.dump(prompts_jsonAB, f, indent=4, ensure_ascii=False)

            with open(os.path.join(output_lang_dir, f"{filename2}.json"), "w", encoding="utf-8") as f:
                json.dump(prompts_jsonBA, f, indent=4, ensure_ascii=False)
