import os
import json
import jsonlines
import asyncio
from googletrans import Translator
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def translate_text(text, translator, dest='en'):
    translated = await translator.translate(text, dest=dest)
    return translated.text

async def safe_translate(text, translator, sem, dest='en'):
    async with sem:
        try:
            return await translate_text(text, translator, dest=dest)
        except RetryError:
            return "ERROR"

async def translate_jsonl_file(input_file, output_file, translator, sem, dest='en', previous_file=None):
    print(f"Processing {input_file}")
    
    # Load the previous translation file if it exists
    previous_data = {}
    if previous_file and os.path.exists(previous_file):
        try:
            # Read the previous file properly, handling JSON objects correctly
            with open(previous_file, "r") as prev_reader:
                content = prev_reader.read()
                # Convert the content to a list of individual JSON objects
                # This handles both JSON arrays and individual JSON objects
                if content.strip().startswith('['):
                    previous_items = json.loads(content)
                else:
                    # Handle individual JSON objects (one per line)
                    previous_items = []
                    for line in content.strip().split('\n'):
                        if line.strip():
                            try:
                                previous_items.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse line as JSON: {line[:50]}...")
                
                previous_data = {item.get('index', i): item for i, item in enumerate(previous_items)}
            print(f"Found previous translation file with {len(previous_data)} items")
        except Exception as e:
            print(f"Error loading previous file: {e}")
    
    with open(input_file, "r") as reader, jsonlines.open(output_file, mode='w') as writer:
        data = json.load(reader)

        async def translate_entry(idx, obj):
            obj_index = obj.get('index', idx)
            if obj_index in previous_data:
                return previous_data[obj_index]

            translated_obj = {}
            tasks = {}

            if "story" in obj:
                tasks["story"] = asyncio.create_task(safe_translate(obj["story"], translator, sem, dest=dest))
            if "question" in obj:
                tasks["question"] = asyncio.create_task(safe_translate(obj["question"], translator, sem, dest=dest))
            if "options" in obj and isinstance(obj["options"], list):
                tasks["options"] = [
                    asyncio.create_task(safe_translate(opt, translator, sem, dest=dest))
                    for opt in obj["options"]
                ]

            for key, value in obj.items():
                if key not in ["story", "question", "options"]:
                    translated_obj[key] = value

            if "story" in tasks:
                translated_obj["story"] = await tasks["story"]
            if "question" in tasks:
                translated_obj["question"] = await tasks["question"]
            if "options" in tasks:
                translated_obj["options"] = [await t for t in tasks["options"]]

            return translated_obj

        coros = [translate_entry(i, obj) for i, obj in enumerate(data)]
        translated_results = []
        with tqdm(total=len(coros), desc=f"Translating {os.path.basename(input_file)}") as pbar:
            for coro in asyncio.as_completed(coros):
                translated_results.append(await coro)
                pbar.update(1)

        for result in tqdm(translated_results, desc=f'Writing {os.path.basename(input_file)}'):
            writer.write(result)

async def main():
    input_dir = 'goldv2/original/en/'
    output_dir = 'goldv2/translated_responses'
    DEST_LANG = 'zh-cn'
    output_dir = os.path.join(output_dir, DEST_LANG)
    os.makedirs(output_dir, exist_ok=True)
    
    translator = Translator()
    sem = asyncio.Semaphore(20)  # adjust concurrency as needed
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Check for a previous incomplete translation file
                previous_file = output_file
                # previous_file = "translated_responses/hi/BC_1.json"
                if not os.path.exists(previous_file):
                    previous_file = None
                
                await translate_jsonl_file(input_file, output_file, translator, sem,
                                          dest=DEST_LANG, previous_file=previous_file)
                print(f'Translated {input_file} to {output_file}')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
