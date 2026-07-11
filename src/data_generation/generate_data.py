import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

PROMPT_DIR = Path("prompts")
SCENARIOS_FILE = PROMPT_DIR / "scenarios.md"
OUTPUT_BASE = Path("openai_responses")

MODEL = "gpt-4o-2024-08-06"
RUNS_PER_SCENARIO = 10

VALUES = [
    "Trustworthiness and Honesty",
    "Basic Justice (being fair)",
    "Respect for the autonomy of others",
]
VALUE_PAIRS = [
    [VALUES[0], VALUES[1]],
    [VALUES[1], VALUES[2]],
    [VALUES[2], VALUES[0]],
]


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    return OpenAI(api_key=api_key)


def parse_scenarios(file_path):
    scenarios = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    scenario_blocks = content.split("\n<next>\n\n")
    
    for block in scenario_blocks:
        lines = block.strip().split("\n")
        if not lines or lines[0].strip() == "":
            continue
        
        scenario_name_line = lines[0].strip()
        scenario_name = scenario_name_line.split(".")[1].strip()
        
        sublist = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("-"):
                sublist.append(line)
        
        scenarios.append({
            "SCENARIO_NAME": scenario_name,
            "SCENARIO_SUBLIST": "\n".join(sublist)
        })
    
    return scenarios


def load_templates():
    with open(PROMPT_DIR / "prompt0.md", "r", encoding="utf-8") as f:
        template0 = f.read()

    with open(PROMPT_DIR / "prompt1.md", "r", encoding="utf-8") as f:
        template1 = f.read()

    with open(PROMPT_DIR / "prompt2.md", "r", encoding="utf-8") as f:
        template2 = f.read()

    return template0, template1, template2


def generate_api(client, prompt):
    sent_tokens = len(prompt.split())
    # print(f"Sending prompt:\n{prompt}")

    # Call the GPT-4 model through the OpenAI API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens = 4096
    )
    
    # Extract the response text and count tokens received
    response_text = response.choices[0].message.content
    received_tokens = len(response_text.split())
    
    print(f"Words sent: {sent_tokens}, Words received: {received_tokens}")
    
    # Return the response and token counts
    return response_text, sent_tokens, received_tokens


def main():
    client = get_client()
    template0, template1, template2 = load_templates()
    scenarios = parse_scenarios(SCENARIOS_FILE)
    total_sent_tokens = 0
    total_received_tokens = 0

    for v, value_pair in enumerate(VALUE_PAIRS):
        value_1 = value_pair[0]
        value_2 = value_pair[1]

        for s, scenario in enumerate(scenarios):
            scenario_name = scenario["SCENARIO_NAME"]
            sublist = scenario["SCENARIO_SUBLIST"]

            filled_prompt = template0.format(
                SCENARIO_NAME=scenario_name,
                SCENARIO_SUBLIST=sublist,
                VALUE_1=value_1,
                VALUE_2=value_2,
            ) + template1 + template2.format(SCENARIO_NAME=scenario_name)

            print(f"Scenario: {s}, Values: {v}")
            output_dir = OUTPUT_BASE / f"valuepair_{v}"
            output_dir.mkdir(parents=True, exist_ok=True)

            for r in tqdm(range(RUNS_PER_SCENARIO), desc="Generating responses"):
                response, sent_tokens, received_tokens = generate_api(client, filled_prompt)
                filename = output_dir / f"scenario_{s}_value_{v}_run_{r}.txt"

                with open(filename, "w", encoding="utf-8") as output_file:
                    output_file.write(response)

                total_sent_tokens += sent_tokens
                total_received_tokens += received_tokens

    print(f"Total words sent: {total_sent_tokens}")
    print(f"Total words received: {total_received_tokens}")


if __name__ == "__main__":
    main()
