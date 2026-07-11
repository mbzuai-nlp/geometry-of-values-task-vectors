import json
import os

def convert_to_dpo_format(input_file, output_file, task):
    """
    Convert preference data to DPO format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        task: Task type (AB, BA, BC, CB, CA, AC)
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dpo_data = []
    
    # Define the mapping for what each task prefers
    task_preferences = {
        'AB': ('A', 'B'),  # A is chosen, B is rejected
        'BA': ('B', 'A'),  # B is chosen, A is rejected
        'BC': ('B', 'C'),  # B is chosen, C is rejected
        'CB': ('C', 'B'),  # C is chosen, B is rejected
        'CA': ('C', 'A'),  # C is chosen, A is rejected
        'AC': ('A', 'C')   # A is chosen, C is rejected
    }
    
    chosen_option, rejected_option = task_preferences[task]
    
    for item in data:
        prompt = item['prompt']
        is_swapped = item.get('swapped', False)
        
        # If swapped is True, the options were swapped in the prompt to avoid position bias
        # So we need to flip our chosen/rejected assignments
        if is_swapped:
            actual_chosen = 'B'
            actual_rejected = 'A'
        else:
            actual_chosen = 'A'
            actual_rejected = 'B'
        # This messes up the reverse priority case. Do not use without modification. Data updated later seperately
        
        # Create DPO format entry
        dpo_entry = {
            'prompt': prompt,
            'chosen': actual_chosen,
            'rejected': actual_rejected,
            'index': item['index'],
            'swapped': is_swapped
        }
        
        dpo_data.append(dpo_entry)
    
    # Save the converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(dpo_data)} entries to DPO format")
    print(f"Saved to: {output_file}")

# Example usage - convert one task at a time
if __name__ == "__main__":
    # lang = "zh-cn"
    langs = ["en", "ar", "es", "hi", "zh-cn"]
    tasks = ["AB", "BA", "BC", "CB", "CA", "AC"]
    # task = "AB"
    
    for lang in langs:
        for task in tasks:
            # Convert training data
            train_input = f"datav2/prompt_response/{lang}/train/{task}.json"
            train_output = f"datav2/dpo_format/{lang}/train/{task}.json"
            convert_to_dpo_format(train_input, train_output, task)
            
            # Convert test data
            test_input = f"datav2/prompt_response/{lang}/test/{task}.json"
            test_output = f"datav2/dpo_format/{lang}/test/{task}.json"
            convert_to_dpo_format(test_input, test_output, task)
            
            #  Convert dev data
            dev_input = f"datav2/prompt_response/{lang}/dev/{task}.json"
            dev_output = f"datav2/dpo_format/{lang}/dev/{task}.json"
            convert_to_dpo_format(dev_input, dev_output, task)

            print(f"\nConversion complete for {lang} {task}") 