#!/usr/bin/env python3
"""
Script to create dev sets by carving out 400 samples from train files.
Maintains consistency across languages and paired files (AB/BA, BC/CB, CA/AC).
"""

import json
import os
import random
from typing import Dict, List, Set

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(filepath: str, data: List[Dict]) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_sample_indices(data: List[Dict], n_samples: int, seed: int = 42) -> Set[str]:
    """Get n_samples random indices from the data."""
    random.seed(seed)
    all_indices = [item['index'] for item in data]
    selected_indices = random.sample(all_indices, n_samples)
    return set(selected_indices)

def split_data_by_indices(data: List[Dict], dev_indices: Set[str]) -> tuple:
    """Split data into dev and train based on indices."""
    dev_data = []
    train_data = []
    
    for item in data:
        if item['index'] in dev_indices:
            dev_data.append(item)
        else:
            train_data.append(item)
    
    return dev_data, train_data

def main():
    languages = ['en', 'es', 'hi', 'ar', 'zh-cn']
    file_pairs = [
        ('AB.json', 'BA.json'),
        ('BC.json', 'CB.json'), 
        ('CA.json', 'AC.json')
    ]
    
    n_dev_samples = 400
    base_path = 'datav2/prompt_response'
    
    print(f"Creating dev sets with {n_dev_samples} samples each...")
    print(f"Processing languages: {languages}")
    
    # Step 1: Determine dev indices for each file pair using English data
    dev_indices_by_pair = {}
    
    for i, (file1, file2) in enumerate(file_pairs):
        print(f"\nProcessing pair {i+1}: {file1}/{file2}")
        
        # Load English data for this pair to get indices
        en_file1_path = os.path.join(base_path, 'en', 'train', file1)
        en_data = load_json_file(en_file1_path)
        
        # Get random sample of indices for this pair
        dev_indices = get_sample_indices(en_data, n_dev_samples, seed=42+i)
        dev_indices_by_pair[f"{file1}_{file2}"] = dev_indices
        
        print(f"Selected {len(dev_indices)} indices for {file1}/{file2} pair")
    
    # Step 2: Process each language
    for lang in languages:
        print(f"\n--- Processing language: {lang} ---")

        lang_train_path = os.path.join(base_path, lang, 'train')
        lang_dev_path = os.path.join(base_path, lang, 'dev')

        # Create dev directory if it doesn't exist
        os.makedirs(lang_dev_path, exist_ok=True)
        
        # Process each file pair
        for file1, file2 in file_pairs:
            pair_key = f"{file1}_{file2}"
            dev_indices = dev_indices_by_pair[pair_key]
            
            print(f"  Processing {file1} and {file2}")
            
            # Process first file in pair
            file1_path = os.path.join(lang_train_path, file1)
            file1_data = load_json_file(file1_path)
            file1_dev, file1_train = split_data_by_indices(file1_data, dev_indices)
            
            # Process second file in pair
            file2_path = os.path.join(lang_train_path, file2)
            file2_data = load_json_file(file2_path)
            file2_dev, file2_train = split_data_by_indices(file2_data, dev_indices)
            
            # Verify splits are consistent
            assert len(file1_dev) == len(file2_dev) == n_dev_samples, \
                f"Dev splits inconsistent: {file1}={len(file1_dev)}, {file2}={len(file2_dev)}, expected={n_dev_samples}"
            assert len(file1_train) == len(file2_train) == (3600 - n_dev_samples), \
                f"Train splits inconsistent: {file1}={len(file1_train)}, {file2}={len(file2_train)}"
            
            # Save dev files
            save_json_file(os.path.join(lang_dev_path, file1), file1_dev)
            save_json_file(os.path.join(lang_dev_path, file2), file2_dev)
            
            # Save updated train files
            save_json_file(os.path.join(lang_train_path, file1), file1_train)
            save_json_file(os.path.join(lang_train_path, file2), file2_train)
            
            print(f"    {file1}: {len(file1_train)} train + {len(file1_dev)} dev")
            print(f"    {file2}: {len(file2_train)} train + {len(file2_dev)} dev")
    
    # Step 3: Verification
    print(f"\n--- Verification ---")
    for lang in languages:
        print(f"\n{lang}:")
        train_path = os.path.join(base_path, lang, 'train')
        dev_path = os.path.join(base_path, lang, 'dev')
        
        for filename in ['AB.json', 'BA.json', 'BC.json', 'CB.json', 'CA.json', 'AC.json']:
            train_data = load_json_file(os.path.join(train_path, filename))
            dev_data = load_json_file(os.path.join(dev_path, filename))
            
            print(f"  {filename}: {len(train_data)} train + {len(dev_data)} dev = {len(train_data) + len(dev_data)} total")
    
    print(f"\n✅ Dev set creation completed!")
    print(f"Each language now has:")
    print(f"  - train/: 6 files with 3200 samples each")
    print(f"  - dev/: 6 files with 400 samples each")

if __name__ == "__main__":
    main() 
