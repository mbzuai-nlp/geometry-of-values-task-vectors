import torch
import copy
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, AutoConfig

def load_model(model_name_or_path, use_float16=False, device='cuda'):
    print("loading model")

    if use_float16:
        print("float16")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map = device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map = device)
    return model, tokenizer

def save_to_dir(model, tokenizer, save_dir, config=None):
    os.makedirs(save_dir, exist_ok=True)
    if config is None:
        model.config.save_pretrained(save_dir)
    else:
        config.save_pretrained(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
    tokenizer.save_pretrained(save_dir)