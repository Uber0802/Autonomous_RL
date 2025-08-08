import json
import os

_DATA_DIR = os.path.dirname(__file__)

def get_dataset_statistics(unnorm_key):
    fixed_meta_path = os.path.join(_DATA_DIR, "openvla_hf_config.json")
    with open(fixed_meta_path, "r") as f:
        config = json.load(f)
    
    return config["norm_stats"][unnorm_key]