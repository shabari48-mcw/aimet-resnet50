import json
import os 
from pathlib import Path

    
    
@staticmethod
def parse_json(json_path:str)->dict:
    """
    Function to parse a json file and return a dictionary
    Args:
    json_path : str : Path to the json file
    Returns:
    dict : Dictionary containing the json data
    """
    with open(json_path, "r") as file:
        data = json.load(file)
        flat_dict = {}

        def flatten(data, parent_key=''):
            if isinstance(data, dict):
                for k, v in data.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    flatten(v, new_key)
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    new_key = f"{parent_key}[{i}]"
                    flatten(v, new_key)
            else:
                flat_dict[parent_key] = data

        flatten(data)
        return flat_dict
