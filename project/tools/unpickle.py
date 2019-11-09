import json
import pickle
import argparse
from typing import List, Dict, Union

def unpickle(filename: str) -> Union[List[str], Dict[str, Union[int, Dict[str, Dict[str, float]]]]]:
    """ Take a .pkl (binary) file, load it into python and print return the equivalent
    list of ASCII strings or equivalent dictionary. """
    unpickled_data = None
    with open(filename, "rb") as binary_data:
        unpickled_data = pickle.load(binary_data)
    return unpickled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file to unpickle")
    args = parser.parse_args()
    unpickled_data = unpickle(args.filename)
    output_filename = ".".join(args.filename.split(".")[:-1]) + ".txt"
    with open(output_filename, "w+") as f:
        if isinstance(unpickled_data, list):
            f.write("\n".join(unpickled_data))
        elif isinstance(unpickled_data, dict):
            f.write(json.dumps(unpickled_data, indent=4, sort_keys=True))
        f.close()
