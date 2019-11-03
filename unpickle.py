import pickle
import argparse
from typing import List

def unpickle(filename: str) -> List[str]:
    """ Take a .pkl (binary) file, load it into python and print return the equivalent
    list of ASCII strings. """
    unpickled_data = []  # type: List[str]
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
        f.write("\n".join(unpickled_data))
