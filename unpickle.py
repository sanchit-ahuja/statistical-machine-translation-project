import pickle
import argparse

def unpickle(filename: str) -> None:
    """ Take a .pkl (binary) file, load it into python and print it back in ASCII. """
    output_filename = ".".join(filename.split(".")[:-1]) + ".txt"
    unpickled_data = ""
    with open(filename, "rb") as binary_data:
        unpickled_data = pickle.load(binary_data)
    with open(output_filename, "w+") as f:
        f.write("\n".join(unpickled_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file to unpickle")
    args = parser.parse_args()
    unpickle(args.filename)
