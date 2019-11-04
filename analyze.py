import json
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dutch_word", help="The dutch word to translate")
    args = parser.parse_args()
    dutch_word = args.dutch_word

    with open("trained_translation_table.pkl", "rb") as f:
        data = pickle.load(f)
        f.close()
    sorted_data = sorted(data[args.dutch_word].items(), key=lambda entry: entry[1])[::-1]
    for e in sorted_data[0:15]:
        print(e)
