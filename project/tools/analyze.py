import json
import pickle
import argparse

# Manually check to see how well the translation_probabilities_table was generated with this command line tool.
# Pass a Dutch word (not a phrase) and see the top 10 most likely translations to English.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dutch_word", help="The dutch word to translate.")
    parser.add_argument("-t", "--translation_probabilities_table", help="The translations probabilities table pkl file.", required=True)
    args = parser.parse_args()
    dutch_word = args.dutch_word
    filename = args.translation_probabilities_table
    with open(filename, "rb") as f:
        data = pickle.load(f)["data"]
        f.close()
    sorted_data = sorted(data[args.dutch_word].items(), key=lambda entry: entry[1])[::-1]
    for e in sorted_data[0:15]:
        print(e)
