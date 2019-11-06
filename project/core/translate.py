import argparse
from project.tools.unpickle import unpickle
from typing import List, Dict
from project.core.preprocess import get_sentences_from_document, clean_sentences, printv

def translate(dutch_sentence: str) -> str:
    # NOTE: We will assume a linear correlation/alignment between
    # English words and Dutch word since both have similar origins.
    translation_table = unpickle("translation_probabilities_table.pkl")
    english_words = []  # type: List[str]
    for word in dutch_sentence.split():
        try:
            entry = translation_table["data"][word]
            print((entry))
            english_estimate = sorted(entry.items(), key=lambda entry: entry[1])[-1][0]
            english_words.append(english_estimate)
        except KeyError:
            # Then it might just be a name, number or a rare word, so let's just use it as it.
            english_words.append(word)
    return " ".join(english_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("document", help="The text file containing dutch sentences to translate. Each sentence must occur on a new line.")
    parser.add_argument("-o", "--output", help="The file to write the output to.")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode.", action="store_true")
    args = parser.parse_args()
    raw_sentences = get_sentences_from_document(args.document)
    normalized_sentences = clean_sentences(raw_sentences, keep_numbers=True)
    translated_sentences = [translate(sentence) for sentence in normalized_sentences]
    if args.output:
        with open(args.output, "w+") as f:
            for sentence in translated_sentences:
                f.write(sentence + ".\n")
    else:
        for sentence in translated_sentences:
            print(sentence + "\n", end=".")
