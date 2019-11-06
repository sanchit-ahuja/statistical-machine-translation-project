import argparse
from unpickle import unpickle
from typing import List, Dict
from preprocess import get_sentences_from_document, clean_sentences, printv

def translate(dutch_sentence: str, translation_table: Dict[str, Dict[str, float]]) -> str:
    # NOTE: We will assume a linear correlation/alignment between
    # English words and Dutch word since both have similar origins.
    english_words = []  # type: List[str]
    for word in dutch_sentence.split():
        try:
            entry = translation_table[word]
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
    parser.add_argument("-v", "--verbose", help="Enable verbose mode.")
    args = parser.parse_args()
    verbose = args.verbose
    printv("Getting and normalizing sentences to translate... ", verbose, end="")
    raw_sentences = get_sentences_from_document(args.document)
    normalized_sentences = clean_sentences(raw_sentences, keep_numbers=True)
    printv("Done.", verbose)
    printv("Loading translations probability table... ", verbose, end="")
    unpickled_data = unpickle("translation_probabilities_table.pkl")
    assert(isinstance(unpickled_data, Dict))
    translation_table = unpickled_data["data"]
    assert(isinstance(translation_table, Dict))
    printv("Done.", verbose)
    printv("Translating sentences... ", verbose, end="")
    translated_sentences = [translate(sentence, translation_table) for sentence in normalized_sentences]
    printv("Done.", verbose)
    printv("Writing results to file... ", verbose, end="")
    if args.output:
        with open(args.output, "w+") as f:
            for sentence in translated_sentences:
                f.write(sentence + ".\n")
    else:
        for sentence in translated_sentences:
            print(sentence + "\n", end=".")
    printv("Done.", verbose)
