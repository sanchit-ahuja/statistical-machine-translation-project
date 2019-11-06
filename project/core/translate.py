import argparse
from project.tools.unpickle import unpickle
from typing import List, Dict
from project.core.preprocess import get_sentences_from_document, clean_sentences, printv

def translate(dutch_sentence: str, translation_table: Dict[str, Dict[str, float]],
              augment: bool=False) -> str:
    # NOTE: We will assume a linear correlation/alignment between
    # English words and Dutch word since both have similar origins.
    english_words = [""]  # type: List[str]
    last_alternative_estimate = ""  # the word with the second highest probability for the last Dutch word.
    last_choosen_probability = 0.0  # probability of the last word added to english_words
    for word in dutch_sentence.split():
        try:
            entry = translation_table[word]
            sorted_entry = sorted(entry.items(), key=lambda entry: entry[1])
            english_estimate, current_probability = sorted_entry[-1]
            if augment:
                if english_estimate == english_words[-1]:
                    # Case handled here: often phrases like "u bent" ("you are" in English) are
                    # translated to "you you" when the model has not received sufficient training
                    # because the are often occurs with the word you and the word you is very
                    # common. To handle this case we look at any case where we might have the same
                    # word occurring twice and then look at which of the two is more likely to be
                    # the correct one and which Dutch word needs to be translated to it's second
                    # most probable translation.
                    if current_probability > last_choosen_probability:
                        # Then this word is more likely the correct translation and the previous
                        # translation need to be changed.
                        english_words[-1] = last_alternative_estimate
                        last_alternative_estimate, last_choosen_probability = sorted_entry[-2][0], current_probability
                    else:
                        # Then this word is more likely to be it's second most likely translation.
                        english_estimate, current_probability = sorted_entry[-2]
                        last_alternative_estimate, last_choosen_probability = sorted_entry[-3][0], current_probability
                else:
                    last_alternative_estimate, last_choosen_probability = sorted_entry[-2][0], current_probability
            english_words.append(english_estimate)
        except KeyError:
            # Then it might just be a name, number or a rare word, so let's just use it as it.
            english_words.append(word)
    english_words.pop(0)
    return " ".join(english_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("document", help="The text file containing dutch sentences to translate. Each sentence must occur on a new line.")
    parser.add_argument("-o", "--output", help="The file to write the output to.")
    parser.add_argument("-t", "--translation_probabilities_table", help="The translations probability table to be used.", default="translation_probabilities_table.pkl")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode.", action="store_true")
    parser.add_argument("-a", "--augment", help="Try to do some optimiztion and improve the translation using ad hoc methods.", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    raw_sentences = get_sentences_from_document(args.document)
    normalized_sentences = clean_sentences(raw_sentences, keep_numbers=True)
    printv("Done.", verbose)
    printv("Loading translations probability table... ", verbose, end="")
    unpickled_data = unpickle(args.translation_probabilities_table)
    assert(isinstance(unpickled_data, Dict))
    translation_table = unpickled_data["data"]
    assert(isinstance(translation_table, Dict))
    printv("Done.", verbose)
    printv("Translating sentences... ", verbose, end="")
    translated_sentences = [translate(sentence, translation_table, args.augment) for sentence in normalized_sentences]
    printv("Done.", verbose)
    printv("Writing results to file... ", verbose, end="")
    if args.output:
        with open(args.output, "w+") as f:
            for sentence in translated_sentences:
                f.write(sentence + ".\n")
    else:
        for sentence in translated_sentences:
            print(sentence + "\n", end=".")
