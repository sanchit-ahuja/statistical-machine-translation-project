import argparse
from project.tools.unpickle import unpickle
from typing import List, Dict
from project.core.preprocess import get_sentences_from_document, clean_sentences, printv
from project.core.train import TranslationMatrix

# TODO: Refactor and de-duplicate the code b/w table and matrix

def translate_from_matrix(dutch_sentences: List[str], translation_matrix: TranslationMatrix, augment: bool=False) -> List[str]:
    # NOTE: We will assume a linear correlation/alignment between
    # English words and Dutch word since both have similar origins.
    translated_sentences = []
    for dutch_sentence in dutch_sentences:
        english_words = [""]  # type: List[str]
        last_alternative_estimate = ""  # the word with the second highest probability for the last Dutch word.
        last_choosen_probability = 0.0  # probability of the last word added to english_words
        for word in dutch_sentence.split():
            try:
                sorted_entry = translation_matrix.get_sorted_row(word)
                english_estimate, current_probability = sorted_entry[-1]
                if augment:
                    if english_estimate == english_words[-1]:
                        if current_probability > last_choosen_probability:
                            english_words[-1] = last_alternative_estimate
                            last_alternative_estimate, last_choosen_probability = sorted_entry[-2][0], current_probability
                        else:
                            english_estimate, current_probability = sorted_entry[-2]
                            last_alternative_estimate, last_choosen_probability = sorted_entry[-3][0], current_probability
                    else:
                        last_alternative_estimate, last_choosen_probability = sorted_entry[-2][0], current_probability
                english_words.append(english_estimate)
            except KeyError:
                # Then it might just be a name, number or a rare word, so let's just use it as it.
                english_words.append(word)
        english_words.pop(0)
        translated_sentences.append(" ".join(english_words))
    return translated_sentences

def translate_from_table(dutch_sentence: str, translation_table: Dict[str, Dict[str, float]],
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
    parser.add_argument("-o", "--output", help="The file to write the output to.", default="translation.txt")
    parser.add_argument("-t", "--translation_probabilities_table", help="The translations probability table to be used.", default="translation_probabilities_table.pkl")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode.", action="store_true")
    parser.add_argument("-x", "--matrix", help="Indicate that this is a matrix instead of a table.", action="store_true")  # TODO: automatically detect and act.
    parser.add_argument("-a", "--augment", help="Try to do some optimiztion and improve the translation using ad hoc methods.", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    raw_sentences = get_sentences_from_document(args.document)
    normalized_sentences = clean_sentences(raw_sentences, keep_numbers=True)
    printv("Done.", verbose)
    if args.matrix:
        printv("Loading translations probability matrix... ", verbose, end="")
        translation_matrix = TranslationMatrix.thaw(args.translation_probabilities_table)
        printv("Done.", verbose)
        printv("Translating sentences... ", verbose, end="")
        translated_sentences = translate_from_matrix(normalized_sentences, translation_matrix, args.augment)
        printv("Done.", verbose)
    else:
        printv("Loading translations probability table... ", verbose, end="")
        unpickled_data = unpickle(args.translation_probabilities_table)
        translation_table = unpickled_data["data"]
        printv("Done.", verbose)
        printv("Translating sentences... ", verbose, end="")
        translated_sentences = [translate_from_table(sentence, translation_table, args.augment) for sentence in normalized_sentences]
        printv("Done.", verbose)
    printv("Writing results to file... ", verbose, end="")
    if args.output:
        with open(args.output, "w+") as f:
            for sentence in translated_sentences:
                f.write(sentence + ".\n")
    else:
        for sentence in translated_sentences:
            print(sentence + "\n", end=".")
