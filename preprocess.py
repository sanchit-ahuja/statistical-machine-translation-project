import re
import string
import pickle
import argparse
from typing import List, Dict, Set
from collections import Counter
from unicodedata import normalize


def printv(msg: str, v: bool, **kwargs) -> None:
    """ Used to print a message only if the verbosity option has been enabled. """
    kwargs.update(flush=True)
    if v:
        print(msg, **kwargs)
    return

def get_sentences_from_document(filename: str, percentage: int=100) -> List[str]:
    """ Load the target text file and return a list containing every sentence in it.
        Optionally, a percentage argument can also be passed to control the number of
        lines to process - useful for generating smaller datasets. """
    with open(filename, "r", encoding="utf8") as file:
        raw_text = file.read()
        sentences = raw_text.strip().split("\n")
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage exceeded bounds.")
    if percentage != 100:
        stop_index = int((len(sentences)*percentage)/100)
        sentences = sentences[0:stop_index]
    return sentences

def get_vocabulary_count(sentences: List[str]) -> Dict[str, int]:
    """ Return a mapping: {unique_word -> number_of_occurrences} based on the given sentences. """
    voc = Counter()  # type: Counter
    for sentence in sentences:
        words = sentence.split()
        voc.update(words)
    return voc

def reduce_vocab(vocab: Dict[str, int], threshold: int) -> Set[str]:
    """ Return a set of words which have occurred at least the `threshold` number of times. """
    return set([word for word, cnt in vocab.items() if cnt >= threshold])

def update_dataset(sentences: List[str], vocab: Set[str]) -> List[str]:
    """ Take the original sentences and then use the cleaned and reduced vocabulary to update them. """
    new_sentences = []
    for line in sentences:
        new_words = []
        for t in line.split():
            if t in vocab:
                new_words.append(t)
            else:
                new_words.append("uup")
        new_words = [word for word in new_words if word !="s"]
        new_line = " ".join(new_words)
        new_sentences.append(new_line)
    return new_sentences

def clean_sentences(lines: List[str]) -> List[str]:
    """ Normalize/clean the input lines by:
            1. Converting phonetic Latin characters to their simple English form. E.g. o` -> o
            2. Making all words lowercase.
            3. Removing punctuation characters.
            4. Removing all non-printable characters.
            5. Removing all numeric characters. """
    cleaned_sentences =  []
    re_pat = re.compile("[^%s]" % re.escape(string.printable))  # filter printable chars
    table = str.maketrans("", "", string.punctuation)
    # `table` is a dictionary mapping the ascii value of each punctutation character to None.
    # We use this to remove punctutation characters in the code below.
    for line in lines:
        line = normalize("NFD", line).encode("ascii", "ignore").decode("UTF-8")  # normalizing chars such as o` and o.
        words = line.split()  # tokenize on whitespaces
        lowercase_words = [word.lower() for word in words]  # lowercase all the words
        punctuationless_words = [word.translate(table) for word in lowercase_words]  # remove all the punctuation marks
        # WARNING: a small issue is that i'll -> ill and there is already a word called ill.
        # We need to expand English contractions like i'll to i will. For now we expect that
        # it will not affect the probability too noticeably.
        printable_words = [re_pat.sub("", word) for word in punctuationless_words]  # remove non-printable chars
        non_numeric_words = [word for word in printable_words if word.isalpha()]  # remove numbers
        cleaned_sentences.append(" ".join(non_numeric_words))
    return cleaned_sentences

def preprocess_file(input_filename: str, output_filename: str, vocab_reduction_threshold: int=4,
                    percentage: int=100, verbose: bool=False) -> None:
    """ Take one of the original input files and then completely preprocess it. """
    # TODO: Add a check to skip steps if cleaned data exists.
    printv("Getting sentences from document... ", verbose, end="")
    raw_sentences = get_sentences_from_document(input_filename, percentage)
    printv("Done.", verbose)
    printv("Cleaning/Normalizing sentences... ", verbose, end="")
    cleaned_sentences = clean_sentences(raw_sentences)
    printv("Done.", verbose)
    printv("Dumping un-reduced data to {}... ".format(output_filename + ".pkl"), verbose, end="")
    pickle.dump(cleaned_sentences, open(output_filename + ".pkl", "wb"))
    printv("Done.", verbose)
    printv("Generating reduced vocabulary... ", verbose, end="")
    vocab = get_vocabulary_count(cleaned_sentences)
    reduced_vocab = reduce_vocab(vocab, vocab_reduction_threshold)
    printv("Done.", verbose)
    printv("Updating dataset with reduced vocabulary... ", verbose, end="")
    new_data = update_dataset(cleaned_sentences, reduced_vocab)
    printv("Done.", verbose)
    printv("Dumping reduced data to {}... ".format(output_filename + ".reduced.pkl"), verbose, end="")
    pickle.dump(new_data, open(output_filename + ".reduced.pkl", "wb"))
    printv("Done.", verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path of file to process.")
    parser.add_argument("-o", "--output", help="path of file to write processed data to process without an extension.", default="output")
    parser.add_argument("-t", "--vocab_reduction_threshold", help="set the vocab reduction threshold.", default=4)
    parser.add_argument("-p", "--percentage", help="the percentage of the dataset to process, useful for generating smaller test sets.", default=100)
    parser.add_argument("-v", "--verbose", help="Print what's going on to the console.", action="store_true")
    args = parser.parse_args()
    percentage = int(args.percentage)
    if args.verbose:
        print("Beginning preprocessing, generating dataset ({}% of the input).".format(percentage))
    preprocess_file(args.input, args.output, int(args.vocab_reduction_threshold), percentage, args.verbose)
