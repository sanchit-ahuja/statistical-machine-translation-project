import gc
import time
import pickle
import argparse
from typing import List, Dict, Set, Any
from collections import defaultdict

# from other code in this folder:
from unpickle import unpickle
from preprocess import printv

TranslationTable = Dict[str, Dict[str, float]]  # {foreign word: {english word: probability}}

def get_vocab(sentences: List[str]) -> Set[str]:
    """ Given a list of sentences, return the set of unique words."""
    vocab = set()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            vocab.add(word)
    return vocab

def converged(dutch_vocab: Set[str], english_vocab: Set[str], translation_table: TranslationTable,
              counts: TranslationTable, convergence_factor: float, max_iterations: int,
              iteration: int=0, verbose: bool=False) -> bool:
    """ After we have reached the maximum number of iterations/epoaches or have reached the convergence
        condition (negligible change observed in probabilities) then we can stop training."""
    printv("Testing convergence... ", verbose, end="")
    if iteration == max_iterations:
        # Since training takes a long while, we want to be able to limit the number of
        # iterations then just stop and use whatever model we have at the end of max_iterations
        # number of iterations.
        printv("Done.", verbose)
        return True
    if iteration != 0:
        for dutch_word in dutch_vocab:
            for english_word in english_vocab:
                if abs(translation_table[dutch_word][english_word] - counts[dutch_word][english_word]) > convergence_factor:
                    # Then that means that some improvement has happened
                    # somewhere and that we should continue.
                    printv("Done.", verbose)
                    return False
        printv("Done.", verbose)
        return True
    printv("Done.", verbose)
    return False

def write_back_data(data: Dict[str, Any], output_filename: str, verbose: bool=False) -> None:
    printv("Writing back training data to {}.pkl...".format(output_filename), verbose, end="")
    with open(args.output + ".pkl", "wb") as f:
        pickle.dump(data, f)
    printv("Done.", verbose)

def engine(dutch_sentences: List[str], english_sentences: List[str], max_iterations: int,
           convergence_factor: float, output_filename: str, resume_from_file: str,
           write_back_epoach: bool=False, verbose: bool=False) -> TranslationTable:
    """ The engine for training the statistical machine translator based on the IBM Model 1
        (TODO: explain more here in this docstring). """
    printv("Determining the vocabularies... ", verbose, end="")
    english_vocab = get_vocab(english_sentences)
    dutch_vocab = get_vocab(dutch_sentences)
    printv("Done.", verbose)

    if resume_from_file:
        printv("Reloading the translation probabilities table... ", verbose, end="")
        reloaded_data = unpickle(resume_from_file)
        iteration = reloaded_data["iteration"]
        translation_table = reloaded_data["data"]
        printv("Done.", verbose)
    else:
        printv("Intializing the translation probabilities table... ", verbose, end="")
        iteration = 0
        initial_probability = 1/(len(english_vocab))
        translation_table = {f: {e: initial_probability for e in english_vocab} for f in dutch_vocab}
        printv("Done.", verbose)

    counts = {}  # type: TranslationTable
    printv("Beginning the Expectation-Maximization algorithm.", verbose)
    while not converged(dutch_vocab, english_vocab, translation_table, counts, convergence_factor, max_iterations, iteration, verbose):
        start_time = time.time()
        iteration += 1

        printv("Intializing the counts and totals... ", verbose, end="")
        counts = {f: {e: 0.0 for e in english_vocab} for f in dutch_vocab}
        totals = {f: 0.0 for f in dutch_vocab}
        printv("Done.", verbose)

        printv("Calculating probabilities and collecting counts... ", verbose, end="")
        for english_sentence, dutch_sentence in zip(english_sentences, dutch_sentences):
            subtotals = defaultdict(float)  # type: Dict[str, float]
            for english_word in english_sentence.split():
                for dutch_word in dutch_sentence.split():
                    subtotals[english_word] += translation_table[dutch_word][english_word]
            for english_word in english_sentence.split():
                for dutch_word in dutch_sentence.split():
                    amount = translation_table[dutch_word][english_word]/subtotals[english_word]
                    counts[dutch_word][english_word] += amount
                    totals[dutch_word] += amount
        printv("Done.", verbose)

        printv("Updating translations probabilities table... ", verbose, end="")
        for dutch_word in dutch_vocab:
            for english_word in english_vocab:
                translation_table[dutch_word][english_word] = counts[dutch_word][english_word]/totals[dutch_word]
        printv("Done.", verbose)

        counts = {}
        totals = {}
        gc.collect()
        end_time = time.time()
        print("Completed iteration {} in {} seconds".format(iteration, end_time - start_time))
        write_back_data({"iteration": iteration, "data": translation_table}, output_filename, verbose)

    return translation_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Print what's going on to the console.", action="store_true")
    parser.add_argument("-o", "--output", help="The file to write to without the file extension.", default="translation_probabilities_table")
    parser.add_argument("-p", "--percentage", help="Choose a preprocessed percentage to train on. Accepted values: [1, 3, 5, 10]", default=1)
    parser.add_argument("-m", "--max_iterations", help="The maximum number of iterations to run for before terminating training.", default=5)
    parser.add_argument("-c", "--convergence_factor", help="The convergence factor as a decimal, e.g. 0.0001 is 0.01%.", default=0.0001)
    parser.add_argument("-w", "--write_back_epoach" , help="This flag indicates that the translation probabilities table data should be written back to disk after each epoach.", action="store_true")
    parser.add_argument("-r", "--resume_from_file" , help="Resume training using an existing dataset by specifying the file to use.", default="")
    args = parser.parse_args()

    training_set = int(args.percentage)
    if training_set not in [1, 3, 5, 10]:
        raise ValueError("Invaild percentage value. Valid values: [1, 3, 5, 10].")

    printv("Beginning training with the {}% dataset.".format(training_set), args.verbose)
    dutch_sentences = unpickle("datasets/dutch/dutch_{}p_5t.reduced.pkl".format(training_set))
    assert isinstance(dutch_sentences, List)  # for mypy
    english_sentences = unpickle("datasets/english/english_{}p_5t.reduced.pkl".format(training_set))
    assert isinstance(english_sentences, List)  # for mypy
    engine(dutch_sentences, english_sentences, int(args.max_iterations), float(args.convergence_factor),
           args.output, args.resume_from_file, args.write_back_epoach, args.verbose)
    printv("Training complete.", args.verbose)
