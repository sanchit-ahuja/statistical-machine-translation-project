import gc
import time
import pickle
import argparse
from typing import List, Dict, Set, Any, Tuple
from collections import defaultdict

# from other code in this folder:
from project.core.preprocess import printv
from project.tools.unpickle import unpickle

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
    if iteration != 0 and counts != {}:
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

def write_back_data(data: Any, output_filename: str, verbose: bool=False) -> None:
    printv("Writing back training data to {}.pkl...".format(output_filename), verbose, end="")
    with open(output_filename + ".pkl", "wb") as f:
        pickle.dump(data, f)
    printv("Done.", verbose)


def train_table(dutch_sentences: List[str], english_sentences: List[str], max_iterations: int,
                convergence_factor: float, output_filename: str, resume_from_file: str,
                write_back_epoach: bool=False, verbose: bool=False) -> None:
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
        if write_back_epoach or (iteration == max_iterations):
            write_back_data({"iteration": iteration, "data": translation_table}, output_filename, verbose)


class TranslationMatrix:
    # NOTE: The variables are named with the assumption of Dutch -> English
    # translation but in practice we could always swap the values passed to
    # english_vocab and dutch_vocab for reverse translation. No problems.
    def __init__(self, dutch_vocab: List[str], english_vocab: List[str], initial_probability: float=None,
                 iteration: int=0, matrix: List[List[float]]=None) -> None:
            self.iteration = iteration
            self.dutch_vocab = sorted(dutch_vocab)
            self.english_vocab = sorted(english_vocab)
            if matrix is None:
                if initial_probability is None: 
                    initial_probability = 1/len(english_vocab)
                self.matrix = [[initial_probability for i in range (len(self.english_vocab))] for j in range (len(self.dutch_vocab))]
            else:
                self.matrix = matrix
            self.dutch_vocab_index = {word: index for index, word in enumerate(self.dutch_vocab)}
            self.english_vocab_index = {word: index for index, word in enumerate(self.english_vocab)}

    def get_translation_probability(self, dutch_word: str, english_word: str) -> float:
        di = self.dutch_vocab_index[dutch_word]
        ei = self.english_vocab_index[english_word]
        return self.matrix[di][ei]

    def get_sorted_row(self, dutch_word: str) -> List[Tuple[str, float]]:
        di = self.dutch_vocab_index[dutch_word]
        row = self.matrix[di]
        unsorted_pairs = [(self.english_vocab[ei], eng_prb) for ei, eng_prb in enumerate(row)]
        return sorted(unsorted_pairs, key=lambda x: x[1])

    def freeze(self, output_filename: str) -> None:
        data = {
            "iteration": self.iteration,
            "matrix": self.matrix,
            "english_vocab": self.english_vocab,
            "dutch_vocab": self.dutch_vocab
        }
        write_back_data(data, output_filename)

    @classmethod
    def thaw(cls, filename) -> "TranslationMatrix":
        data = unpickle(filename)
        return cls(
            dutch_vocab = data["dutch_vocab"],
            english_vocab = data["english_vocab"],
            matrix = data["matrix"]
        )


def train_matrix(dutch_sentences: List[str], english_sentences: List[str], max_iterations: int,
                 output_filename: str, resume_from_file: str, write_back_epoach: bool=False,
                 verbose: bool=False) -> None:
    if resume_from_file:
        printv("Reloading the translation matrix instance... ", verbose, end="")
        translation_matrix = TranslationMatrix.thaw(resume_from_file)
    else:
        printv("Creating a new translation matrix instance... ", verbose, end="")
        translation_matrix = TranslationMatrix(list(get_vocab(dutch_sentences)), list(get_vocab(english_sentences)))
    printv("Done.", verbose)

    counts = [
            [0.0 for i in range (len(translation_matrix.english_vocab))]
            for j in range (len(translation_matrix.dutch_vocab))
    ]
    printv("Beginning the Expectation-Maximization algorithm.", verbose)
    # To further improve speed, instead of convergence, we just specify the max number
    # of iterations to run for instead of testing convergence each iteration.
    # convergence takes too long anyways.
    while translation_matrix.iteration < max_iterations:
        start_time = time.time()
        translation_matrix.iteration += 1
        # count's iteration value does not need to increase, we made counts a TranslationMatrix just for the methods.
        totals = [0.0]*len(translation_matrix.dutch_vocab)

        printv("Calculating probabilities and collecting counts... ", verbose, end="")
        for english_sentence, dutch_sentence in zip(english_sentences, dutch_sentences):
            subtotals = defaultdict(float)  # type: Dict[str, float]
            # This first loop is for finding the subtotals which will contain the normalization factors for
            # the counts collection stage.
            for english_word in english_sentence.split():
                for dutch_word in dutch_sentence.split():
                    subtotals[english_word] += translation_matrix.get_translation_probability(dutch_word, english_word)
            # This loop is for the counts collection stage.
            for english_word in english_sentence.split():
                for dutch_word in dutch_sentence.split():
                    di = translation_matrix.dutch_vocab_index[dutch_word]
                    ei = translation_matrix.english_vocab_index[english_word]
                    amount = translation_matrix.matrix[di][ei]/subtotals[english_word]
                    counts[di][ei] += amount
                    totals[di] += amount 
                    # totals contains yet more normalization factors for the updating translations probabilities stage.
        printv("Done.", verbose)

        printv("Updating translations probabilities table... ", verbose, end="")
        for dutch_word in translation_matrix.dutch_vocab:
            for english_word in translation_matrix.english_vocab:
                di = translation_matrix.dutch_vocab_index[dutch_word]
                ei = translation_matrix.english_vocab_index[english_word]
                translation_matrix.matrix[di][ei] = counts[di][ei]/totals[di]
        printv("Done.", verbose)

        # clear counts.
        l1 = len(translation_matrix.dutch_vocab)
        l2 = len(translation_matrix.english_vocab)
        for i in range(l1):
            for j in range(l2):
                counts[i][j] = 0.0

        end_time = time.time()
        print("Completed iteration {} in {} seconds".format(translation_matrix.iteration, end_time - start_time))
        if write_back_epoach or (translation_matrix.iteration == max_iterations):
            printv("Freezing the translations probability matrix... ", verbose, end="")
            translation_matrix.freeze(output_filename)
            printv("Done.", verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Print what's going on to the console.", action="store_true")
    parser.add_argument("-o", "--output", help="The file to write to without the file extension.", default="translation_probabilities_table")
    parser.add_argument("-p", "--percentage", help="Choose a preprocessed percentage to train on. Accepted values: [1, 3, 5, 10]", default=1)  # TODO: allow using non-standard datasets
    parser.add_argument("-m", "--max_iterations", help="The maximum number of iterations to run for before terminating training.", default=5)
    parser.add_argument("-c", "--convergence_factor", help="The convergence factor as a decimal, e.g. 0.0001 is 0.01%%.", default=0.0001)
    parser.add_argument("-w", "--write_back_epoach" , help="This flag indicates that the translation probabilities table data should be written back to disk after each epoach.", action="store_true")
    parser.add_argument("-r", "--resume_from_file" , help="Resume training using an existing dataset by specifying the file to use.", default="")
    parser.add_argument("-i", "--invert", help="Invert the order of translation to English -> Dutch.", action="store_true")
    parser.add_argument("-x", "--matrix", help="Generate a translation probabilities matrix (list of lists) instead of a table (dict of dicts)", action="store_true")
    args = parser.parse_args()

    training_set = int(args.percentage)
    if training_set not in [1, 3, 5, 10]:
        raise ValueError("Invaild percentage value. Valid values: [1, 3, 5, 10].")

    printv("Beginning training with the {}% dataset.".format(training_set), args.verbose)
    dutch_sentences = unpickle("datasets/training/dutch/dutch_{}p_5t.reduced.pkl".format(training_set))
    english_sentences = unpickle("datasets/training/english/english_{}p_5t.reduced.pkl".format(training_set))

    if args.invert:
        english_sentences, dutch_sentences = dutch_sentences, english_sentences

    if args.matrix:
        if args.output == "translation_probabilities_table":
            args.output = "translation_probabilities_matrix"
        train_matrix(dutch_sentences, english_sentences, int(args.max_iterations), args.output,
                     args.resume_from_file, args.write_back_epoach, args.verbose)
    else:
        train_table(dutch_sentences, english_sentences, int(args.max_iterations), float(args.convergence_factor),
                    args.output, args.resume_from_file, args.write_back_epoach, args.verbose)
    printv("Training complete.", args.verbose)
