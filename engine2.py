from typing import List, Dict, Set, Any
import collections
from collections import defaultdict
import time
import gc
from unpickle import unpickle
from engine import get_vocab,converged,engine,printv,write_back_data


def engine2(dutch_sentences, english_sentences,max_iterations = 5):
    a_dict = defaultdict(float)
    translation_table_prev = unpickle("translation_probabilities_table.pkl")

    # translation_table_prev = engine(dutch_sentences,english_sentences,max_iterations,0.0001,"","",False,False)

    for i in range(5): #Convergence loop running it for 5 iterations for obvious reasons
        count = defaultdict(float)
        total = defaultdict(float)
        count_a = defaultdict(float)
        total_a = defaultdict(float)
        s_total = defaultdict(float)
        for english_sentence, dutch_sentence in zip(english_sentences,dutch_sentences):
            le = len(english_sentence)
            lf = len(dutch_sentence)
            for (j,e) in enumerate(le,1):
                s_total[e] = 0
                for (i,f) in enumerate(lf,1):
                    a_dict[(i,j,le,lf)] = 1.0*(1/(lf+1))
                    s_total[e] += translation_table_prev


