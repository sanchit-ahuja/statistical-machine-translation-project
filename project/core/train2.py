from typing import List, Dict, Set, Any
import collections
from collections import defaultdict
import time
import gc
import pickle
from project.tools.unpickle import unpickle
from project.core.train import get_vocab,converged,train,printv,write_back_data


def train2(dutch_sentences, english_sentences):
    a_dict = defaultdict(float)
    translation_table_prev = unpickle("translation_probabilities_table.pkl")
    translation_table_prev = translation_table_prev["data"]

    cnt = 0
    for i in range(6):
        cnt += 1
        print("Iteration: ",cnt)
        count = defaultdict(float)
        total = defaultdict(float)
        count_a = defaultdict(float)
        total_a = defaultdict(float)
        s_total = defaultdict(float)
        for english_sentence, dutch_sentence in zip(english_sentences,dutch_sentences):
            le = len(english_sentence)
            lf = len(dutch_sentence)
            #Compute normalization
            for (j,e) in enumerate(english_sentence.split(),1):
                s_total[e] = 0
                for (i,f) in enumerate(dutch_sentence.split(),1):
                    a_dict[(i,j,le,lf)] = 1.0*(1/(lf+1))
                    s_total[e] += translation_table_prev[f][e]*a_dict[(i,j,le,lf)]
            # print('Normalization')
            #Compute Counts
            for (j,e) in enumerate(english_sentence.split(),1):
                for (i,f) in enumerate(dutch_sentence.split(),1):
                    # a_dict[(i,j,le,lf)] = 1.0*(1/(lf+1))
                    c = (translation_table_prev[f][e]*a_dict[(i,j,le,lf)])/s_total[e]
                    count[(e,f)] += c
                    total[f] += c
                    count_a[(i,j,le,lf)] += c
                    total_a[(j,le,lf)] += c
            # print('Cnt')

        #estimate probabilites
        final_translation_prob = defaultdict(float)
        final_alignment_prob = defaultdict(float)
        for (e,f) in count.keys():
            final_translation_prob[(e,f)] = count[(e,f)]/total[f]

        for i,j,le,lf in count_a.keys():
            final_alignment_prob[(i,j,le,lf)] = count_a[(i,j,le,lf)]/total_a[(j,le,lf)]

    return final_alignment_prob,final_translation_prob





if __name__ == "__main__":


    dutch_sentences = unpickle("datasets/training/dutch/dutch_1p_5t.reduced.pkl")
    # dutch_sentences = dutch_sentences[:5]
    english_sentences = unpickle("datasets/training/english/english_1p_5t.reduced.pkl")
    # english_sentences = english_sentences[:5]

    # alignment training model:
    final_alignment_prob,final_translation_prob = train2(dutch_sentences,english_sentences) #a is the translation probabilty b is the alignment prob
    with open ('final_alignment_prob.pkl','wb') as f:
        pickle.dump(final_alignment_prob,f)
    with open ('final_translation_prob.pkl','wb') as f:
        pickle.dump(final_translation_prob,f)





