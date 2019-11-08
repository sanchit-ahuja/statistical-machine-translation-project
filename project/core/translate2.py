from typing import List, Dict, Set, Any
import collections
from collections import defaultdict
import time
import gc
import pickle
from project.tools.unpickle import unpickle
from project.core.train import get_vocab,converged,train,printv,write_back_data
from project.core.translate import translate

alignment_prob = unpickle("final_alignment_prob.pkl")
translation_prob = unpickle("final_translation_prob.pkl")
translation_table_prev = unpickle("translation_probabilities_table.pkl")["data"]
dutch_sentences = unpickle("datasets/training/dutch/dutch_1p_5t.reduced.pkl")
english_sentences = unpickle("datasets/training/english/english_1p_5t.reduced.pkl")
english_sentences = english_sentences[:5]
dutch_sentences = dutch_sentences[:5]

def handle_alignment(translation_prob, alignment_prob,english_sentence,dutch_sentence):
    translation_ans = defaultdict(float)
    l_e = len(english_sentence)
    l_f = len(dutch_sentence)
    final_english_sentence= dict()
    for (j,e) in enumerate(english_sentence.split(),1):
        cur_max = (0,-1)
        for (i,f) in enumerate(dutch_sentence.split(),1):
            print(translation_prob[(e,f)])
            val = translation_prob[(e,f)]*alignment_prob[(i,j,l_e,l_f)]
            if cur_max[1] < val:
                cur_max = (i,val)
        translation_ans[j] = cur_max[0]
        final_english_sentence[translation_ans[j]]= e
    # print(translation_ans)
    return final_english_sentence


# st=time.time()
# input to following function is the Dutch, original sentence from corpus followed by the translated english sentence by IBM model 1
def final_sentence_list(dutch_sentences,english_sentences,translation_prob,alignment_prob): 
    sentence_list = []
    for es,ds in zip(english_sentences,dutch_sentences):
        l = handle_alignment(translation_prob,alignment_prob,es,ds)
        l = {i: l[i] for i in sorted(l)}
        # print(l)
        sentence = ""
        for word in l:
            sentence += l[word] + " "
        sentence_list.append(sentence)
    
    return sentence_list



english_sentence_translate_ibm1 = []

for ds in dutch_sentences:
    temp = translate(ds,translation_table_prev,True)
    english_sentence_translate_ibm1.append(temp)

# print(handle_alignment(translation_prob,alignment_prob,english_sentences[1],dutch_sentences[1]))
print(handle_alignment(translation_prob,alignment_prob,english_sentence_translate_ibm1[1],dutch_sentences[1]))
# for k in english_sentences:
#     print(k)

final_list = []


