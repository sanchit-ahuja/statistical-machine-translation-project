# code to get pearsons coefficent

import time
import argparse
from math import log10,sqrt
from typing import List, Dict, Set
from collections import Counter
from project.core.preprocess import get_sentences_from_document,get_vocabulary_count,clean_sentences

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# takes sentence lists of both documents and generates two lists corresponding to their word vector spaces
def doc_vectors(sentence_list1: List[str],sentence_list2: List[str]) -> List:
    sentence_list1= clean_sentences(sentence_list1)
    sentence_list2= clean_sentences(sentence_list2)
    # getting vocab count for both
    raw_voc1=get_vocabulary_count(sentence_list1)
    raw_voc2=get_vocabulary_count(sentence_list2)

    # tf handling(logarithmic) -- ignoring idf for now- in voc1 and voc2, the term k has score voc1[k] and voc2[k]
    # note-
    st=time.time()
    voc1={k: (1+log10(v)) for k,v in raw_voc1.items()}
    voc2={k: (1+log10(v)) for k,v in raw_voc2.items()}

    # Calculating the magnitude of vectors- voc1 and voc 2 contain the non-zero parts of their respective vectors- squaring these yield the magnitude
    st=time.time()
    magnitude1=0.0
    magnitude2=0.0
    for k in voc1:
        j=voc1[k]
        magnitude1+=j*j
    for k in voc2:
        j=voc2[k]
        magnitude2+=j*j
    magnitude1=sqrt(magnitude1)
    magnitude2=sqrt(magnitude2)

    # sorting the keys by order
    voc1={i: voc1[i] for i in sorted(voc1)}
    voc2={i: voc2[i] for i in sorted(voc2)}

    # now to make the vectors:
    # merging the vocab dicts in order to get full vocabulary- focusing on keys here
    merged_words= {**voc1,**voc2}.keys()
    iterator1=list(range(len(merged_words)))
    merge_dict=dict(zip(merged_words,iterator1))

    # merge_dict now comprises of a list of the merged keys,each mapped to a value
    # i.e. a particular key will map to the same place in vector1 and vector2
    # using this mapping,we can create a vector

    # creating the actual vectors:
    len_merge_dict=len(merge_dict)
    vector1=[0.0]*len_merge_dict
    vector2=[0.0]*len_merge_dict

    # dot_product
    dot_product=0.0
    sum_vector1=0.0
    sum_vector2=0.0
    for key in merge_dict:
        # c1 and c2 are used to check if key is in both vocabs
        c1=0
        c2=0
        if(key in voc1):
            c1=1
            vector1[merge_dict[key]]=voc1[key]
            sum_vector1+= voc1[key]
        if(key in voc2):
            c2=1
            vector2[merge_dict[key]]=voc2[key]
            sum_vector2+= voc2[key]

        #since in a dot product, the only non-zero products happen when both components at that same point are non-zero
        # e.g. [0,2,3] and [1,2,0] - only at second place can a non-zero product occur:
        # meaning that product happen only at shared words
        if(c1==1 and c2==1):
            dot_product+= voc1[key]*voc2[key]

    # returning a list of neccesary items for future calculations
    return [vector1,vector2,magnitude1,magnitude2,dot_product,len_merge_dict,sum_vector1,sum_vector2]


def cosine_similiarity(list_vectors: List) -> float:

    # for any two vectors v1,v2:
    # cosine_similiarity(v1,v2) = dot_product(v1,v2)/(magnitude(v1)*magnitude(v2))
    if(list_vectors[2]*list_vectors[3]==0):
        return 0
    return (list_vectors[4]/(list_vectors[2]*list_vectors[3]))

def pearsons_coefficient(list_vectors: List) -> float:

    # for two vectors v1,v2 of size m,
    # the Pearsons Corelation Coefficient is defined as:
    # C = (m*dot_product(v1,v2) - S(v1)*S(v2)) / (sqrt( (m*magnitude(v1)^2 - S(v1)^2)  * (m*(magnitude(v2))^2 - S(v2)*S(v2)) ))
    # where magnitude(v) is magnitude of vector v, and S(v) is the sum total of the components of v. e.g.  if v is [1,0,3], S(v)=1+0+3
    numerator= (list_vectors[5]*list_vectors[4]) - (list_vectors[6]*list_vectors[7])
    denominator=sqrt( abs((list_vectors[5] * (list_vectors[2]**2) - (list_vectors[6]**2)) * (list_vectors[5] * (list_vectors[3]**2) - (list_vectors[7]**2))) )
    if(denominator==0):
        return 0
    return numerator/denominator

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("doc1", help="The first document input")
    parser.add_argument("doc2", help="Print what's going on to the console.")
    args = parser.parse_args()
    # list of both vectors
    vector_list=doc_vectors(get_sentences_from_document(args.doc1),get_sentences_from_document(args.doc2))

    print("Cosine similiarity between docs is:{}".format(cosine_similiarity(vector_list)))
    print("Pearsons_coefficient between docs is:{}".format(pearsons_coefficient(vector_list)))
