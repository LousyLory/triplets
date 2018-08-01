import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from string import ascii_lowercase
#from src.cnn_ws.string_embeddings.phoc import build_phoc_descriptor, get_most_common_n_grams
#from src.cnn_ws.evaluation import eval_cnn

def visualize_matches(Im, words):
    prob = random.uniform(0.455831, 0.6032201)
    num_valid = int(len(words)*prob)
    num_invalid = len(words) - num_valid
    match = np.ones(len(words))
    Q = np.random.permutation(range(0, len(words)))
    for i in range(0,num_invalid):
        match[Q[i]] = 0

    closest_word = list(range(0, len(words)))
    for i in range(len(words)):
        if match[i] == 0:
            Q =  list(range(0,len(words)))
            Q.remove(i)
            id_ = random.choice(Q)
            closest_word[i] = id_
        else:
            pass
    return match, closest_word

def PHOC(word_strings):
    #word = str(word).lower()

    unigrams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]
    # unigrams = get_unigrams_from_strings(word_strings=[elem[1] for elem in words])
    bigram_levels = None
    bigrams = None

    word_embeddings = build_phoc_descriptor(words=word_strings,
                                                 phoc_unigrams=unigrams,
                                                 bigram_levels=bigram_levels,
                                                 phoc_bigrams=bigrams, unigram_levels=(1,2,4,8))
    print(len(word_embeddings))    
    return word_embeddings

def find_dist(i, embeddings):
    original = embeddings[i]
    Q =  list(range(0, len(embeddings)))
    Q.remove(i)
    init_ = 500000
    for j in Q:
        d = np.linalg.norm(original-embeddings[j])
        if d < init_:
            min_dist_id = j
            init_ = d
    return min_dist_id, init_