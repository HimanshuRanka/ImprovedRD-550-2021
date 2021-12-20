import json
import os.path as osp
import random
import OpenHowNet
# OpenHowNet.download()
hownet_dict = OpenHowNet.HowNetDict()
import nltk
from nltk.corpus import wordnet
import morfessor


def get_lexnames(word):
    lexnames = set()
    for synset in wordnet.synsets(word):
        lexnames.add(synset.lexname())
    print(lexnames)


def get_sememes(word):
    sememes = hownet_dict.get_sememes_by_word(word, display='list', merge=True, expanded_layer=-1)
    print(sememes)
    all_sememes = []
    for sememe in sememes:
        print(sememe)
        all_sememes.append(str(sememe).split("|")[0])
    print(all_sememes)

def get_morphemes_model(path):
    io = morfessor.MorfessorIO(encoding="UTF-8",
                     compound_separator=r'\s+',
                     atom_separator=None,
                     lowercase=False)
    return io.read_binary_model_file(path)

def get_morphemes(model, word, viterbiSmooth=0, viterbiMaxLen=30):
    print(model.viterbi_segment(word, viterbiSmooth, viterbiMaxLen)[0])


# one layer down dog -> husky, lab, poodle etc.
def get_hyponyms(word):
    hypos = set()
    print(wordnet.synsets(word))
    for synset in wordnet.synsets(word):
        for hypo in synset.hyponyms():
            hypos.add(hypo.name())
    print(hypos)


# one layer up dog -> animal
def get_hypernyms(word):
    hyps = set()
    print(wordnet.synsets(word))
    for synset in wordnet.synsets(word):
        for hyper in synset.hypernyms():
            hyps.add(hyper.name())
    print(hyps)


# same layer dog -> hound wiener
def get_synonyms(word):
    syns = set()
    print(wordnet.synsets(word))
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syns.add(lemma.name())
    print(syns)

model = get_morphemes_model("./morphemes/model.bin")
word = "excess"
get_synonyms(word)
get_hypernyms(word)
get_hyponyms(word)
get_sememes(word)
get_lexnames(word)
get_morphemes(model, word)
