import json
import os.path as osp
import random
import OpenHowNet
from nltk.corpus import wordnet
import morfessor

# OpenHowNet.download()
HOWNET_DICT = OpenHowNet.HowNetDict()


def get_morphemes_model(path):
    io = morfessor.MorfessorIO(encoding="UTF-8",
                               compound_separator=r'\s+',
                               atom_separator=None,
                               lowercase=False)
    return io.read_binary_model_file(path)


def get_lexnames(word):
    lexnames = set()
    for synset in wordnet.synsets(word):
        lexnames.add(synset.lexname())
    return list(lexnames)


def get_sememes(word):
    sememes = HOWNET_DICT.get_sememes_by_word(word, display='list', merge=True, expanded_layer=-1)
    all_sememes = []
    for sememe in sememes:
        all_sememes.append(str(sememe).split("|")[0])
    return all_sememes


def get_morphemes(model, word, viterbiSmooth=0, viterbiMaxLen=30):
    return [affix for affix in model.viterbi_segment(word, viterbiSmooth, viterbiMaxLen)[0] if affix != word]


# one layer down dog -> husky, lab, poodle etc.
def get_hyponyms(word):
    hypos = set()
    for synset in wordnet.synsets(word):
        for hypo in synset.hyponyms():
            hypos.add((hypo.name().split(".")[0].replace("_", " "), hypo.definition()))
    return hypos


# one layer up dog -> animal
def get_hypernyms(word):
    hyps = set()
    for synset in wordnet.synsets(word):
        for hyper in synset.hypernyms():
            hyps.add((hyper.name().split(".")[0].replace("_", " "), hyper.definition()))
    return hyps


# same layer dog -> hound wiener
def get_synonyms(word):
    syns = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            definitions = lemma.synset().definition()
            for definition in definitions.split(";"):
                syns.add((lemma.name().replace("_", " "), definition))
    return syns


def main():
    model = get_morphemes_model("./morphemes/model.bin")
    all_words = json.load(open(osp.join("..", "all_words.json"), "r"))
    all_data = []
    max_exp = 0
    for init_word in all_words:
        print(f'The word is: {init_word} --------------------')
        all_word_def_pairs = set.union(
            get_synonyms(init_word),
            get_hypernyms(init_word),
            get_hyponyms(init_word)
        )
        if len(all_word_def_pairs) > max_exp:
            max_exp = len(all_word_def_pairs)
        for word, definition in all_word_def_pairs:
            data_point = {
                "word": word,
                "lexnames": get_lexnames(word),
                "root_affix": get_morphemes(model, word),
                "sememes": get_sememes(word),
                "definitions": definition
            }
            all_data.append(data_point)

    print(f'siz of atast: {len(all_data)}. \n Max expansion was {max_exp}')
    json.dump(all_data, open(osp.join("..", "data", "data_all_exp.json"), "w"))
    print("created new dataset")


if __name__ == "__main__":
    main()
