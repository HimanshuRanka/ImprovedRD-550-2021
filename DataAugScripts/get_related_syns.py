import json
import os.path as osp
import random
import OpenHowNet
import tensorflow as tf
import datetime
import tensorflow_hub as hub
from nltk.corpus import wordnet
import morfessor
from cosine_similarity import get_cosine_similarity

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


# same layer dog -> hound wiener
def get_synonyms(word):
    syns = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            definitions = lemma.synset().definition()
            for definition in definitions.split(";"):
                syns.add((lemma.name().replace("_", " "), definition))
    return syns

def get_description_similarity_score(word, definition, embed):
    max_score = 0
    best_idx = 0
    for idx, synset in enumerate(wordnet.synsets(word)):
        score = get_cosine_similarity([definition, synset.definition()], embed)
        # print("============================================")
        # # removed the word that we are looking for as that affected cosine similarity in a not good way
        # print(definition + " \n" + synset.definition())
        # print(score)
        # print("============================================")
        if score > max_score:
            max_score = score
            best_idx = idx
    return max_score, best_idx


def main():
    print("setting up model")
    print(tf.__version__)

    time = datetime.datetime.now()
    embed = hub.load("./new_encoder_model")
    print(datetime.datetime.now() - time)

    print("model setup")
    model = get_morphemes_model("../SynonymScripts/morphemes/model.bin")
    original_data = json.load(open(osp.join("..", "data", "data_train.json"), "r", encoding="utf-8"))
    all_data = []
    for init_data in original_data:
        # print(f'The word is - {init_data["word"]}: {init_data["definitions"]} --------------------')
        word = init_data["word"]
        description = init_data["definitions"]
        score, idx = get_description_similarity_score(word, description, embed)
        # print(f'best similarity score was: {score}')
        syns = set()

        all_data.append(init_data)
        if score >= 0.35:
            for lemma in wordnet.synsets(word)[idx].lemmas():
                syns.add((lemma.name().replace("_", " "), description))

            for word, definition in syns:
                data_point = {
                    "word": word,
                    "lexnames": get_lexnames(word),
                    "root_affix": get_morphemes(model, word),
                    "sememes": get_sememes(word),
                    "definitions": definition
                }
                all_data.append(data_point)

    # print(all_data)

    print(f'siz of atast: {len(all_data)}.')
    print(len(original_data))
    json.dump(all_data, open(osp.join("..", "data", "data_syn_aug.json"), "w"))
    print("created new dataset")


if __name__ == "__main__":
    main()
