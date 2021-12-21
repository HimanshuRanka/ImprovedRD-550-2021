import json
import os.path as osp

data = json.load(open(osp.join("..", "EnglishReverseDictionary", "data", "vec_inuse.json"), "r", encoding='utf-8'))
words = data.keys()

print(len(words))
with open(osp.join("..", "syn_target_words.txt"), "w") as output:
    for word in words:
        output.write(word + "\n")

