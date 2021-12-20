import json
import os.path as osp

data = json.load(open(osp.join("..", "data", "data_all_exp.json"), "r", encoding='utf-8'))
words = set()
print(data[:2])
for pair in data:
    words.add(pair["word"])

words = list(words)
print(len(words))
json.dump(words, open(osp.join("..", "new_words.json"), "w"))