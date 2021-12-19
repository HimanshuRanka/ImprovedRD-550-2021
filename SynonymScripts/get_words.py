import json
import os.path as osp

data = json.load(open(osp.join("..", "data", "data_train.json"), "r", encoding='utf-8'))
words = set()
print(data[:2])
for pair in data:
    words.add(pair["word"])

words = list(words)
print(len(words))
json.dump(words, open(osp.join("..", "all_words.json"), "w"))