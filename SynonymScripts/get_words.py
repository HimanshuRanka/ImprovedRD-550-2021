import json
import os.path as osp

data = json.load(open(osp.join("..", "EnglishReverseDictionary", "data", "data_test_500_rand1_unseen.json"), "r", encoding='utf-8'))
words = set()

for pair in data:
    words.add(pair["word"])

words = list(words)
print(len(words))
with open(osp.join("..", "syn_target_words.txt"), "w") as output:
    for word in words:
        output.write(word + "\n")

