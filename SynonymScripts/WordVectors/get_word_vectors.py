import json
import gensim.downloader as api
import os.path as osp

model = api.load("word2vec-google-news-300")

with open(osp.join("..", "..", "syn_target_words.txt"), "r") as w_files:
    words = w_files.read().splitlines()

vecs = {}
removals = []
for word in words:
    print(f'on word {word}')
    if ' ' in word:
        removals.append(word)
    else:
        try:
            vecs[word] = model[word].tolist()
        except:
            removals.append(word)

print(len(vecs))
print(len(removals))
json.dump(vecs, open(osp.join("..", "..", "data", "new_vecs_inuse.json"), "w"), indent=4)
json.dump(removals, open(osp.join("..", "..", "data", "removals.json"), "w"), indent=4)
