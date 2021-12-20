import json
import gensim.downloader as api
import os.path as osp

model = api.load("word2vec-google-news-300")

words = json.load(open(osp.join("..", "..", "new_words.json"), "r"))
vecs = {}
removals = []
for word in words:
    print(f'on word {word}')
    try:
        vecs[word] = model[word.replace(" ", "_")].tolist()
    except:
        removals.append(word)

print(len(vecs))
print(len(removals))
json.dump(vecs, open(osp.join("..", "..", "data", "new_vecs_inuse.json"), "w"), indent=4)
json.dump(removals, open(osp.join("..", "..", "data", "removals.json"), "w"), indent=4)
