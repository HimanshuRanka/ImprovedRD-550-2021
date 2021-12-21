import json
import os.path as osp

data = json.load(open(osp.join("..", "..", "data", "new_vecs_inuse.json"), "r"))
data2 = json.load(open(osp.join("..", "..", "EnglishReverseDictionary", "data", "vec_inuse.json"), "r"))

data2.update(data)

json.dump(data2, open(osp.join("..", "..", "EnglishReverseDictionary", "data", "vec_inuse.json"), "w"), indent=4)
