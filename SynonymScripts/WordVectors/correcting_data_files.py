import json
import os.path as osp

data = json.load(open(osp.join("..", "..", "data", "data_syn_aug.json"), "r"))
removals = json.load(open(osp.join("..", "..", "data", "removals.json"), "r"))
updated_data = []
print(len(removals))
for point in data:
    if point["word"] not in removals:
        updated_data.append(point)

print(len(data))
print(len(updated_data))

json.dump(updated_data, open(osp.join("..", "..", "data", "data_syn_aug_clean.json"), "w"))
