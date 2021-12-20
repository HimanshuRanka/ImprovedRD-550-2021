import json
import os.path as osp

data = json.load(open(osp.join("..", "data", "data_all_exp_clean.json"), "r", encoding='utf-8'))

lexnames = set()
print(data[:2])
for pair in data:
    for part in pair["lexnames"]:
        lexnames.add(part)

lexnames = list(lexnames)
print(len(lexnames))

with open(osp.join("..", "new_lexnames_all.txt"), "w") as output:
    for lex in lexnames:
        output.write(lex + "\n")

#----------------------------------------------------------------

root_affixes = set()
root_affix_freq = {}
print(data[:2])
for pair in data:
    for part in pair["root_affix"]:
        root_affixes.add(part)
        if part in root_affix_freq:
            root_affix_freq[part] += 1
        else:
            root_affix_freq[part] = 1

root_affixes = list(root_affixes)
print(len(root_affixes))

with open(osp.join("..", "new_root_affixes_all.txt"), "w") as output:
    for lex in root_affixes:
        output.write(lex + "\n")

with open(osp.join("..", "new_root_affixes_freq.txt"), "w") as output:
    for lex in root_affix_freq:
        output.write(lex + " " + str(root_affix_freq[lex]) + "\n")

#----------------------------------------------------------------

sememes = set()
print(data[:2])
for pair in data:
    for part in pair["sememes"]:
        sememes.add(part)

sememes = list(sememes)
print(len(sememes))

with open(osp.join("..", "new_sememes_all.txt"), "w") as output:
    for lex in sememes:
        output.write(lex + "\n")