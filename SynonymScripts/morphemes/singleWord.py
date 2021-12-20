import morfessor

io = morfessor.MorfessorIO(encoding="UTF-8",
                     compound_separator=r'\s+',
                     atom_separator=None,
                     lowercase=False)

model = io.read_binary_model_file("model.bin")
word = "devourer"
print(model.viterbi_segment(word, 0, 30)[0])

model