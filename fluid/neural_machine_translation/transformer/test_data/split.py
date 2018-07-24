no=0
lines=[]
lines.append([])
lines.append([])
with open("./train.tok.clean.bpe.32000.en-de") as f:
    for line in f:
        lines[no % 2].append(line)
        no+=1

print len(lines[0]), len(lines[1])

with open("./train.tok.clean.bpe.32000.en-de.train_0", "w") as f:
    for line in lines[0]:
        f.write(line)

with open("./train.tok.clean.bpe.32000.en-de.train_1", "w") as f:
    for line in lines[1]:
        f.write(line)
