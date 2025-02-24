filename = "wiki.txt"
dataset = open(filename, "rb")
indexed_lines = open("indexed_lines.txt", "w")

i = -1
pos = 0
line = True

while line:
    line = dataset.readline()

    pos_beg = pos
    pos_end = pos + len(line)
    pos = pos_end

    if line[:17] == "================ ":
        continue
    i += 1

    if pos_end - pos_beg > 100:
        indexed_lines.write(str(pos_beg) + "\t" + str(pos_end) + "\n")

    if i%10000 == 0:
        print(i)
