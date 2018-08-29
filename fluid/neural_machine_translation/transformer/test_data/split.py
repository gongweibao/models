def load_line(fname):
    with open(fname) as f:
        for line_no, line in enumerate(f):
            yield line_no, line

def get_fname(fname, fno):
    return "%s_%.3d" % (fname, fno)

def split_file(fname, file_nums):
    fhs = []
    for i in range(0, file_nums):
        fhs.append(None)

    for line_no, line in load_line(fname):
        fno = line_no % file_nums
        if fhs[fno] is None:
            fhs[fno] = open(get_fname(fname, fno), "wb")
        
        fhs[fno].write(line)

    for i in range(0, file_nums):
        fhs[i].close()

split_file("train.tok.clean.bpe.32000.en-de", 36)
