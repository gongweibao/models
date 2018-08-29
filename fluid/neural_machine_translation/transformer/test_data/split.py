import sys
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split.py origin_file_name splited_file_nums")
        sys.exit(1)

    origin_file = sys.argv[1]
    splited_file_nums = int(sys.argv[2])

    print("begin split file {} to {} parts".format(origin_file, splited_file_nums))
    split_file(origin_file, splited_file_nums)
