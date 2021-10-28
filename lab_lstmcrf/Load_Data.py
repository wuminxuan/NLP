split_tag='\n'

def getData(filepath):
    data = []
    with open(filepath, 'r') as fr:
        word_seq = []
        tag_seq = []
        while (True):
            line = fr.readline()
            if (line == ''):
                data.append((word_seq, tag_seq))
                break
            elif (line == split_tag):
                data.append((word_seq, tag_seq))
                word_seq = []
                tag_seq = []
            else:
                line_split = line.split()
                word_seq.append(line_split[0])
                tag_seq.append(line_split[3])
        return data


