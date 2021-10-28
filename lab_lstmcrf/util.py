import torch
START_TAG="<start>"
STOP_TAG="<stop>"


def argmax(vec):
    _,idx=torch.max(vec,1)
    return idx.item()

#以一种数值稳定性算法求log_sum_exp
def log_sum_exp(vec):
    max_score=vec[0,argmax(vec)]
    max_score_broadcast=max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))

def prepare_sequence(seq,to_ix):
    idxs=[to_ix[w] for w in seq]
    return torch.tensor(idxs,dtype=torch.long)

def get_entity(char_seq, tag_seq):
    length = len(char_seq)
    entity = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B':
            if 'ent' in locals().keys():
                entity.append(ent)
                del ent
            ent = char
            if i + 1 == length:
                entity.append(ent)
        if tag == 'I':
            ent = ent + " " + char
            if i + 1 == length:
                entity.append(ent)
        if tag not in ['B', 'I']:
            if 'ent' in locals().keys():
                entity.append(ent)
                del ent
            continue
    return entity