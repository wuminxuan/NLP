import torch
START_TAG="<start>"
STOP_TAG="<stop>"
from Load_Data import getData

tag_to_ix={"B-ORG":0,"B-LOC":1,"B-MISC":2,"I-ORG":3,"I-MISC":4,"I-PER":5,"I-LOC":6,
           "O":7,START_TAG:8,STOP_TAG:9}
training_data = getData('./data/eng.train')
test_data=getData('./data/eng.testa')
use_cuda=False
word_to_ix = {}

for sentence,tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word]=len(word_to_ix)

for sentence,tags in test_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word]=len(word_to_ix)

ix_to_tag={}
for k,v in tag_to_ix.items():
    ix_to_tag[v]=k


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

def is_entity(word_tag):
    if(word_tag in ["B-ORG","B-LOC","B-MISC","I-ORG","I-MISC","I-PER","I-LOC"]):
        return True
    else:
        return False