from BiLSTM_CRF import *
import torch.optim as optim
from util import prepare_sequence,get_entity
from Load_Data import getData

EMBEDDING_DIM=8
HIDDEN_DIM=4
tag_to_ix={"B-LOC":0,"B-MISC":1,"I-ORG":2,"I-MISC":3,"I-PER":4,"I-LOC":5,
           "O":6,START_TAG:7,STOP_TAG:8}
training_data = getData('./data/eng.train54019')
test_data=getData('./data/eng.testa')

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
model=BiLSTM_CRF(len(word_to_ix),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)
optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-4)

for epoch in range(300):
    for sentence,tags in training_data[0:3]:
        #1.clear the accumulates gradients out before each instance
        model.zero_grad()

        #2.get inputs ready for the netword,turn them into tensors of word indices
        sentence_in=prepare_sequence(sentence,word_to_ix)
        targets=torch.tensor([tag_to_ix[t] for t in tags],dtype=torch.long)

        #3.run forward pass
        loss=model.neg_log_likelihood(sentence_in,targets)

        #4.compute the loss,gradients and update the parameters
        loss.backward()
        optimizer.step()
    print('epoch:',epoch,'finished')

torch.save(model,'./model/train_model')
train_model=torch.load('./model/train_model')
with torch.no_grad():
    precheck_sent=prepare_sequence(test_data[0][0],word_to_ix)
    path_score,state_path=train_model(precheck_sent)
    y_pred=[ix_to_tag[x] for x in state_path]
    print(y_pred)
    # entity_list=get_entity(test_data[3][0],y_pred)
    # for x in entity_list:
    #     print(x)

