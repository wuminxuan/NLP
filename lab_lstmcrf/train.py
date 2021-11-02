from BiLSTM_CRF import *
import torch.optim as optim
from util import prepare_sequence,word_to_ix,tag_to_ix,\
    training_data,use_cuda
import datetime


EMBEDDING_DIM=10
HIDDEN_DIM=4

model=BiLSTM_CRF(len(word_to_ix),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)
optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-4)

for epoch in range(200):
    now_time=datetime.datetime.now()
    for sentence,tags in training_data:
        #1.clear the accumulates gradients out before each instance
        model.zero_grad()
        #2.get inputs ready for the netword,turn them into tensors of word indices
        sentence_in=prepare_sequence(sentence,word_to_ix)
        if (use_cuda):
            targets=torch.cuda.LongTensor([tag_to_ix[t] for t in tags])
        else:
            targets=torch.LongTensor([tag_to_ix[t] for t in tags])

        #3.run forward pass
        loss=model.neg_log_likelihood(sentence_in,targets)

        #4.compute the loss,gradients and update the parameters
        loss.backward()
        optimizer.step()
    end_time = datetime.datetime.now()
    print('epoch:',epoch+1,'finished,use time:',end_time-now_time)

torch.save(model,'./model/train_model_big_200')

