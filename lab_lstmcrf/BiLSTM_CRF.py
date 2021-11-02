import torch.nn as nn
import torch
from util import STOP_TAG,START_TAG,argmax,log_sum_exp,use_cuda

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,tag_to_ix,embedding_dim,hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.vacab_size=vocab_size
        self.tag_to_ix=tag_to_ix
        self.target_size=len(tag_to_ix)
        self.word_embeds=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim//2,num_layers=1,bidirectional=True)

        #将lstm输出映射到tag中
        self.hidden2tag=nn.Linear(hidden_dim,self.target_size)

        #初始化状态转移矩阵[target_size,target_size]
        #T(i,j)表示从状态j转移到状态i的概率,第i行就是所有状态转移到状态i的概率
        self.transitions=nn.Parameter(torch.randn(self.target_size,self.target_size))

        #规定不能其他状态转移到start和从stop转移到其他状态
        self.transitions.data[tag_to_ix[START_TAG],:]=-10000
        self.transitions.data[:,tag_to_ix[STOP_TAG]]=-10000

        self.hidden=self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2,1,self.hidden_dim//2),torch.randn(2,1,self.hidden_dim//2))

    #前向传播
    def forward(self,sentence):
        #获得BiLstm中的emission socres
        lstm_feats=self._get_lstm_features(sentence)

        #用维特比算法找到CRF中的最佳路径
        score,tag_seq=self._viterbi_decode(lstm_feats)
        return score,tag_seq

    def _viterbi_decode(self,feats):
        backpointers=[]

        #初始化
        init_vvars=torch.full((1,self.target_size),-10000)
        init_vvars[0][self.tag_to_ix[START_TAG]]=0

        forward_var=init_vvars

        #先算每一步中的最优路径
        for feat in feats:
            bptrs_t=[] #holds the backpointers for this step
            viterbivars_t=[]

            #遍历一遍下一步的节点，到下一个tag的路径长度
            for next_tag in range(self.target_size):
                #根据定义transitions[next_tag]就是记录的其他的tag转移到next_tag的概率
                next_tag_var=forward_var+self.transitions[next_tag]
                #找到这一步中的每个节点的最大路径长度,然后只保留这一条路径长度为此节点在下一轮计算中的
                #forward_var
                best_tag_id=argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var=(torch.cat(viterbivars_t)+feat).view(1,-1)
            backpointers.append(bptrs_t)

        terminal_var=forward_var+self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id=argmax(terminal_var)

        #到达STOP_TAG的最大分数
        path_score=terminal_var[0][best_tag_id]

        #根据backpointers去回溯最优路径
        best_path=[best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id=bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        #将START_TAG pop出来
        start=best_path.pop()
        assert start==self.tag_to_ix[START_TAG]

        best_path.reverse()
        return path_score,best_path



    def _get_lstm_features(self,sentence):
        self.hidden=self.init_hidden()
        embeds=self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out,self.hidden=self.lstm(embeds,self.hidden)
        lstm_out=lstm_out.view(len(sentence),self.hidden_dim)

        lstm_feats=self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self,sentence,tags):
        feats=self._get_lstm_features(sentence)
        forward_score=self._forward_alg(feats)
        if(use_cuda):
            gold_score=self._score_sentence(feats,tags).cuda()
        else:
            gold_score = self._score_sentence(feats, tags)
        return forward_score-gold_score

    # _forward_alg求出的是损失函数log_sum_exp这一项
    def _forward_alg(self,feats):
        init_alphas=torch.full((1,self.target_size),-10000.)
        #起始状态score定义为0
        init_alphas[0][self.tag_to_ix[START_TAG]]=0.
        if(use_cuda):
            forward_var=init_alphas.cuda()
        else:
            forward_var = init_alphas

        #依次遍历句子中所有词
        #feats:(seq_len,tag_size) LSTM映射到tag space的结果
        for feat in feats:
            #当前时间步的forward tensor
            alphas_t=[]
            #遍历当前时间步的所有可能状态
            for next_tag in range(self.target_size):
                if(use_cuda):
                    emit_score=feat[next_tag].view(1,-1).expand(1,self.target_size).cuda()
                    trans_score = self.transitions[next_tag].view(1, -1).cuda()
                else:
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
                    trans_score = self.transitions[next_tag].view(1, -1)
                #其他状态转移到next_tag(当前状态)的概率


                #计算log_sum_exp之前 状态i到状态next_tag的值
                next_tag_var=forward_var+trans_score+emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var=torch.cat(alphas_t).view(1,-1)

        #最后一步加上最终状态到STOP_TAG的值
        if(use_cuda):
            terminal_var=forward_var+self.transitions[self.tag_to_ix[STOP_TAG]].cuda()
        else:
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        #求出最终log_sum_exp的值
        alpha=log_sum_exp(terminal_var)
        return alpha

    #这里求出损失函数的另一项
    def _score_sentence(self,feats,tags):
        score=torch.zeros(1)

        #加上起始状态
        if(use_cuda):
            tags=torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]),tags])
        else:
            tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])

        for i,feat in enumerate(feats):
            score=score+self.transitions[tags[i+1],tags[i]]+feat[tags[i+1]]
        score=score+self.transitions[self.tag_to_ix[STOP_TAG],tags[-1]]
        return score





