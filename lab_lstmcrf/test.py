import torch
from util import prepare_sequence,ix_to_tag,word_to_ix,test_data,is_entity

train_model=torch.load('./model/train_model_big_200')

def compare(predict_data,label_data):
    match_count=0
    for i in range(len(predict_data)):
        if(predict_data[i]==label_data[i]):
            match_count+=1
    return match_count/len(predict_data);

def get_TP_TN_FP_FN(test_data):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(test_data)):
        precheck_sent = prepare_sequence(test_data[i][0], word_to_ix)
        path_score, state_path = train_model(precheck_sent)
        predict_data = [ix_to_tag[x] for x in state_path]
        label_data=test_data[i][1]
        for j in range(len(predict_data)):
            if (is_entity(label_data[j]) and is_entity(predict_data[j])):
                TP += 1
            elif ((not is_entity(label_data[j])) and (not is_entity(predict_data[j]))):
                TN += 1
            elif ((not is_entity(label_data[j])) and (is_entity(predict_data[j]))):
                FP += 1
            else:
                FN += 1
    return TP,TN,FP,FN


def count_precision_recall_F1(test_data):
    TP, TN, FP, FN=get_TP_TN_FP_FN(test_data)
    precision=TP/(TP+FP) if TP!=0 else 0
    recall=TP/(TP+FN) if TP!=0 else 0
    F1=(2*precision*recall)/(precision+recall) if (precision!=0 and recall!=0) else 0
    return precision,recall,F1

with torch.no_grad():
    # total_correct_count=0
    # for i in range(len(test_data)):
    #     precheck_sent = prepare_sequence(test_data[i][0], word_to_ix)
    #     path_score, state_path = train_model(precheck_sent)
    #     y_pred = [ix_to_tag[x] for x in state_path]
    #     compare_count=compare(y_pred,test_data[i][1])
    #     print(compare_count)
    #     total_correct_count+=compare_count
    precision, recall, F1=count_precision_recall_F1(test_data)
    # print("final count:",total_correct_count/len(test_data))
    print("final average precision:",precision)
    print("final average recall:",recall)
    print("final average F1:",F1)