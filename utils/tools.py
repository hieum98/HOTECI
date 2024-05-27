
from typing import List
from sklearn.metrics import classification_report, confusion_matrix


def padding(sent, max_sent_len = 194, pad_tok=0):
    one_list = [pad_tok] * max_sent_len # none id 
    one_list[0:len(sent)] = sent
    return one_list


def compute_f1(predicted_seqs: List[str], gold_seqs: List[str], task: str):
    
    preds = []
    golds = []
    if task == 'ECI':
        label_idx={
                        "Yes": 0, 
                        "No": 1, 
                        "wrong construct": 2, 
            }
        for i, (predict, gold) in enumerate(zip(predicted_seqs, gold_seqs)):
            if predict.startswith('No'):
                preds.append('No')
            elif predict.startswith('Yes'):
                preds.append('Yes')
            else:
                preds.append('wrong construct')
            
            if gold.startswith('No'):
                golds.append('No')
            elif gold.startswith('Yes'):
                golds.append('Yes')
            else:
                golds.append('wrong construct')
    else:
        raise RuntimeError(f"{task}. Unsurported task!")
    
    CM = confusion_matrix(golds, preds, labels=list(label_idx.keys()))
    print(f"Confusion Matrix: \n {CM}")
    print(classification_report(golds, preds))
    true = sum([CM[i, i] for i in range(1)])
    sum_pred = sum([CM[i, 0].sum() for i in range(len(label_idx.keys()))])
    sum_gold = sum([CM[i].sum() for i in range(1)])
    P = true / sum_pred if sum_pred != 0 else 0
    R = true / sum_gold if sum_pred != 0 else 0
    F1 = 2 * P * R / (P + R) if P + R != 0 else 0
    return P, R, F1, (P, R, F1)
    