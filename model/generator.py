from collections import OrderedDict
import pdb
from typing import List, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from data_modules.input_formats import INPUT_FORMATS
from data_modules.output_format import OUTPUT_FORMATS
from model.sinkhorn import SinkhornDistance
from utils.tools import padding
from torch.nn.utils.rnn import pad_sequence


class GenOT(nn.Module):
    def __init__(self,
                pretrain_model: str,
                tokenizer: str,
                finetune_in_OT: bool = True,
                OT_eps: float = 0.1,
                OT_max_iter: int = 50,
                OT_reduction: str = 'mean',
                k: int = 5,
                n_selected_words: int = 10,
                output_max_length: int = 32):
        super().__init__()
        self.hidden_size = 768 if 'base' in pretrain_model else 1024
        self.generator = T5ForConditionalGeneration.from_pretrained(pretrain_model)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer, model_max_length=512)
        self.tokenizer_for_generate = T5Tokenizer.from_pretrained(tokenizer, model_max_length=512)
        # when generating, we will use the logits of right-most token to predict the next token
        # so the padding should be on the left
        self.tokenizer_for_generate.padding_side = 'left'
        self.tokenizer_for_generate.pad_token = self.tokenizer_for_generate.eos_token # to avoid an error
        self.finetune_in_OT = finetune_in_OT
        self.filler = nn.Sequential(OrderedDict([
                                    ('linear', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)),
                                    ('ativ_fn', nn.LeakyReLU(0.2))
                                    ]))
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)
        self.k = k
        self.n_selected_words = n_selected_words
        self.output_max_length = output_max_length
        self.cos = nn.CosineSimilarity(dim=0)

    def compute_performance_reward(self, predicted_seqs: List[str], gold_seqs: List[str], task: str):
        if task == 'ECI':
            n_predict = 0
            n_gold = 0
            tp = 0
            wrong_struct = 0
            for predict, gold in zip(predicted_seqs, gold_seqs):
                if predict.startswith('Yes')==False and predict.startswith('No')==False:
                    wrong_struct = wrong_struct + 1
                elif predict.startswith('Yes') and gold.startswith('Yes'):
                    tp = tp + 1
                    
                if predict.startswith('Yes'):
                    n_predict = n_predict + 1
                if gold.startswith('Yes'):
                    n_gold = n_gold + 1

            if wrong_struct == len(predicted_seqs):
                return -1.0
            elif n_predict==n_gold==0:
                return 0.9
            else:
                p = (tp + 1)/(n_predict + 1)
                r = (tp + 1)/(n_gold + 1)
                f1 = 2 * p * r / (p + r + 1e-9)
                return f1
        elif task == 'TRE':
            label_idx={
            "before": 0, 
            "after": 1, 
            "same time": 2, 
            "no relation": 3
            }
        elif task == 'SRE':
            label_idx={
            "including": 0, 
            "a part of": 1, 
            "coreference": 2, 
            "no relation": 3
            }
        n_predict = 0
        n_gold = 0
        tp = 0
        preds = []
        golds = []
        wrong_struct = 0
        for predict, gold in zip(predicted_seqs, gold_seqs):
            for key, item in label_idx.items():
                if predict.startswith(key) and key != 'no relation':
                    n_predict = n_predict + 1
                if gold.startswith(key) and key != 'no relation':
                    n_gold = n_gold + 1
                if predict.startswith(key) and gold.startswith(key) and key != 'no relation':
                    tp = tp + 1

            if not any([predict.startswith(k) for k in list(label_idx.keys())]):
                wrong_struct = wrong_struct + 1
        
        if wrong_struct == len(predicted_seqs):
            return -1.0
        elif n_predict==n_gold==0:
            return 0.9
        else:
            p = (tp + 1)/(n_predict + 1)
            r = (tp + 1)/(n_gold + 1)
            f1 = 2 * p * r / (p + r + 1e-9)
            return f1
    
    def identify_important_words(self, 
                                contexts: List[str],
                                head_positions: List[Tuple[int, int]], # char position
                                tail_positions: List[Tuple[int, int]],
                                task_description_words: List[str],
                                is_training: bool = True,
                                ):
        """
        [x]TODO: Use fixed label vectors  
        TODO: Add prefix ("Find the important words: ")
        [x]TODO: Use uniform for Y_maginal to accelerate model speed
        """
        bs = len(contexts)

        head_str = [contexts[i][head_positions[i][0]: head_positions[i][1]] for i in range(bs)]
        before_head_str = [contexts[i][: head_positions[i][0]] for i in range(bs)]
        tail_str = [contexts[i][tail_positions[i][0]: tail_positions[i][1]] for i in range(bs)]
        before_tail_str = [contexts[i][: tail_positions[i][0]] for i in range(bs)]

        num_head_subwords = [len(ids) - 1 for ids in self.tokenizer(head_str)['input_ids']] # "- 1" means ignoring the </s> in the last 
        num_tail_subwords = [len(ids) - 1 for ids in self.tokenizer(tail_str)['input_ids']] # "- 1" means ignoring the </s> in the last 
        num_before_head_subwords = [len(ids) - 1 for ids in self.tokenizer(before_head_str)['input_ids']] # "- 1" means ignoring the </s> in the last 
        num_before_tail_subwords = [len(ids) - 1 for ids in self.tokenizer(before_tail_str)['input_ids']] # "- 1" means ignoring the </s> in the last 
        head_subword_ids = [[num_before_head_subwords[i], num_before_head_subwords[i] + num_head_subwords[i]] for i in range(bs)]
        tail_subword_ids = [[num_before_tail_subwords[i], num_before_tail_subwords[i] + num_tail_subwords[i]] for i in range(bs)]

        tokenized_input = self.tokenizer(contexts)['input_ids']
        ns = []
        for sent_ids in tokenized_input:
            ns.append(len(sent_ids) - 1)
        max_ns = max(ns)
        padded_tokenized_input = [padding(sent_ids, max_sent_len=max_ns+1, pad_tok=self.tokenizer.pad_token_id) for sent_ids in tokenized_input]
        padded_tokenized_input = torch.tensor(padded_tokenized_input).cuda()
        
        if self.finetune_in_OT:
            outputs = self.generator.encoder(input_ids=padded_tokenized_input, output_hidden_states=True)
            ouputs_last_hidden_state = outputs.last_hidden_state

            task_description_tokenized = self.tokenizer(task_description_words, return_tensors="pt", padding='longest')
            task_description_embs = self.generator.encoder(input_ids=task_description_tokenized.input_ids.cuda(), 
                                                            output_hidden_states=True)
            task_description_embs = torch.max(task_description_embs.last_hidden_state, dim=1)[0] # (n_words, hidden_size)
        else:
            with torch.no_grad():
                outputs = self.generator.encoder(input_ids=padded_tokenized_input, output_hidden_states=True)
                ouputs_last_hidden_state = outputs.last_hidden_state

                task_description_tokenized = self.tokenizer(task_description_words, return_tensors="pt", padding='longest')
                task_description_embs = self.generator.encoder(input_ids=task_description_tokenized.input_ids.cuda(), 
                                                                output_hidden_states=True)
                task_description_embs = torch.max(task_description_embs.last_hidden_state, dim=1)[0] # (n_words, hidden_size)
                
        X_presentations = []
        Y_presentations = []
        P_X = []
        P_Y = []
        context_ids = []
        trigger_embs = []
        for i in range(bs):
            _ns = ns[i]
            trigger_id = list(range(head_subword_ids[i][0], head_subword_ids[i][1])) + list(range(tail_subword_ids[i][0], tail_subword_ids[i][1]))
            context_id = list(set(range(_ns)) - set(trigger_id))
            context_ids.append(context_id)
        
            trigger_id = list(set(trigger_id))
            trigger_emb = ouputs_last_hidden_state[i, trigger_id]
            head_emb = torch.max(ouputs_last_hidden_state[i, list(range(head_subword_ids[i][0], head_subword_ids[i][1]))], dim=0)[0]
            tail_emb = torch.max(ouputs_last_hidden_state[i, list(range(tail_subword_ids[i][0], tail_subword_ids[i][1]))], dim=0)[0]
            trigger_embs.append([head_emb, tail_emb])
            context_emb = ouputs_last_hidden_state[i, context_id]
            null_presentation = torch.zeros_like(trigger_emb[0]).unsqueeze(0)
            X_emb = torch.cat([null_presentation, trigger_emb, task_description_embs])
            X_presentations.append(X_emb)
            Y_presentations.append(context_emb)

            if self.k <= len(context_id) // (len(trigger_id) + len(task_description_words)):
                k = self.k
            else:
                k = len(context_id) // (len(trigger_id) + len(task_description_words))
            if k == 0:
                k = 1
            X_maginal = torch.tensor([1.0 * k] * (len(trigger_id) + len(task_description_words)), dtype=torch.float)
            X_maginal = [torch.tensor([max(0, len(context_id) - k * (len(trigger_id) + len(task_description_words)))]), X_maginal]
            X_maginal = torch.cat(X_maginal, dim=0)
            X_maginal = X_maginal / torch.sum(X_maginal)
            P_X.append(X_maginal)
            Y_maginal = [min([abs(idx - trigger_idx) for trigger_idx in trigger_id]) for idx in context_id]
            Y_maginal = torch.tensor(Y_maginal, dtype=torch.float)
            Y_maginal = F.softmax(Y_maginal) # farer word, higher sample rate 
            P_Y.append(Y_maginal)
            assert Y_maginal.size(0) == context_emb.size(0)
            assert X_maginal.size(0) == X_emb.size(0)
        

        X_presentations = pad_sequence(X_presentations, batch_first=True)
        Y_presentations = pad_sequence(Y_presentations, batch_first=True)
        X_presentations = self.filler(X_presentations)
        Y_presentations = self.filler(Y_presentations)
        P_X = pad_sequence(P_X, batch_first=True)
        P_Y = pad_sequence(P_Y, batch_first=True)

        cost, pi, C = self.sinkhorn(X_presentations, Y_presentations, P_X, P_Y) # pi: (bs, nX, nY)
        values, aligns = torch.max(pi, dim=1)
        selected_words = []
        mask = torch.zeros_like(pi)
        _pi = pi / (pi.sum(dim=2, keepdim=True) + 1e-10)
        for i in range(bs):
            _selected_words_with_mapping = {}
            nY = len(context_ids[i])
            for j in range(nY):
                if aligns[i, j] != 0:
                    prob = _pi[i, aligns[i, j], j]
                    mapping = (i, aligns[i, j], j)
                    context_word_id = context_ids[i][j]
                    context_word = self.tokenizer.decode([tokenized_input[i][context_word_id]])
                    selected_word = (context_word_id, context_word)
                    if _selected_words_with_mapping.get(j) != None:
                        if _selected_words_with_mapping[j][0] < prob:
                            _selected_words_with_mapping[j] = (prob, mapping, j, selected_word)
                    else:
                        _selected_words_with_mapping[j] = (prob, mapping, j, selected_word)
            _selected_words = []
            if self.n_selected_words != None:
                sorted_by_prob = list(_selected_words_with_mapping.values())
                sorted_by_prob.sort(key=lambda x: x[0], reverse=True)
                for item in sorted_by_prob[:self.n_selected_words]:
                    if item[0] >= 1e-3:
                        _selected_words.append(item[-1])
                        indicate = item[1]
                        mask[indicate[0], indicate[1], indicate[2]] = 1
            else:
                for item in _selected_words_with_mapping.values():
                    if item[0] >= 1e-3:
                        _selected_words.append(item[-1])
                        indicate = item[1]
                        mask[indicate[0], indicate[1], indicate[2]] = 1
            _selected_words.extend([(num_before_head_subwords[i], head_str[i]), (num_before_tail_subwords[i], tail_str[i])])
            selected_words.append(_selected_words)
            
        log_probs = torch.sum((torch.log(_pi + 1e-10) * mask).view((bs, -1)), dim=-1)
        return cost, torch.mean(log_probs, dim=0), selected_words, trigger_embs, num_before_head_subwords, num_before_tail_subwords

    def forward(self,
                task: str,
                input_format_type: str,
                output_format_type: str,
                contexts: List[str],
                head_positions: List[Tuple[int, int]], # char position
                tail_positions: List[Tuple[int, int]],
                task_description_words: List[str],
                labels: List[str],
                head_sentences: List[str],
                head_pos_in_sent: List[Tuple[int, int]],
                tail_sentences: List[str],
                tail_pos_in_sent: List[Tuple[int, int]],
                is_training: bool = True,
                is_warm_up: bool = False,
                ):
        """
        TODO: Add dep-path in the generating sequences
        """
        input_formater = INPUT_FORMATS[input_format_type]()
        output_formater = OUTPUT_FORMATS[output_format_type]()
        log_important_words = []
        cost, log_probs, selected_words, trigger_embs, \
        num_before_head_subwords, num_before_tail_subwords = self.identify_important_words(contexts=contexts,
                                                                                        head_positions=head_positions,
                                                                                        tail_positions=tail_positions,
                                                                                        task_description_words=task_description_words,
                                                                                        is_training=is_training)
                    
        bs = len(contexts)
        inputs = []
        outputs = []
        head_strs = []
        tail_strs = []
        for i in range(bs):
            if len(selected_words[i]) > 2:
                log_important_words.append(selected_words[i])
            else:
                log_important_words.append(None)
            input_txt, head_str, tail_str = input_formater.format_input(context=contexts[i], head_position=head_positions[i], tail_position=tail_positions[i])
            inputs.append(input_txt)
            outpt = output_formater.format_output(important_words=selected_words[i], head=head_str, tail=tail_str, label=labels[i],
                                                num_before_head_subword=num_before_head_subwords[i], 
                                                num_before_tail_subword=num_before_tail_subwords[i])
            outputs.append(outpt)
            head_strs.append(head_str)
            tail_strs.append(tail_str)

        if is_training:
            input_encoded = self.tokenizer(inputs,
                                        padding='longest',
                                        truncation=True,
                                        return_tensors="pt")
            output_encoded = self.tokenizer(outputs,
                                            padding='longest',
                                            truncation=True,
                                            return_tensors="pt")
            output_ids = output_encoded.input_ids
            output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100.

            model_output = self.generator(input_ids=input_encoded.input_ids.cuda(), 
                                        attention_mask=input_encoded.attention_mask.cuda(), 
                                        labels=output_ids.cuda(),
                                        output_hidden_states=True)
            
            mle_loss = model_output.loss
            logits = model_output.logits
            predicted_seq = torch.max(logits, dim=2)[1] * output_encoded.attention_mask.cuda() # this means model only concern about the true output. 
            predicted_seq = self.tokenizer.batch_decode(predicted_seq, skip_special_tokens=True)
        else:
            mle_loss = 0.0
            input_encoded = self.tokenizer_for_generate(inputs,
                                        padding='longest',
                                        truncation=True,
                                        return_tensors="pt")
            predicted_seq = self.generator.generate(input_ids=input_encoded['input_ids'].cuda(),
                                                    attention_mask=input_encoded['attention_mask'].cuda(),
                                                    do_sample=False, # disable sampling to test if batching affects output
                                                    top_k=20, 
                                                    top_p=0.95, 
                                                    max_length=self.output_max_length, 
                                                    num_return_sequences=1, 
                                                    num_beams=8,)
            predicted_seq = self.tokenizer.batch_decode(predicted_seq, skip_special_tokens=True)
        
        performance_reward = self.compute_performance_reward(predicted_seqs=predicted_seq, gold_seqs=outputs, task=task)
        
        return cost, log_probs, mle_loss, predicted_seq, outputs, performance_reward, log_important_words, inputs


