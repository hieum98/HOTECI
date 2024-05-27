from collections import OrderedDict, defaultdict
from math import cos
import pdb
from typing import List
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from model.sinkhorn import SinkhornDistance
from transformers import T5Tokenizer, T5EncoderModel


class SentenceSelectOT(nn.Module):
    def __init__(self,
                hidden_size: int,
                OT_eps: float = 0.1,
                OT_max_iter: int = 50,
                OT_reduction: str = 'mean',
                dropout: float = 0.5,
                k: int = 1,
                kg_weight: float = 0.2,
                finetune: bool = True,
                encoder_name_or_path: str = 't5-base',
                tokenizer_name_or_path: str = 't5-base',
                n_selected_sents: int = 5,
                use_rnn: bool = False,
                ):
        super().__init__()
        self.finetune = finetune
        if self.finetune == True:
            self.encoder = T5EncoderModel.from_pretrained(encoder_name_or_path, output_hidden_states=True)
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path)
            self.hidden_size = 768 if 'base' in encoder_name_or_path else 1024
        else:
            self.hidden_size = hidden_size
        self.use_rnn = use_rnn
        if use_rnn:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2 , 2, 
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.filler = nn.Sequential(OrderedDict([
                                    ('linear', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)),
                                    ('ativ_fn', nn.LeakyReLU(0.2))
                                    ]))
        self.sinkhorn = SinkhornDistance(eps=OT_eps, max_iter=OT_max_iter, reduction=OT_reduction)
        self.k = k
        self.kg_weight = kg_weight
        self.dropout = nn.Dropout(dropout)
        self.n_selected_sents = n_selected_sents
        self.cos = nn.CosineSimilarity(dim=1)
    
    def encode_with_rnn(self, inp: torch.Tensor(), ls: List[int]) -> torch.Tensor(): # (batch_size, max_ns, hidden_dim*2)
        packed = pack_padded_sequence(inp, ls, batch_first=True, enforce_sorted=False)
        rnn_encode, _ = self.lstm(packed)
        outp, _ = pad_packed_sequence(rnn_encode, batch_first=True)
        return outp

    def forward(self,
                doc_sentences: List[List[str]],
                doc_embs: List[torch.Tensor], # List of tensors which has size (ns, hidden_size)
                kg_sents: List[List[str]],
                kg_sent_embs: List[torch.Tensor],
                host_ids: List[List[int]], # index of host sentence in document
                context_sentences_scores: List[List[int]], # scores is computed by doc-coref tree
                is_training: bool = True):
        """
        [x]TODO: Add a trainabel filler which will dot product with embedding of X and Y to get new embedding vetor before feed them into Optimal Transport
        TODO: Softmin instead of softmax (encourage sample farer sentence?)
        TODO: Consider chossing k sentences as k consecutive actions 
        [x]TODO: Filler is a (1, hidden_size) tensor instead of linear
        """
        bs = len(host_ids)
        if self.finetune == True:
            doc_embs = []
            kg_sent_embs = []
            for i in range(bs):
                doc_input_ids = self.tokenizer(['</s>' + sent for sent in doc_sentences[i]], 
                                            return_tensors="pt", 
                                            padding='longest',
                                            truncation=True,
                                            ).input_ids
                doc_outputs = self.encoder(input_ids=doc_input_ids.cuda())
                doc_embs.append(doc_outputs.last_hidden_state[:, 0])

                kg_input_ids = self.tokenizer(['</s>' + sent for sent in kg_sents[i]], 
                                            return_tensors="pt",
                                            padding='longest',
                                            truncation=True,).input_ids
                kg_outputs = self.encoder(input_ids=kg_input_ids.cuda())
                kg_sent_embs.append(kg_outputs.last_hidden_state[:, 0])

        ns = [doc_emb.size(0) for doc_emb in doc_embs]
        n_kg_sent = [kg_sent_emb.size(0) for kg_sent_emb in kg_sent_embs]
        if self.use_rnn == True:
            doc_embs = pad_sequence(doc_embs, batch_first=True)
            doc_embs = self.encode_with_rnn(doc_embs, ns)
        else:
            doc_embs = pad_sequence(doc_embs, batch_first=True)
        kg_sent_embs = pad_sequence(kg_sent_embs, batch_first=True)
        doc_embs = self.dropout(doc_embs)
        kg_sent_embs = self.dropout(kg_sent_embs)
        
        X_presentations = []
        Y_presentations = []
        P_X = []
        P_Y = []
        context_ids = []
        host_embs = []
        for i in range(bs):
            _ns = ns[i]
            _n_kg_sent = n_kg_sent[i]
            host_id = host_ids[i]
            context_id = list(set(range(_ns)) - set(host_id))
            context_ids.append(context_id)
            host_id = list(set(host_id))
            host_sentences_emb = doc_embs[i, host_id]
            host_embs.append(host_sentences_emb)
            context_sentences_emb = doc_embs[i, context_id]
            null_presentation = torch.zeros_like(host_sentences_emb[0]).unsqueeze(0)
            X_emb = torch.cat([null_presentation, host_sentences_emb], dim=0)
            X_presentations.append(X_emb)
            context_sentences_emb = torch.cat([context_sentences_emb, kg_sent_embs[i, 0:_n_kg_sent]], dim=0)
            Y_presentations.append(context_sentences_emb)

            if self.k <= len(context_id) // len(host_id):
                k = self.k
            else:
                k = len(context_id) // len(host_id)
            if k == 0:
                k = 1
            X_maginal = torch.tensor([1.0 * k] * len(host_id), dtype=torch.float)
            X_maginal = [torch.tensor([max(0, len(context_id) - k * len(host_id))]), X_maginal]
            X_maginal = torch.cat(X_maginal, dim=0)
            X_maginal = X_maginal / torch.sum(X_maginal)
            P_X.append(X_maginal)
            context_score = context_sentences_scores[i]
            Y_maginal = torch.tensor(context_score, dtype=torch.float)
            Y_maginal = F.softmax(Y_maginal) # farer sentence, higher sample rate 
            Y_maginal = torch.cat([Y_maginal * (1.0 - self.kg_weight), self.kg_weight * torch.tensor([1.0 / _n_kg_sent] * _n_kg_sent, dtype=torch.float)], 
                                dim=0)
            P_Y.append(Y_maginal)
            assert Y_maginal.size(0) == context_sentences_emb.size(0)
            assert X_maginal.size(0) == X_emb.size(0)

        X_presentations = pad_sequence(X_presentations, batch_first=True)
        Y_presentations = pad_sequence(Y_presentations, batch_first=True)
        X_presentations = self.filler(X_presentations)
        Y_presentations = self.filler(Y_presentations)
        P_X = pad_sequence(P_X, batch_first=True)
        P_Y = pad_sequence(P_Y, batch_first=True)

        cost, pi, C = self.sinkhorn(X_presentations, Y_presentations, P_X, P_Y) # pi: (bs, nX, nY)
        values, aligns = torch.max(pi, dim=1)
        selected_sents = []
        selected_sent_embs = []
        mask = torch.zeros_like(pi)
        _pi = pi / (pi.sum(dim=2, keepdim=True) + 1e-10)
        for i in range(bs):
            _selected_sents_with_mapping = {}
            nY = n_kg_sent[i] + len(context_ids[i])
            for j in range(nY):
                if aligns[i, j] != 0:
                    prob = _pi[i, aligns[i, j], j]
                    mapping = (i, aligns[i, j], j)
                    if j < len(context_ids[i]):
                        context_sent_id = context_ids[i][j]
                        context_sent = doc_sentences[i][context_sent_id]
                        selected_sent = (context_sent_id, context_sent)
                        selected_sent_emb = doc_embs[i, context_sent_id]
                    else:
                        kg_sent_id = j - len(context_ids[i])
                        kg_sent = kg_sents[i][kg_sent_id]
                        selected_sent = (9999, kg_sent) # this means we put kg sent in the tail of the augmented doc
                        selected_sent_emb = kg_sent_embs[i, kg_sent_id]
                    if _selected_sents_with_mapping.get(j) != None:
                        if _selected_sents_with_mapping[j][0] < prob:
                            _selected_sents_with_mapping[j] = (prob, mapping, j, selected_sent_emb, selected_sent)
                    else:
                        _selected_sents_with_mapping[j] = (prob, mapping, j, selected_sent_emb, selected_sent)
            
            _selected_sents = []
            _selected_sent_embs = []
            if self.n_selected_sents != None:
                sorted_by_prob = list(_selected_sents_with_mapping.values())
                sorted_by_prob.sort(key=lambda x: x[0], reverse=True)
                for item in sorted_by_prob[:self.n_selected_sents]:
                    if item[0] >= 1e-3:
                        _selected_sents.append(item[-1])
                        _selected_sent_embs.append(item[-2])
                        indicate = item[1]
                        mask[indicate[0], indicate[1], indicate[2]] = 1
            else:
                for item in _selected_sents_with_mapping.values():
                    if item[0] >= 1e-3:
                        _selected_sents.append(item[-1])
                        _selected_sent_embs.append(item[-2])
                        indicate = item[1]
                        mask[indicate[0], indicate[1], indicate[2]] = 1
            selected_sents.append(_selected_sents)
            selected_sent_embs.append(_selected_sent_embs)
        
        log_probs = torch.sum((torch.log(_pi + 1e-10) * mask).view((bs, -1)), dim=-1)
        return cost, torch.mean(log_probs, dim=0), selected_sents


            
