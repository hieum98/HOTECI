from collections import defaultdict
import re
import requests
from typing import List
from trankit import Pipeline
import networkx as nx
from sentence_transformers import SentenceTransformer, util


def check_sub_str(str, sub_str):
    return sub_str in str


class AtomicRetriever(object):
    def __init__(self,
                kb_path) -> None:
        self.p = Pipeline('english', cache_dir='./trankit')
        # self.sim_evaluator = SentenceTransformer('all-MiniLM-L12-v1')
        self.choosen_rel = ['Causes', 'CausesDesire', 'Desires', 'HasA', 'HasSubEvent',
                            'HinderedBy', 'isAfter', 'isBefore', 'MotivatedByGoal', 'NotDesires',
                            'UsedFor', 'oEffect', 'ReceivesAction', 'PartOf', 'xEffect', 'xIntent,','xReason',]
        self.rel_to_text = {
            'Causes': 'causes', 
            'CausesDesire': 'makes someone want', 
            'Desires': 'desires', 
            'HasA': 'has', 
            'HasSubEvent': 'includes the event',
            'HinderedBy': 'can be hindered by', 
            'isAfter': 'happens after', 
            'isBefore': 'happens before', 
            'MotivatedByGoal': 'is a step towards accomplishing the goal', 
            'NotDesires': 'do not desire',
            'UsedFor': 'uses for', 
            'oEffect': ', as a result, Y or others will', 
            'ReceivesAction': 'can receive or be affected by the action', 
            'PartOf': 'is a part of', 
            'xEffect': ', as a result, PersonX will', 
            'xIntent': 'because PersonX wanted',
            'xReason': 'because', 
        }
        self.kb = self.load_kb(kb_path)
        self.event_to_concept = defaultdict(list)
        self.seq_emb_cache = {}
        
    def load_kb(self, kb_path):
        train = kb_path + 'train.tsv'
        dev = kb_path + 'dev.tsv'
        test = kb_path + 'test.tsv'
        kb_path = [train, dev, test]
        kb = defaultdict(list)
        for split_path in kb_path:
            for line in open(split_path, encoding='UTF-8'):
                triples = line.split('\t')
                if triples[1] in self.choosen_rel and 'none' not in triples[2]:
                    kb[triples[0]].append((triples[1], triples[2]))
        return kb

    def retrive_from_atomic(self, input_seq: List[str], trigger_token_id: List[int], top_k: int=3):
        """Retrieve triplet from Atomic and compute similarity score between triplet and input sentences"""
        k_hop_seq = ' '.join(input_seq)
        event_mention = [input_seq[idx] for idx in trigger_token_id]
        lemmatized_mention = [t['lemma'] for t in self.p.lemmatize(event_mention, is_sent=True)['tokens']]

        knowledge_sents = []
        for m, lemmatized_m in zip(event_mention, lemmatized_mention):
            if self.event_to_concept.get((m, lemmatized_m)) == None:
                triples = []
                for head in self.kb.keys():
                    if m in head or lemmatized_m in head:
                        for (rel, tail) in self.kb[head]:
                            triples.append([head, rel, tail])
                self.event_to_concept[(m, lemmatized_m)] = triples
            else:
                triples = self.event_to_concept[(m, lemmatized_m)]

            for triple in triples:
                knowledge_sent = ' '.join([triple[0], self.rel_to_text[triple[1]], triple[2]]) + ' .'
                if self.seq_emb_cache.get(knowledge_sent) == None:
                    embeddings1 = self.sim_evaluator.encode([knowledge_sent], convert_to_tensor=True)
                    self.seq_emb_cache[knowledge_sent] = embeddings1
                else:
                    embeddings1 = self.seq_emb_cache[knowledge_sent]

                if self.seq_emb_cache.get(k_hop_seq) == None:
                    embeddings2 = self.sim_evaluator.encode([k_hop_seq], convert_to_tensor=True)
                    self.seq_emb_cache[k_hop_seq] = embeddings2
                else:
                    embeddings2 = self.seq_emb_cache[k_hop_seq]

                cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1)
                score = float(cosine_scores[0][0])
                knowledge_sents.append((knowledge_sent, score))
        
        knowledge_sents.sort(key=lambda x: x[1], reverse=True)
        return knowledge_sents[0:top_k]


class ConceptNetRetriever(object):
    def __init__(self) -> None:
        # self.sim_evaluator = SentenceTransformer('all-MiniLM-L12-v1')
        self.rel_to_text  = {
            'CapableOf': 'is capable of', 
            'IsA': 'is a', 
            'Causes': 'causes', 
            'MannerOf': 'is a specific way to do', 
            'CausesDesire': 'makes someone want', 
            'UsedFor': 'uses for', 
            'HasSubevent': 'includes the event', 
            'HasPrerequisite': 'has a precondition of', 
            'NotDesires': 'do not desire', 
            'Entails': 'entails', 
            'ReceivesAction': 'can receive or be affected by the action', 
            'CreatedBy': 'is created by', 
            'Desires': 'desires'
        }
        self.chosen_rel = list(self.rel_to_text.keys())
        # self.p = Pipeline('english', cache_dir='./trankit')
        self.seq_emb_cache = {}
    
    def retrieve_from_conceptnet(self, input_seq: List[str], trigger_token_id: List[int], top_k: int=5):
        """Retrieve triplets/ sample sentences from ConceptNet and compute similarity score between triplet and input sentences"""
        k_hop_seq = ' '.join(input_seq)
        event_mention = [input_seq[idx] for idx in trigger_token_id]
        lemmatized_mention = []
        knowledge_sents = []
        for event in ['_'.join(event_mention), '_'.join(lemmatized_mention)]:
            try:
                obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
            except:
                print(f"Cannot retrieve {event} from ConceptNet!")
                continue
            for e in obj['edges']:
                # print(e)
                if e['start'].get('language') == 'en' and e['end'].get('language') == 'en':
                    if e['rel']['label'] in self.chosen_rel:
                        if e['surfaceText'] != None:
                            knowledge_sent = re.sub(r'[\[\]]','', e['surfaceText'])
                        else:
                            knowledge_sent = ' '.join([e['start']['label'], self.rel_to_text[e['rel']['label']], e['end']['label']])
                        if self.seq_emb_cache.get(knowledge_sent) == None:
                            embeddings1 = self.sim_evaluator.encode([knowledge_sent], convert_to_tensor=True)
                            self.seq_emb_cache[knowledge_sent] = embeddings1
                        else:
                            embeddings1 = self.seq_emb_cache[knowledge_sent]

                        if self.seq_emb_cache.get(k_hop_seq) == None:
                            embeddings2 = self.sim_evaluator.encode([k_hop_seq], convert_to_tensor=True)
                            self.seq_emb_cache[k_hop_seq] = embeddings2
                        else:
                            embeddings2 = self.seq_emb_cache[k_hop_seq]
                        cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1)
                        score = float(cosine_scores[0][0])
                        knowledge_sents.append((knowledge_sent, score))
        
        knowledge_sents.sort(key=lambda x: x[1], reverse=True)
        return knowledge_sents[0:top_k]

                        
                

if __name__ == '__main__':

    sent = ["A", "woman", "has", "been", "arrested", "in", "connection", "with", "the", "murder", "of", "Ciaran", "Noonan."]

    atomic_retriever = AtomicRetriever('datasets/atomic2020_data-feb2021/atomic2020_data-feb2021/')
    atomic_sents = atomic_retriever.retrive_from_atomic(sent, trigger_token_id=[4], top_k=3)

    conceptnet_retriever = ConceptNetRetriever()
    conceptnet_sets = conceptnet_retriever.retrieve_from_conceptnet(sent,trigger_token_id=[4], top_k=5)

    print(f'atomic sents: {atomic_sents}')
    print(f"conceptnet sents: {conceptnet_sets}")


