from collections import defaultdict
import enum
from itertools import combinations
import re
import networkx as nx
import torch
from data_modules.utils import tokenized_to_origin_span
from data_modules.retriever import AtomicRetriever, ConceptNetRetriever

DATAPOINT = {}


def register_datapoint(func):
    DATAPOINT[str(func.__name__)] = func
    return func


def get_datapoint(type, mydict):
    return DATAPOINT[type](mydict)


@register_datapoint
def hoteere_data_point(my_dict):
    """
    Data point format for HOTEERE model
    """
    data_points = []

    doc_sentences = []
    doc_heads = []
    for sent in my_dict['sentences']:
        doc_sentences.append(sent['content'])
        doc_heads.append(sent['heads'])
    doc_content = my_dict['doc_content']
    sent_spans = [(sent['sent_start_char'], sent['sent_end_char']) for sent in my_dict['sentences']]

    doc_graph = nx.Graph()
    sent_id_pair = combinations(range(len(doc_sentences)), 2)
    edges = []
    for (sid1, sid2) in sent_id_pair:
        if abs(sid1 - sid2) == 1:
            edges.append((sid1, sid2))
        for cluster in my_dict['coref']:
            if sid1 in cluster['sentences'] and sid2 in cluster['sentences']:
                edges.append((sid1, sid2))
    doc_graph.add_edges_from(edges)
    spl = dict(nx.all_pairs_shortest_path_length(doc_graph))
    
    for key, val in my_dict['relation_dict'].items():
        eid1, eid2 = re.sub('\W+','', key.split(',')[0].strip()), re.sub('\W+','', key.split(',')[1].strip())
        if my_dict['event_dict'].get(eid1) == None:
            e1 = my_dict['event_dict'].get(int(eid1))
        else:
            print(f"There is no {eid1} in list event!")
            continue

        if my_dict['event_dict'].get(eid2) == None:
            e2 = my_dict['event_dict'].get(int(eid2))
        else:
            print(f"There is no {eid2} in list event!")
            continue
        
        sid1, sid2 = e1['sent_id'], e2['sent_id']
        e1_span = (e1['start_char'] - my_dict['sentences'][sid1]['sent_start_char'], e1['end_char'] - my_dict['sentences'][sid1]['sent_start_char'])
        assert doc_sentences[sid1][e1_span[0]: e1_span[1]] == e1['mention'], \
                        f"{doc_sentences[sid1][e1_span[0]: e1_span[1]]} - {e1['mention']}"
        e2_span = (e2['start_char'] - my_dict['sentences'][sid2]['sent_start_char'], e2['end_char'] - my_dict['sentences'][sid2]['sent_start_char'])
        assert doc_sentences[sid2][e2_span[0]: e2_span[1]] == e2['mention'], \
                        f"{doc_sentences[sid2][e2_span[0]: e2_span[1]]} - {e2['mention']}"
        
        knowledge_sentences = list(set(list(e1['knowledge_sentences'].keys()) + list(e2['knowledge_sentences'].keys())))
        kg_sent_embs = []
        for kg_sent in knowledge_sentences:
            if e1['knowledge_sentences'].get(kg_sent) != None:
                kg_sent_embs.append(e1['knowledge_sentences'].get(kg_sent))
            else:
                kg_sent_embs.append(e2['knowledge_sentences'].get(kg_sent))
        assert len(kg_sent_embs) == len(knowledge_sentences)
        kg_sent_embs = torch.stack(kg_sent_embs, dim=0)
        # print(f"kg_sent_embs: {kg_sent_embs.size()}")
        ctx_sids = list(set(range(len(doc_sentences))) - set([sid1, sid2]))
        score_over_doc_tree = {
            sid1: [spl[sid1][sid] for sid in ctx_sids],
            sid2: [spl[sid2][sid] for sid in ctx_sids],
            'score': [min([spl[sid1][sid], spl[sid2][sid]]) for sid in ctx_sids]
        } 

        data_point = {
            'doc_content': doc_content,
            'doc_presentation': my_dict['doc_presentation'],
            'sentences_context': doc_sentences,
            'sentence_spans': sent_spans,
            'sentences_tokens': [sent['tokens'] for sent in my_dict['sentences']],
            'sentences_pos': [sent['pos'] for sent in my_dict['sentences']],
            'sentences_heads': [sent['heads'] for sent in my_dict['sentences']],
            'sentences_token_span_sent': [sent['token_span_SENT'] for sent in my_dict['sentences']], # need to +1
            'sentences_token_span_doc': [sent['token_span_DOC'] for sent in my_dict['sentences']], # need to +1
            'coref': my_dict['coref'],
            'score_over_doc_tree': score_over_doc_tree,
            'kg_sentences': knowledge_sentences,
            'kg_sent_embs': kg_sent_embs,
            'trigger1': {
                'mention': e1['mention'],
                'sent_span': e1_span,
                'sent_id': sid1,
                'token_list': e1['token_id']
            },
            'trigger2': {
                'mention': e2['mention'],
                'sent_span': e2_span,
                'sent_id': sid2,
                'token_list': e2['token_id']
            },
            'label': val
        }
        data_points.append(data_point)
    
    return data_points
