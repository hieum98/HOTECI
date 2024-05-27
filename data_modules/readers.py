from itertools import combinations
import json
import pdb
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
# from nltk import sent_tokenize
from bs4 import BeautifulSoup as Soup
import csv
# from trankit import Pipeline
from data_modules.utils import find_m_id, find_sent_id, get_mention_span, id_lookup, sent_id_lookup, span_SENT_to_DOC, tokenized_to_origin_span
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging

# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

# p = Pipeline('english', cache_dir='./trankit')
# p.add('danish')
# p.add('spanish')
# p.add('turkish')
# p.add('urdu')


# =========================
#       HiEve Reader
# =========================
def tsvx_reader(dir_name, file_name):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}

    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = line[1]
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2])
            my_dict["event_dict"][int(line[1])] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = int(line[1])
            event_id2 = int(line[2])
            rel = line[3]
            my_dict["relation_dict"][f"{(event_id1, event_id2)}"] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # add NoRel 
    # event_pairs = combinations(my_dict['event_dict'].keys(), 2)
    # for eid1, eid2 in event_pairs:
    #     e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
    #     sid1, sid2 = e1['sent_id'], e2['sent_id']
    #     if sid1 == sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
    #         my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'

    # Split document into sentences
    sents_tokenized = p.ssplit(my_dict["doc_content"])['sentences']
    for sent in sents_tokenized:
        sent_dict = {}
        sent_dict["sent_id"] = sent['id'] - 1
        sent_dict["content"] = sent['text']
        sent_dict["sent_start_char"] = sent['dspan'][0]
        sent_dict["sent_end_char"] = sent['dspan'][1] + 1 # because trankit format

        # Tokenized, Part-Of-Speech Tagging, Dependency Parsing
        parsed_tokens = p.posdep(sent_dict['content'].split(), is_sent=True)['tokens']
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        for token in parsed_tokens:
            sent_dict['tokens'].append(token['text'])
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        if sent_id == None:
            print("False to find sent_id")
            print(f'mydict: {my_dict}')
            print(f"event: {event_dict}")
            continue
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        if not all(tok in  my_dict["event_dict"][event_id]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]):
            print(f'{my_dict["event_dict"][event_id]["mention"]} - \
            {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]}')
            print(f'{my_dict["event_dict"][event_id]}  - {my_dict["sentences"][sent_id]}')
    
    doc_content = my_dict['doc_content']
    sent_spans = [(sent['sent_start_char'], sent['sent_end_char']) for sent in my_dict['sentences']]
    corefed_doc = predictor.predict(document=doc_content)
    doc_tokens = corefed_doc['document']
    token_span_doc = tokenized_to_origin_span(doc_content, doc_tokens)
    coref_clusters = []
    for i, c in enumerate(corefed_doc['clusters']):
        cluster = defaultdict(list)
        for m in c:
            mention_span = (token_span_doc[m[0]][0], token_span_doc[m[-1]][-1])
            cluster['mentions'].append(mention_span)
            for j, (start, end) in enumerate(sent_spans):
                if start <= mention_span[0] and end >= mention_span[1]:
                    cluster['sentences'].append(j)
        coref_clusters.append(dict(cluster))
    my_dict['coref'] = coref_clusters
    
    return my_dict


# ========================================
#        MATRES: read relation file
# ========================================
# MATRES has separate text files and relation files
# We first read relation files
eiid_to_event_trigger = {}
eiid_pair_to_label = {} 

def matres_reader(matres_file):
    with open(matres_file, 'r', encoding='UTF-8') as f:
        content = f.read().split('\n')
        for rel in content:
            rel = rel.split("\t")
            fname = rel[0]
            trigger1 = rel[1]
            trigger2 = rel[2]
            eiid1 = int(rel[3])
            eiid2 = int(rel[4])
            tempRel = rel[5]

            if fname not in eiid_to_event_trigger:
                eiid_to_event_trigger[fname] = {}
                eiid_pair_to_label[fname] = {}
            eiid_pair_to_label[fname][f"{(eiid1, eiid2)}"] = tempRel
            if eiid1 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid1] = trigger1
            if eiid2 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid2] = trigger2

MATRES_timebank = './datasets/MATRES/timebank.txt'
MATRES_aquaint = './datasets/MATRES/aquaint.txt'
MATRES_platinum = './datasets/MATRES/platinum.txt'
matres_reader(MATRES_timebank)
matres_reader(MATRES_aquaint)
matres_reader(MATRES_platinum)


# =========================
#       MATRES Reader
# =========================
def tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eID_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".tml", "") 
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'
    tree = ET.parse(dir_name + file_name)
    root = tree.getroot()
    MY_STRING = str(ET.tostring(root))
    # ================================================
    # Load the lines involving event information first
    # ================================================
    for makeinstance in root.findall('MAKEINSTANCE'):
        instance_str = str(ET.tostring(makeinstance)).split(" ")
        try:
            assert instance_str[3].split("=")[0] == "eventID"
            assert instance_str[2].split("=")[0] == "eiid"
            eiid = int(instance_str[2].split("=")[1].replace("\"", "")[2:])
            eID = instance_str[3].split("=")[1].replace("\"", "")
        except:
            for i in instance_str:
                if i.split("=")[0] == "eventID":
                    eID = i.split("=")[1].replace("\"", "")
                if i.split("=")[0] == "eiid":
                    eiid = int(i.split("=")[1].replace("\"", "")[2:])
        # Not all document in the dataset contributes relation pairs in MATRES
        # Not all events in a document constitute relation pairs in MATRES
        if my_dict["doc_id"] in eiid_to_event_trigger.keys():
            if eiid in eiid_to_event_trigger[my_dict["doc_id"]].keys():
                my_dict["event_dict"][eiid] = {"eID": eID, "mention": eiid_to_event_trigger[my_dict["doc_id"]][eiid]}
                my_dict["eID_dict"][eID] = {"eiid": eiid}
        
    # ==================================
    #              Load Text
    # ==================================
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")
    
    # ========================================================
    #    Load position of events, in the meantime replacing 
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            # print(event_description)
            # eID = (event_description[1].split("="))[1].replace("\"", "")
            # print(eID)
            for item in event_description:
                if item.startswith("eid"):
                    eID = (item.split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in my_dict["eID_dict"].keys():
                eiid = my_dict['eID_dict'][eID]['eiid']
                my_dict["event_dict"][eiid]["start_char"] = start # loading position of events
                end_char = start + len(my_dict["event_dict"][eiid]['mention'])
                my_dict['event_dict'][eiid]['end_char'] = end_char
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
    
    # =====================================
    # Enter the routine for text processing
    # =====================================
    my_dict["doc_content"] = MY_TEXT
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    sents_tokenized = p.ssplit(my_dict["doc_content"])['sentences']
    for sent in sents_tokenized:
        sent_dict = {}
        sent_dict["sent_id"] = sent['id'] - 1
        sent_dict["content"] = sent['text']
        sent_dict["sent_start_char"] = sent['dspan'][0]
        sent_dict["sent_end_char"] = sent['dspan'][1] + 1 # because trankit format

        # Tokenized, Part-Of-Speech Tagging, Dependency Parsing
        parsed_tokens = p.posdep(sent_dict['content'].split(), is_sent=True)['tokens']
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        for token in parsed_tokens:
            sent_dict['tokens'].append(token['text'])
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)
    
    for event_id, event_dict in my_dict["event_dict"].items():
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"]]).lower() == str(event_dict["mention"]).lower()
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"]]).strip() != ""
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        if sent_id == None:
            print("False to find sent_id")
            print(f'mydict: {my_dict}')
            print(f"event: {event_dict}")
            continue
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        if not all(tok in  my_dict["event_dict"][event_id]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]):
            print(f'{my_dict["event_dict"][event_id]["mention"]} - \
            {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]}')
            print(f'{my_dict["event_dict"][event_id]}  - {my_dict["sentences"][sent_id]}')
    
    if eiid_pair_to_label.get(my_dict['doc_id']) == None:
        return None
    relation_dict = eiid_pair_to_label[my_dict['doc_id']]
    my_dict['relation_dict'] = relation_dict

    doc_content = my_dict['doc_content']
    sent_spans = [(sent['sent_start_char'], sent['sent_end_char']) for sent in my_dict['sentences']]
    corefed_doc = predictor.predict(document=doc_content)
    doc_tokens = corefed_doc['document']
    token_span_doc = tokenized_to_origin_span(doc_content, doc_tokens)
    coref_clusters = []
    for i, c in enumerate(corefed_doc['clusters']):
        cluster = defaultdict(list)
        for m in c:
            mention_span = (token_span_doc[m[0]][0], token_span_doc[m[-1]][-1])
            cluster['mentions'].append(mention_span)
            for j, (start, end) in enumerate(sent_spans):
                if start <= mention_span[0] and end >= mention_span[1]:
                    cluster['sentences'].append(j)
        coref_clusters.append(dict(cluster))
    my_dict['coref'] = coref_clusters

    return my_dict


# =========================
#       ESC Reader
# =========================
def cat_xml_reader(dir_name, file_name, intra=True, inter=False):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')

    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None
    
    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['t_id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['content'] = ' '.join(v)
        sent_dict["sent_start_char"] = start
        sent_dict["sent_end_char"] = end = start + len(sent_dict['content'])
        assert sent_dict['content'] == my_dict['doc_content'][start: end]
        
        sent_dict['tokens'] = v
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        sent_dict['pos'] = []
        parsed_tokens = p.posdep(sent_dict['tokens'], is_sent=True)['tokens']
        for token in parsed_tokens:
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(f"This doc {my_dict['doc_id']} was not annotated!")
        return None
    
    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'action' in item.name:
            eid = int(item.attrs['m_id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['t_id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['start_char'], my_dict['event_dict'][eid]['end_char'] = start, start + len(mention)
                my_dict['event_dict'][eid]['token_id'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = sent_id = find_sent_id(my_dict['sentences'], [start, start + len(mention)])
                if not all(tok in  my_dict["event_dict"][eid]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]):
                    print(f'{my_dict["event_dict"][eid]["mention"]} - \
                    {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]}')
                    print(f'{my_dict["event_dict"][eid]}  - {my_dict["sentences"][sent_id]}')
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    event_pairs = list(combinations(my_dict['event_dict'].keys(), 2))
    if intra==True:
        for item in xml_dom.find('relations').children:
            if type(item)== bs4.element.Tag and 'plot_link' in item.name:
                r_id = item.attrs['r_id']
                if item.has_attr('signal'):
                    signal = item.attrs['signal']
                else:
                    signal = ''
                try:
                    r_typ = item.attrs['reltype']
                except:
                    # print(my_dict['doc_id'])
                    # print(item)
                    continue
                cause = item.attrs['causes']
                cause_by = item.attrs['caused_by']
                head = int(item.find('source').attrs['m_id'])
                tail = int(item.find('target').attrs['m_id'])

                assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
                my_dict['relation_dict'][f"{(head, tail)}"] = r_typ

        # Add Norel into data
        for eid1, eid2 in event_pairs:
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            sid1, sid2 = e1['sent_id'], e2['sent_id']
            if sid1 == sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
                my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'
                
    event_pairs = list(combinations(my_dict['event_dict'].keys(), 2))
    if inter==True:
        dir_name = './datasets/EventStoryLine/annotated_data/v0.9/'
        inter_dir_name = dir_name.replace('annotated_data', 'evaluation_format/full_corpus') + 'event_mentions_extended/'
        file_name = file_name.replace('.xml.xml', '.xml')
        lines = []
        try:
            with open(inter_dir_name+file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            print("{} is not exit!".format(inter_dir_name+file_name))
        for line in lines:
            rel = line.strip().split('\t')
            r_typ = rel[2]
            head_span, tail_span = get_mention_span(rel[0]), get_mention_span(rel[1])
            # print(head_span, tail_span)
            head, tail = find_m_id(head_span, my_dict['event_dict']), find_m_id(tail_span, my_dict['event_dict'])
            assert head != None and tail != None, f"doc: {inter_dir_name+file_name}, line: {line}, rel: {rel}"

            if r_typ != 'null':
                my_dict['relation_dict'][f"{(head, tail)}"] = r_typ

        # Add Norel into data
        for eid1, eid2 in event_pairs:
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            sid1, sid2 = e1['sent_id'], e2['sent_id']
            if sid1 != sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
                my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'
    
    doc_content = my_dict['doc_content']
    sent_spans = [(sent['sent_start_char'], sent['sent_end_char']) for sent in my_dict['sentences']]
    corefed_doc = predictor.predict(document=doc_content)
    doc_tokens = corefed_doc['document']
    token_span_doc = tokenized_to_origin_span(doc_content, doc_tokens)
    coref_clusters = []
    for i, c in enumerate(corefed_doc['clusters']):
        cluster = defaultdict(list)
        for m in c:
            mention_span = (token_span_doc[m[0]][0], token_span_doc[m[-1]][-1])
            cluster['mentions'].append(mention_span)
            for j, (start, end) in enumerate(sent_spans):
                if start <= mention_span[0] and end >= mention_span[1]:
                    cluster['sentences'].append(j)
        coref_clusters.append(dict(cluster))
    my_dict['coref'] = coref_clusters
    
    return my_dict


# =========================
#     Causal-TB Reader
# =========================
def ctb_cat_reader(dir_name, file_name):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')
    
    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None
    
    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        # sent_dict['token_span_doc'] = _sent_token_span_doc[k] # from 1
        sent_dict['content'] = ' '.join(v)
        sent_dict["sent_start_char"] = start
        sent_dict["sent_end_char"] = end = start + len(sent_dict['content'])
        # sent_dict['tokens'] = v
        # sent_dict['pos'] = []
        assert sent_dict['content'] == my_dict['doc_content'][start: end]

        sent_dict['tokens'] = v
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        sent_dict['pos'] = []
        parsed_tokens = p.posdep(sent_dict['tokens'], is_sent=True)['tokens']
        for token in parsed_tokens:
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(f"This doc {my_dict['doc_id']} was not annotated!")
        return None
    
    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'event' in item.name:
            eid = int(item.attrs['id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                # my_dict['event_dict'][eid]['d_span'] = (start, start + len(mention))
                my_dict['event_dict'][eid]['start_char'], my_dict['event_dict'][eid]['end_char'] = start, start + len(mention)
                my_dict['event_dict'][eid]['token_id'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = sent_id = find_sent_id(my_dict['sentences'], mention_span)
                if not all(tok in  my_dict["event_dict"][eid]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]):
                    print(f'{my_dict["event_dict"][eid]["mention"]} - \
                    {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]}')
                    print(f'{my_dict["event_dict"][eid]}  - {my_dict["sentences"][sent_id]}')
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    for item in xml_dom.find('relations').children:
        if type(item)== bs4.element.Tag and 'clink' in item.name:
            r_id = item.attrs['id']
            r_typ = 'PRECONDITION'
            head = int(item.find('source').attrs['id'])
            tail = int(item.find('target').attrs['id'])

            assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
            my_dict['relation_dict'][f"{(head, tail)}"] = r_typ
    
    # Add Norel into data
    event_pairs = combinations(my_dict['event_dict'].keys(), 2)
    for eid1, eid2 in event_pairs:
        e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
        sid1, sid2 = e1['sent_id'], e2['sent_id']
        if sid1 == sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
            my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'

    # coref 
    doc_content = my_dict['doc_content']
    sent_spans = [(sent['sent_start_char'], sent['sent_end_char']) for sent in my_dict['sentences']]
    corefed_doc = predictor.predict(document=doc_content)
    doc_tokens = corefed_doc['document']
    token_span_doc = tokenized_to_origin_span(doc_content, doc_tokens)
    coref_clusters = []
    for i, c in enumerate(corefed_doc['clusters']):
        cluster = defaultdict(list)
        for m in c:
            mention_span = (token_span_doc[m[0]][0], token_span_doc[m[-1]][-1])
            cluster['mentions'].append(mention_span)
            for j, (start, end) in enumerate(sent_spans):
                if start <= mention_span[0] and end >= mention_span[1]:
                    cluster['sentences'].append(j)
        coref_clusters.append(dict(cluster))
    my_dict['coref'] = coref_clusters

    return my_dict


if __name__ == '__main__':
    my_dict = tsvx_reader(dir_name="datasets/hievents_v2/processed/", file_name="article-1526.tsvx")
    with open("article-1526.tsvx.json", 'w') as f:
        json.dump(my_dict,f, indent=6)
    
    my_dict = cat_xml_reader(dir_name="datasets/EventStoryLine/annotated_data/v0.9/", file_name="1/1_1ecbplus.xml.xml", intra=True, inter=True)
    with open("1_1ecbplus.xml.xml.json", 'w') as f:
        json.dump(my_dict,f, indent=6)

    my_dict = tml_reader(dir_name="datasets/MATRES/te3-platinum/", file_name="bbc_20130322_1150.tml")
    with open("bbc_20130322_1150.tml.json", 'w') as f:
        json.dump(my_dict,f, indent=6)

