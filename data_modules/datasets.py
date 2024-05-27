from collections import defaultdict
import json
import os
import random
from typing import Dict, List, Tuple
import pickle
from tqdm import tqdm
from data_modules.base_dataset import BaseDataset
from data_modules.input_example import Entity, InputExample, Relation, RelationType
from data_modules.preprocessor import load


DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(name: str,
                data_dir: str,
                fold: int = 0,
                split: str = 'train',):
    '''
    Load a registered dataset
    '''
    return DATASETS[name](data_dir=data_dir, fold=fold, split=split)


class EEREDataset(BaseDataset):
    sample_rate = 1
    relation_types = None
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use
    input_format_type = None
    output_format_type = None

    def load_schema(self):
        self.relation_types = {
            short: RelationType(short=short, natural=natural)
            for short, natural in self.natural_relation_types.items()
        }

    def load_data(self, data_path: str, load_fold:int = 0) -> List[InputExample]:
        examples = {}
        self.load_schema()
        print(f"Loading {self.name} from {data_path} with sample rate is {self.sample_rate}")
        corpus = load(self.name, load_fold=load_fold, save_cache=False)
        for fold in corpus.keys():
            train, val, test = corpus[fold]
            example_train, example_test, example_val = [], [], []
            for i, datapoint in enumerate(train):
                if datapoint['label'] in ['NoRel'] and random.uniform(0, 1) >= self.sample_rate:
                    continue
                ev1 = Entity(id=f'e0-{i}',
                            mention=datapoint['trigger1']['mention'],
                            start_char_in_sent=datapoint['trigger1']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger1']['sent_span'][1],
                            sent_id=datapoint['trigger1']['sent_id']
                            )
                ev2 = Entity(id=f'e1-{i}',
                            mention=datapoint['trigger2']['mention'],
                            start_char_in_sent=datapoint['trigger2']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger2']['sent_span'][1],
                            sent_id=datapoint['trigger2']['sent_id']
                            )
                example = InputExample(
                    dataset=self.name, 
                    id=i,
                    triggers=[ev1, ev2],
                    relations=[Relation(self.relation_types[datapoint['label']], ev1, ev2)],
                    context=datapoint['sentences_context'],
                    sentences_presentation=datapoint['doc_presentation'],
                    kg_sents=datapoint['kg_sentences'],
                    kg_sents_embs=datapoint['kg_sent_embs'],
                    cluster=datapoint['coref'],
                    score_over_doc_tree=datapoint['score_over_doc_tree'],
                    input_format_type=self.input_format_type,
                    output_format_type=self.output_format_type
                )
                example_train.append(example)
            for i, datapoint in enumerate(val):
                if datapoint['label'] in ['NoRel'] and random.uniform(0, 1) >= self.sample_rate:
                    continue
                ev1 = Entity(id=f'e0-{i}',
                            mention=datapoint['trigger1']['mention'],
                            start_char_in_sent=datapoint['trigger1']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger1']['sent_span'][1],
                            sent_id=datapoint['trigger1']['sent_id']
                            )
                ev2 = Entity(id=f'e1-{i}',
                            mention=datapoint['trigger2']['mention'],
                            start_char_in_sent=datapoint['trigger2']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger2']['sent_span'][1],
                            sent_id=datapoint['trigger2']['sent_id']
                            )
                example = InputExample(
                    dataset=self.name, 
                    id=i,
                    triggers=[ev1, ev2],
                    relations=[Relation(self.relation_types[datapoint['label']], ev1, ev2)],
                    context=datapoint['sentences_context'],
                    sentences_presentation=datapoint['doc_presentation'],
                    kg_sents=datapoint['kg_sentences'],
                    kg_sents_embs=datapoint['kg_sent_embs'],
                    cluster=datapoint['coref'],
                    score_over_doc_tree=datapoint['score_over_doc_tree'],
                    input_format_type=self.input_format_type,
                    output_format_type=self.output_format_type
                )
                example_val.append(example)
            for i, datapoint in enumerate(test):
                if datapoint['label'] in ['NoRel'] and random.uniform(0, 1) >= self.sample_rate:
                    continue
                ev1 = Entity(id=f'e0-{i}',
                            mention=datapoint['trigger1']['mention'],
                            start_char_in_sent=datapoint['trigger1']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger1']['sent_span'][1],
                            sent_id=datapoint['trigger1']['sent_id']
                            )
                ev2 = Entity(id=f'e1-{i}',
                            mention=datapoint['trigger2']['mention'],
                            start_char_in_sent=datapoint['trigger2']['sent_span'][0],
                            end_char_in_sent=datapoint['trigger2']['sent_span'][1],
                            sent_id=datapoint['trigger2']['sent_id']
                            )
                example = InputExample(
                    dataset=self.name, 
                    id=i,
                    triggers=[ev1, ev2],
                    relations=[Relation(self.relation_types[datapoint['label']], ev1, ev2)],
                    context=datapoint['sentences_context'],
                    sentences_presentation=datapoint['doc_presentation'],
                    kg_sents=datapoint['kg_sentences'],
                    kg_sents_embs=datapoint['kg_sent_embs'],
                    cluster=datapoint['coref'],
                    score_over_doc_tree=datapoint['score_over_doc_tree'],
                    input_format_type=self.input_format_type,
                    output_format_type=self.output_format_type
                )
                example_test.append(example)
            examples[fold] = {
                'train': example_train,
                'val': example_val,
                'test': example_test
            }
        return examples


@register_dataset
class HiEveDataset(EEREDataset):
    name = 'HiEve'
    sample_rate = 1.0
    natural_relation_types = {
                            "SuperSub": "including", 
                            "SubSuper": "a part of", 
                            "Coref": "coreference", 
                            "NoRel": "no relation"
                            }
    input_format_type = 'SRE_input'
    output_format_type = 'SRE_output'


@register_dataset
class MATRESDataset(EEREDataset):
    name = 'MATRES'
    sample_rate = 1
    natural_relation_types = {
                            "BEFORE": "before", 
                            "AFTER": "after", 
                            "EQUAL": "same time", 
                            "VAGUE": "no relation"
                            }
    input_format_type = 'TRE_input'
    output_format_type = 'TRE_output'


@register_dataset
class ESLDataset(EEREDataset):
    name = 'ESL'
    sample_rate = 1
    natural_relation_types = {
                            'FALLING_ACTION': 'is caused by', 
                            'PRECONDITION': 'causes', 
                            'NoRel': "has no relation to"
                            }
    input_format_type = 'ECI_input'
    output_format_type = 'ECI_output'


@register_dataset
class CTBDataset(EEREDataset):
    name = 'Causal-TB'
    sample_rate = 1
    natural_relation_types = {
                            'PRECONDITION': 'causes', 
                            'NoRel': "has no relation to"
                            }
    input_format_type = 'ECI_input'
    output_format_type = 'ECI_output'

