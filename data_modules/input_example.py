from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple, Union
from torch import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    id: Tuple[str, int] = None              # mention id
    mention: str = None                     # mention of entity
    type: Optional[EntityType] = None       # entity type
    start_char_in_sent: int = None          # start char in sent 
    end_char_in_sent: int = None            # end char in sent 
    sent_id: Optional[int] = None           # sentence id which containt entity
    token_id_in_sent: Optional[List[int]] = None    # position of entity in sentence

    def to_tuple(self):
        return self.mention, self.id

    def __hash__(self):
        return hash((self.mention, self.id))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType  # relation type
    head: Entity        # head of the relation
    tail: Entity        # tail of the relation

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()



@dataclass
class InputExample:
    """
    A single training/ testing example
    """
    dataset: Optional[str] = None
    id: Optional[str] = None
    triggers: List[Entity] = None
    relations: List[Relation] = None
    context: List[str] = None
    sentences_presentation: Tensor = None # (ns, hidden_size)
    kg_sents: List[str] = None
    kg_sents_embs: Tensor = None # (num_kg_sent, hidden_size)
    cluster: List[Dict[int, Any]] = None
    score_over_doc_tree: Dict = None
    input_format_type: str = None
    output_format_type: str = None

    

