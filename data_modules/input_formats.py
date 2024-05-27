from abc import ABC, abstractmethod
from re import template
from typing import Dict, Tuple


class BaseInputFormat(ABC):
    name = None

    @abstractmethod
    def format_input(self, *args, **kwargs) -> str:
        raise NotImplementedError


INPUT_FORMATS : Dict[str, BaseInputFormat] = {}


def register_input_format(format_class: BaseInputFormat):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


@register_input_format
class ECIInputFormat(BaseInputFormat):
    """
    TODO: Try template: Does {head} cause {tail}?
    """
    name = 'ECI_input'
    task_prefix = 'Identify causality:'
    start_trigger_token = '['
    end_trigger_token = ']'
    template = "{task_prefix}\n\n{context}\n\nIs there a causal relation between [{head}] and [{tail}]?"

    def format_input(self, context: str, head_position: Tuple[int, int], tail_position: Tuple[int, int]) -> str:
        """Add maker and feed data into template"""
        
        # Find the position of triggers and add marker, aka., []
        head_str = context[head_position[0]: head_position[1]]
        tail_str = context[tail_position[0]: tail_position[1]]
        ordered_position = list(head_position) + list(tail_position)
        ordered_position.sort(reverse=True)
        for pos in ordered_position:
            if pos == head_position[0] or pos == tail_position[0]:
                context = context[: pos] + '[' + context[pos :]
            if pos == head_position[1] or pos == tail_position[1]:
                context = context[: pos] + ']' + context[pos :]
        
        # Feed data into template
        input_txt = self.template.format(task_prefix=self.task_prefix,
                                        context=context, 
                                        head=head_str, 
                                        tail=tail_str)
        return input_txt, head_str, tail_str


@register_input_format
class REInputFormat(BaseInputFormat):
    """
    TODO: 
    """
    name = 'RE_input'
    task_prefix = 'Relation classification:'
    start_trigger_token = '['
    end_trigger_token = ']'
    template = "{task_prefix}\n\n{context}\n\nRelation between {head} and {tail} is "

    def format_input(self, context: str, head_position: Tuple[int, int], tail_position: Tuple[int, int]) -> str:
        head_str = context[head_position[0]: head_position[1]]
        tail_str = context[tail_position[0]: tail_position[1]]
        ordered_position = list(head_position) + list(tail_position)
        ordered_position.sort(reverse=True)
        for pos in ordered_position:
            if pos == head_position[0] or pos == tail_position[0]:
                context = context[: pos] + '[' + context[pos :]
            if pos == head_position[1] or pos == tail_position[1]:
                context = context[: pos] + ']' + context[pos :]
        
        input_txt = self.template.format(task_prefix=self.task_prefix,
                                        context=context, 
                                        head=head_str, 
                                        tail=tail_str)
        return input_txt, head_str, tail_str


@register_input_format
class TREInputFormat(REInputFormat):
    """
    TODO: 
    """
    name = 'TRE_input'
    task_prefix = 'Temporal relation classification:'


@register_input_format
class SREInputFormat(REInputFormat):
    """
    TODO: 
    """
    name = 'SRE_input'
    task_prefix = 'Subevent relation classification:'
