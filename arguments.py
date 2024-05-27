from curses import meta
from dataclasses import dataclass, field
from typing import List, Optional
import transformers


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    batch_size: int = field(
        default = 8,
        metadata= {"help": "Batch size"}
    )
    
    n_fold: int = field(
        default = 1,
        metadata={"help": "Number folds of dataset"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    selector_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of selector"}
    )

    generator_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate of generator"}
    )

    gradient_clip_val: float = field(
        default=0.0,
        metadata={"help":"Gradient clipping value"}
    )

    num_epoches : int = field(
        default=5,
        metadata={"help": "number pretrain epoches"}
    )

    seed: int = field(
        default=1741,
        metadata={"help": "seeding for reproductivity"}
    )
    weight_mle: float = field(
        default=0.8,
        metadata={"help": "weight of generating mle loss"}
    )

    weight_selector_loss: float = field(
        default=0.5,
        metadata={"help": "weight of selector loss"}
    )

    finetune_selector_encoder: bool = field(
        default=True,
        metadata={"help": "Fine-tune selector encoder or not"}
    )

    finetune_in_OT_generator: bool = field(
        default=True,
        metadata={"help": "Fine-tune generator encoder (in OT) or not"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    kg_weight: float = field(
        default=0.5,
        metadata={"help": "Probability of KG senentence in selector"}
    )

    n_align_sents: Optional[int] = field(
        default=5,
        metadata={"help": "Number align sentences"}
    )

    n_align_words: Optional[int] = field(
        default=10,
        metadata={"help": "Number align words"}
    )

    n_selected_sents: Optional[int] = field(
        default=None,
        metadata={"help": "Number selected sentences"}
    )

    n_selected_words: Optional[int] = field(
        default=None,
        metadata={"help": "Number selected words"}
    )

    output_max_length: Optional[int] = field(
        default=64,
        metadata={"help": "Max length of Output sequences"}
    )

    use_rnn: Optional[bool] = field(
        default=False,
        metadata={'help': "Use rnn to imporve sentence consequentive ability"}
    )



