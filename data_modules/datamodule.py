from torch.utils.data import DataLoader 
import pytorch_lightning as pl

from arguments import DataTrainingArguments
from data_modules.datasets import load_dataset


class EEREDataModule(pl.LightningDataModule):
    """
    Dataset processing for Event Event Relation Extraction.
    """
    def __init__(self,
                data_name: str,
                batch_size: int,
                data_dir: str,
                fold: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_name = data_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.fold = fold
        self.train = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='train')
        self.val = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='val')
        self.test = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='test')
        self.my_collate = self.train.my_collate
    
    def train_dataloader(self):
        
        dataloader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.my_collate
        )
        return dataloader
    
    def val_dataloader(self):
       
        dataloader = DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.my_collate
        )
        return dataloader
    
    def test_dataloader(self):
        
        dataloader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.my_collate
        )
        return dataloader