import argparse
import configparser
import pdb
import shutil
from typing import Dict
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn as nn
import os
import optuna
from transformers import HfArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from data_modules.datamodule import EEREDataModule
from model.HOTEERE import HOTEERE


def run(defaults: Dict):
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    print("Hyperparams: {}".format(defaults))
    defaults.update(dict(config.items(job)))

    for key in defaults:
        if defaults[key] in ['True', 'False']:
            defaults[key] = True if defaults[key]=='True' else False
        if defaults[key] == 'None':
            defaults[key] = None
    
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    # print(second_parser.parse_args_into_dataclasses(remaining_args))
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)
    data_args.datasets = job

    if args.tuning:
        training_args.output_dir = './tuning_experiments'
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    f1s = []
    ps = []
    rs = []
    val_f1s = []
    val_ps = []
    val_rs = []
    for i in range(data_args.n_fold):
        print(f"TRAINING AND TESTING IN FOLD {i}: ")
        dm = EEREDataModule(data_name=data_args.datasets,
                            batch_size=data_args.batch_size,
                            data_dir=data_args.data_dir,
                            fold=i)
        number_step_in_epoch = len(dm.train_dataloader())
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}-fold{i}'
            f'-slr{training_args.selector_lr}'
            f'-glr{training_args.generator_lr}'
            f'-eps{training_args.num_epoches}'
            f'-mle_weight{training_args.weight_mle}'
            f'-selector_weight{training_args.weight_selector_loss}'
            f'-SOT_{model_args.kg_weight}_{model_args.n_selected_sents}'
            f'-WOT_{model_args.n_selected_words}')
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass
        if data_args.n_fold != 1:
            output_dir = os.path.join(output_dir,f'fold{i}')
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
        checkpoint_callback = ModelCheckpoint(
                                    dirpath=output_dir,
                                    save_top_k=1,
                                    monitor='f1_dev',
                                    mode='max',
                                    save_weights_only=True,
                                    filename='{epoch}-{f1_dev:.2f}', # this cannot contain slashes 
                                    )
        lr_logger = LearningRateMonitor(logging_interval='step')
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"logs_{args.job}")

        model = HOTEERE(weight_mle=training_args.weight_mle,
                        num_training_step=int(number_step_in_epoch * training_args.num_epoches),
                        selector_lr=training_args.selector_lr,
                        generator_lr=training_args.generator_lr,
                        weight_selector_loss=training_args.weight_selector_loss,
                        OT_eps=1e-3,
                        OT_max_iter=75,
                        OT_reduction='mean',
                        dropout=0.5,
                        kg_weight=model_args.kg_weight,
                        finetune_selector_encoder=training_args.finetune_selector_encoder,
                        finetune_in_OT_generator=training_args.finetune_in_OT_generator,
                        encoder_name_or_path=model_args.model_name_or_path,
                        tokenizer_name_or_path=model_args.tokenizer_name_or_path,
                        n_align_sents=model_args.n_align_sents,
                        n_align_words=model_args.n_align_words,
                        n_selected_sents=model_args.n_selected_sents,
                        n_selected_words=model_args.n_selected_words,
                        output_max_length=model_args.output_max_length,
                        use_rnn=model_args.use_rnn)
        
        trainer = Trainer(
            logger=tb_logger,
            min_epochs=5,
            max_epochs=training_args.num_epoches, 
            accelerator="gpu", 
            devices=[args.gpu],
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            num_sanity_val_steps=3, 
            val_check_interval=1.0, # use float to check every n epochs 
            callbacks = [lr_logger, checkpoint_callback],
        )

        print("Training....")
        dm.setup('fit')
        trainer.fit(model, dm)

        best_model = HOTEERE.load_from_checkpoint(checkpoint_callback.best_model_path)
        print("Testing .....")
        dm.setup('test')
        trainer.test(best_model, dm)
        # print(best_model.model_results)
        p, r, f1 = best_model.model_results
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        print(f"RESULT IN FOLD {i}: ")
        print(f"F1: {f1}")
        print(f"P: {p}")
        print(f"R: {r}")
        # shutil.rmtree(f'{output_dir}')
    
    f1 = sum(f1s)/len(f1s)
    p = sum(ps)/len(ps)
    r = sum(rs)/len(rs)
    print(f"F1: {f1} - P: {p} - R: {r}")
    
    return p, r, f1

def objective(trial: optuna.Trial):
    defaults = {
        'num_epoches': trial.suggest_categorical('num_epoches', [20]),
        'batch_size': trial.suggest_categorical('batch_size', [24]),
        'weight_mle': trial.suggest_categorical('weight_mle', [0.75]),
        'selector_lr': trial.suggest_categorical('selector_lr', [5e-5]),
        'generator_lr': trial.suggest_categorical('generator_lr', [1e-4]),
        'weight_selector_loss': trial.suggest_categorical('weight_selector_loss', [0.25]),
        'kg_weight': trial.suggest_categorical('kg_weight', [0.005]),
        'n_align_sents': trial.suggest_categorical('n_align_sents', [2]),
        'n_align_words': trial.suggest_categorical('n_align_words', [1]),
        'n_selected_sents': trial.suggest_categorical('n_selected_sents', [None]),
        'n_selected_words': trial.suggest_categorical('n_selected_words', [None]),
        'output_max_length': trial.suggest_categorical('output_max_length', [64]),
        'finetune_selector_encoder': trial.suggest_categorical('finetune_selector_encoder', [False]),
        'finetune_in_OT_generator': trial.suggest_categorical('finetune_in_OT_generator', [False]),
        'use_rnn': trial.suggest_categorical('use_rnn', [True])
    } 

    seed_everything(1741, workers=True)

    dataset = args.job
    
    p, r, f1 = run(defaults=defaults)

    record_file_name = 'result.txt'
    if args.tuning:
        record_file_name = f'result_{args.job}.txt'

    if f1 > 0.5:
        with open(record_file_name, 'a', encoding='utf-8') as f:
            f.write(f"Dataset: {dataset} \n")
            f.write(f"Random_state: 1741\n")
            f.write(f"Hyperparams: \n {defaults}\n")
            f.write(f"F1: {f1}  \n")
            f.write(f"P: {p} \n")
            f.write(f"R: {r} \n")
            f.write(f"{'--'*10} \n")

    return f1


 
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='tune hyperparameters')

    args, remaining_args = parser.parse_known_args()
    if args.tuning:
        print("tuning ......")
        # sampler = optuna.samplers.TPESampler(seed=1741)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=19)
        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))

