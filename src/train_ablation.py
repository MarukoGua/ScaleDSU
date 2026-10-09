import os
import gc
import copy
import json
import logging
import argparse
import re

import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader

from model.ablation import AblationDSU, AblationInput
from model.conv import ConvConfig
import data_util
from metrics import *
from utils import *


def train_one_epoch(attack_model, dataloader, optimizer, scheduler=None):
    attack_model.train()
    iter_source = iter(dataloader)
    num_iter = len(dataloader)
    cls_losses, mmd_losses = 0.0, 0.0
    total_loss = 0.
    
    for i in range(1, num_iter+1):
        data = next(iter_source)
        traces = data['norm_trace'].cuda().float()
        labels = data['label'].cuda().long()
        optimizer.zero_grad()
        _, mmd_loss, ce_loss = attack_model(traces, labels, add_uncertainty=True)
        loss = ce_loss + mmd_loss

        loss.backward()
        total_loss += loss.item()
        cls_losses += ce_loss.item()
        if type(mmd_loss) != float:
            mmd_losses += mmd_loss.item()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
    
    total_loss /= (i+1)
    cls_losses /= (i+1)
    mmd_losses /= (i+1)
    # if scheduler is not None:    
        # logging.info(f'\tcurrent_lr: {str(scheduler.get_last_lr()[0])}')
    return total_loss, cls_losses, mmd_losses
    

def evaluate(attack_model, valid_dataloader, add_uncertainty=False):
    with torch.no_grad():
        attack_model.eval()
        running_vloss = 0.0
        cls_losses, mmd_losses = 0.0, 0.0
        vpredictions_stat = []
        vlabels_stat = []
        probabilities = []
        
        for i, vdata in enumerate(valid_dataloader):
            vtraces = vdata['norm_trace'].cuda().float()
            vlabels = vdata['label'].cuda().long()

            voutputs, mmd_loss, ce_loss = attack_model(vtraces, vlabels, add_uncertainty)
            cls_losses += ce_loss.item()
            if type(mmd_loss) != float:
                mmd_losses += mmd_loss.item()
            vloss = ce_loss + mmd_loss
            running_vloss += vloss.item()

            vpreds = voutputs.argmax(dim=-1).view(-1).tolist()
            vpredictions_stat.extend(vpreds)
            vlabels_stat.extend(vlabels.view(-1).tolist())
            entropy = torch.nn.functional.softmax(voutputs, dim=1).detach().cpu()
            probabilities.extend(entropy)
    
        avg_vloss = running_vloss / (i+1)
        cls_losses /= (i+1)
        mmd_losses /= (i+1)
        accuracy = sum([1 for m, n in zip(vpredictions_stat, vlabels_stat) if m==n])/len(vlabels_stat)
    
    return {
        'model_output': vpredictions_stat,
        'labels': vlabels_stat,
        'loss': avg_vloss,
        'assist_loss': (cls_losses, mmd_losses),
        'accuracy': accuracy,
        'entropy': np.stack(probabilities)
    }

def setup_logging(log_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_file = os.path.join(log_dir, f'log.txt')
    logging.basicConfig(
        filename=logging_file,
        encoding='utf-8',
        level=logging.INFO,
        format='%(message)s',
        filemode='a')

def profile(training_args):
    s = training_args.seed
    set_seed(s)

    # ==================== TRAIN PHASE ====================
    # Only the Profiling_traces data is loaded: 100k traces for training and
    # 10k traces for validation. No target data is loaded during training,
    # which keeps memory usage low.
    load_func = data_util.DATA_TYPE[training_args.data]
    src_train_data, src_val_data = load_func(training_args, mode='train')

    preprocess = training_args.preprocess
    preprocess_method = data_util.PREPROCESS_METHOD[preprocess]
    src_train_data.apply_preprocess(preprocess_method)
    src_val_data.apply_preprocess(preprocess_method)

    config = ConvConfig.from_config_file(training_args.config_file, length=len(src_train_data[0]['norm_trace']))
    if training_args.augment_position == 'internal':
        attack_model = AblationDSU(
            config, 
            perturbation=training_args.perturbation, 
            level=training_args.noise_level,
            denser=training_args.dense_type,
            uncertainty=training_args.uncertainty, 
            factor=int(training_args.factor),
            dropout=float(training_args.dropout)
            ).cuda()
    elif training_args.augment_position == 'input':
        attack_model = AblationInput(
            config, 
            perturbation=training_args.perturbation, 
            level=training_args.noise_level,
            denser=training_args.dense_type,
            uncertainty=training_args.uncertainty, 
            factor=int(training_args.factor),
            dropout=float(training_args.dropout)
        ).cuda()
    print(attack_model)
    bs = int(config.kwargs['batch_size'])

    src_train_loader = DataLoader(src_train_data, batch_size=bs, shuffle=True, num_workers=4)
    src_val_loader = DataLoader(src_val_data, batch_size=bs, shuffle=False, num_workers=4) 

    lr = float(config.kwargs['lr'])
    optimizer = torch.optim.AdamW(attack_model.parameters(), lr=lr, weight_decay=0.03)
    num_epoch = int(config.kwargs['num_epoch'])
    scheduler = None
    if bool(config.kwargs['use_scheduler']):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            steps_per_epoch=len(src_train_loader), 
            epochs=int(num_epoch), 
            pct_start=0.3,
            div_factor=10,
            final_div_factor=2000
            )
    pbar = tqdm.tqdm(range(num_epoch))

    model_path = f'{training_args.output_dir}/{training_args.perturbation}-{args.augment_position}-{training_args.dense_type}.pt'

    counter = 0
    best_loss = 1000.

    for epoch in pbar:
        train_loss, cls_loss, diff_loss = train_one_epoch(attack_model, src_train_loader, optimizer, scheduler)
        epoch_info = {'train_loss': train_loss, 'cls': cls_loss, 'diff': diff_loss}
        print_info = f'Epoch {epoch}/{num_epoch}. Train loss={round(train_loss, 5)}, assist loss=({round(cls_loss, 5), round(diff_loss, 5)}).'

        evaluate_output = evaluate(attack_model, src_val_loader, add_uncertainty=True)
        valid_loss = evaluate_output['loss']
        assist_loss = [round(i, 5) for i in evaluate_output['assist_loss']]
        accuracy = evaluate_output['accuracy']
        epoch_info['eval loss'] = valid_loss
        epoch_info['accuracy'] = accuracy
        print_info += f'Eval loss: {round(valid_loss, 5)}, acc: {round(accuracy, 5)} assist loss={assist_loss}.'
        pbar.set_description(print_info)
        print_info += '\n'

        logging.info(print_info)

        better = valid_loss < best_loss
        if better:
            best_loss = valid_loss
            counter = 0
            torch.save(attack_model.state_dict(), model_path)
        else:
            counter += 1
        
        if counter > 10:
            break

    # ============ RELEASE TRAIN-PHASE DATA MEMORY ============
    # logging.info('Training finished. Releasing training data memory...')
    del src_train_data, src_val_data, src_train_loader, src_val_loader
    del attack_model, optimizer, scheduler, pbar
    gc.collect()
    torch.cuda.empty_cache()


def attack(args):
    s = args.seed
    set_seed(s)

    # ==================== ATTACK PHASE ====================
    # For each target, load attack traces, run the attack and release the data
    # right after, so that only one target is ever held in memory at a time.
    load_func = data_util.DATA_TYPE[args.data]
    preprocess = args.preprocess
    test_preprocess = data_util.PREPROCESS_METHOD.get(preprocess)
    if test_preprocess is None:
        test_preprocess = data_util.PREPROCESS_METHOD['no_preprocess']
     
    model_path = f'{args.output_dir}/{args.perturbation}-{args.augment_position}-{args.dense_type}.pt'

    nb_trace = int(args.num_trace_attack)
    nb_attack = int(args.num_attack)
    print_info=f"\nperturbation={args.perturbation}, position={args.augment_position}, uncertainty={args.uncertainty}, factor={args.factor}, dropout={args.dropout}, noise_level={args.noise_level}, dense_type={args.dense_type}\n"
    print_info += f"--------------Attack Results------------\n"
    targets = args.targets.split(',')
    attack_results = {}
    config = None
    attack_model= None
    for tgt in targets:
        logging.info(f'[Attack] loading target {tgt} data...')
        args.targets = tgt
        if args.data == 'ches_challenge':
            tgt_datasets = load_func(args, mode='test', tgts=[tgt], n_test=100_000)
        else:
            tgt_datasets = load_func(args, mode='test')

        for tgt_name, tgt_data in tgt_datasets.items():
            tgt_data.apply_preprocess(test_preprocess)

            if config is None:
                sample_length = len(tgt_data[0]['norm_trace'])
                config = ConvConfig.from_config_file(args.config_file, length=sample_length)
            if attack_model is None:
                if args.augment_position == 'internal':
                    attack_model = AblationDSU(
                        config, 
                        perturbation=args.perturbation, 
                        level=args.noise_level,
                        denser=args.dense_type,
                        uncertainty=args.uncertainty, 
                        factor=int(args.factor),
                        dropout=float(args.dropout)
                        ).cuda()
                elif args.augment_position == 'input':
                    attack_model = AblationInput(
                        config, 
                        perturbation=args.perturbation, 
                        level=args.noise_level,
                        denser=args.dense_type,
                        uncertainty=args.uncertainty, 
                        factor=int(args.factor),
                        dropout=float(args.dropout)
                        ).cuda()
                attack_model.load_state_dict(torch.load(model_path))
            bs = int(config.kwargs['batch_size'])

            tgt_loader = DataLoader(tgt_data, batch_size=bs, shuffle=False, num_workers=4)

            tgt_eval = evaluate(attack_model, tgt_loader)
            metrics = compute_metrics(tgt_eval, tgt_data, nb_attack, nb_trace, compute_ge=True)
            NTGE = metrics['trace_needed']
            attack_results[tgt_name] = {'loss': tgt_eval['assist_loss'][0], 'acc': tgt_eval['accuracy'], 'ntge': NTGE}
            print_info += f"(tgt device){tgt_name}: loss={round(tgt_eval['assist_loss'][0], 4)}, acc={tgt_eval['accuracy']}, NTGE={NTGE}.\n"
            logging.info(print_info)
            print_info = ''

            # ======== RELEASE CURRENT TARGET DATA MEMORY ========
            logging.info(f'[Attack] target {tgt_name} finished. Releasing its data memory...')
            del tgt_datasets, tgt_data, tgt_loader, tgt_eval, metrics
            gc.collect()
            torch.cuda.empty_cache()

    return attack_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Train Deep Learning Model for Side Channel Attack',
    )
    
    parser.add_argument('--data', default='xmega')
    parser.add_argument('--data_file', default='../data/xmega')
    parser.add_argument('--source', default='01')
    parser.add_argument('--targets', default='02')
    parser.add_argument('--implementation', default=None)
    parser.add_argument('--preprocess', type=str, default='standardize')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--model_file', default=None)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_byte', default=0, type=int)
    parser.add_argument('--perturbation', type=str)
    parser.add_argument('--uncertainty', default=1.0)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--noise_level', default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dense_type', type=str, default='mlp')
    parser.add_argument('--augment_position', type=str, default='internal')

    
    parser.add_argument('--output_dir', default="./results/xmega")
    parser.add_argument('--num_trace_attack', default=1000)
    parser.add_argument('--num_attack', default=100)

    args = parser.parse_args()
    args.uncertainty = str2bool(args.uncertainty)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, 'ablation')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    
    setup_logging(args.output_dir)
    args.noise_level = float(args.noise_level) / 10
    args.dropout = float(args.dropout) / 10
    profile(args)
    attack(args)

