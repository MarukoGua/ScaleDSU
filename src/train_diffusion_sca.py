import os
import logging
import argparse

import torch
import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from model.diffusion import SCAEncoder
from model.conv import ConvConfig
import data_util
from metrics import *
from utils import *

def train_one_epoch(attack_model, dataloader, optimizer, scheduler, noise_scheduler, step, ce_loss_func, mse_loss_func):
    clean_cls_losses, noisy_cls_losses, gen_losses, total_losses = 0., 0., 0., 0.
    attack_model.train()
    for i, data in enumerate(dataloader):
        traces = data['norm_trace'].cuda().float()
        labels = data['label'].cuda().long()

        noise = torch.randn(traces.shape).cuda()
        bs = traces.shape[0]
        timesteps = torch.randint(0, step, (bs, ), device=traces.device).long()
        noisy_traces = noise_scheduler.add_noise(traces, noise, timesteps)
    
        optimizer.zero_grad()
        
        clean_feat, clean_pred = attack_model(traces, is_noisy=False)
        noisy_feat, noisy_pred = attack_model(noisy_traces, is_noisy=True)
        clean_cls_loss = ce_loss_func(clean_pred, labels)
        noisy_cls_loss = ce_loss_func(noisy_pred, labels)
        gen_loss = mse_loss_func(clean_feat, noisy_feat)
        loss = attack_model.criterion([clean_cls_loss, noisy_cls_loss, gen_loss])
        
        loss.backward()
        optimizer.step()
        
        clean_cls_losses += clean_cls_loss.mean(dim=-1).item()
        noisy_cls_losses += noisy_cls_loss.mean(dim=-1).item()
        gen_losses += gen_loss.mean(dim=-1).item()
        total_losses += loss.item()

        if scheduler is not None:
            scheduler.step()

    clean_cls_losses = round(clean_cls_losses/(i+1), 3)
    noisy_cls_losses = round(noisy_cls_losses/(i+1), 3)
    gen_losses = round(gen_losses /(i+1), 3)
    total_losses = round(total_losses /(i+1), 3)
    return clean_cls_losses, noisy_cls_losses, gen_losses, total_losses
    

def evaluate(attack_model, dataloader, noise_scheduler, step, ce_loss_func, mse_loss_func):
    with torch.no_grad():
        clean_cls_losses, noisy_cls_losses, gen_losses, total_losses = 0., 0., 0., 0.
        attack_model.eval()
        for i, data in enumerate(dataloader):
            traces = data['norm_trace'].cuda().float()
            labels = data['label'].cuda().long()

            noise = torch.randn(traces.shape).cuda()
            bs = traces.shape[0]
            timesteps = torch.randint(0, step, (bs, ), device=traces.device).long()
            noisy_traces = noise_scheduler.add_noise(traces, noise, timesteps)
            
            clean_feat, clean_pred = attack_model(traces, is_noisy=False)
            noisy_feat, noisy_pred = attack_model(noisy_traces, is_noisy=True)
            clean_cls_loss = ce_loss_func(clean_pred, labels)
            noisy_cls_loss = ce_loss_func(noisy_pred, labels)
            gen_loss = mse_loss_func(clean_feat, noisy_feat)
            loss = attack_model.criterion([clean_cls_loss, noisy_cls_loss, gen_loss])
            
            clean_cls_losses += clean_cls_loss.mean(dim=-1).item()
            noisy_cls_losses += noisy_cls_loss.mean(dim=-1).item()
            gen_losses += gen_loss.mean(dim=-1).item()
            total_losses += loss.item()

    clean_cls_losses = round(clean_cls_losses/(i+1), 3)
    noisy_cls_losses = round(noisy_cls_losses/(i+1), 3)
    gen_losses = round(gen_losses /(i+1), 3)
    total_losses = round(total_losses /(i+1), 3)
    return clean_cls_losses, noisy_cls_losses, gen_losses, total_losses

def test(attack_model, tgt_val_loader):
    with torch.no_grad():
        attack_model.eval()
        vpredictions_stat = []
        vlabels_stat = []
        probabilities = []

        running_vloss = 0.

        for i, tgt_batch in enumerate(tgt_val_loader):
            tgt_trace = tgt_batch['norm_trace'].cuda().float()
            tgt_label = tgt_batch['label'].cuda().long()

            _, pred = attack_model(tgt_trace)
            tgt_cls_loss = torch.nn.functional.cross_entropy(pred, tgt_label)
            
            vpreds = pred.argmax(dim=-1).view(-1).tolist()
            vpredictions_stat.extend(vpreds)
            vlabels_stat.extend(tgt_label.view(-1).tolist())
            entropy =torch.nn.functional.softmax(pred, dim=1).detach().cpu()
            probabilities.extend(entropy)

            running_vloss += tgt_cls_loss.item()

        running_vloss /= (i+1)
        assert len(vpredictions_stat) == len(vlabels_stat)
        accuracy = sum([1 for m, n in zip(vpredictions_stat, vlabels_stat) if m==n])/len(vlabels_stat)

        return {
            'model_output': vpredictions_stat,
            'labels': vlabels_stat,
            'loss': running_vloss,
            'accuracy': accuracy,
            'entropy': np.stack(probabilities)
            }

    
def profile(args):
    # set random seeds
    s = args.seed
    set_seed(s)

    load_func = data_util.DATA_TYPE[args.data]
    src_train_data, src_val_data = load_func(args, mode='train')

    preprocess = args.preprocess
    preprocess_method = data_util.PREPROCESS_METHOD.get(preprocess)
    preprocess_method = data_util.PREPROCESS_METHOD['no_preprocess'] if preprocess_method is None else preprocess_method

    src_train_data.apply_preprocess(preprocess_method)
    src_val_data.apply_preprocess(preprocess_method)

    config = ConvConfig.from_config_file(args.config_file, length=len(src_train_data[0]['norm_trace']))
    attack_model = SCAEncoder(config).cuda()
    print(attack_model)
    input()
   
    training_config = config.kwargs

    bs = int(training_config['batch_size'])
    src_train_loader = DataLoader(src_train_data, batch_size=bs, shuffle=True, num_workers=4)
    src_val_loader = DataLoader(src_val_data, batch_size=bs, shuffle=False, num_workers=4)


    lr = float(training_config['lr'])
    optimizer = torch.optim.AdamW([
        {'params': attack_model.cnet.parameters()},
        {'params': attack_model.std}, 
        {'params': attack_model.unet.parameters()}
        ], lr=lr)

    num_epoch = int(training_config['num_epoch'])
    use_scheduler = bool(training_config['use_scheduler'])


    if use_scheduler == 1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            steps_per_epoch=len(src_train_loader), 
            epochs=int(num_epoch), 
            pct_start=0.3,
            div_factor=10,
            final_div_factor=2000
            )
    else:
        scheduler = None

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        clip_sample=False
        )
    step = args.step

    ce_loss_func = torch.nn.CrossEntropyLoss()
    mse_loss_func = torch.nn.MSELoss()
    pbar = tqdm.tqdm(range(num_epoch))

    counter = 0
    best_loss = 1000.

    model_path = f'{args.output_dir}/model.pt'

    for epoch in pbar:
        tclean, tnoisy, tgen, tloss = train_one_epoch(attack_model, src_train_loader, optimizer, scheduler, noise_scheduler, step, ce_loss_func, mse_loss_func)
        print_info = f'Epoch {epoch}/{num_epoch}.\n\tTrain losses: clean ce={tclean}, noisy ce={tnoisy}, diffusion reverse={tgen}, total={tloss}. \n '
        log_vars = torch.log(attack_model.std ** 2)
        precisions = torch.exp(-log_vars)
        print_info += f'\tlog_var values: {[round(i.item(), 4) for i in log_vars]} \n'
        print_info += f'\tscalars: {[round(i.item(), 4) for i in precisions]} \n'
        
        vclean, vnoisy, vgen, vloss = evaluate(attack_model, src_val_loader, noise_scheduler, step, ce_loss_func, mse_loss_func)
        print_info += f'\tValidation loss: clean ce={vclean}, noisy ce={vnoisy}, diffusion reverse={vgen}, total={vloss}.'
        logging.info(print_info)

        better = vloss < best_loss
        stop = vloss <= 0
        if stop:
            break

        if better:
            # best_loss = gen_loss
            best_loss = vloss
            torch.save(attack_model.state_dict(), model_path)
            counter = 0
        else:
            counter += 1
        
        if counter > 20:
            break

def attack(args):
    set_seed(args.seed)

    if args.model_file is not None:
        model_file = args.model_file
    else:
        model_file = os.path.join(args.output_dir, 'model.pt')

    if not os.path.exists(model_file):
        logging.error('where is your model file??')
        exit()

    load_func = data_util.DATA_TYPE[args.data]
    tgt_val_datasets = load_func(args, mode='test')

    preprocess = args.preprocess
    preprocess_method = data_util.PREPROCESS_METHOD.get(preprocess)
    preprocess_method = data_util.PREPROCESS_METHOD['no_preprocess'] if preprocess_method is None else preprocess_method

    for tgt, tgt_data in tgt_val_datasets.items():
            tgt_data.apply_preprocess(preprocess_method)

    sample_length = len(tgt_data[0]['norm_trace'])
    
    config = ConvConfig.from_config_file(args.config_file, length=sample_length)
    attack_model = SCAEncoder(config).cuda()
    attack_model.load_state_dict(torch.load(model_file, weights_only=True))
    bs = int(config.kwargs['batch_size'])

    logging.info(f"\nCross-Device Attack Results\n")
    for tgt, tgt_data in tgt_val_datasets.items():
        dataloader = DataLoader(tgt_data, batch_size=bs, shuffle=False, num_workers=4)
        test_output = test(attack_model, dataloader)

        nb_trace = int(args.num_trace_attack)
        nb_attack = int(args.num_attack)

        metrics = compute_metrics(test_output, tgt_data, nb_attack, nb_trace)

        print_info = f"\tDevice{tgt}: ge={metrics['trace_needed'][0]}, accuracy={metrics['accuracy']}."
        logging.info(print_info)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Train Deep Learning Model for Side Channel Attack',
    )
    
    parser.add_argument('--data', default='xmega')
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--targets', type=str)
    parser.add_argument('--implementation', default=None)
    parser.add_argument('--preprocess', type=str, default='standardize')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--model_file', default=None)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_byte', default=0, type=int)
    parser.add_argument('--step', default=100, type=int)

    parser.add_argument('--output_dir', default="./result/xmega")
    parser.add_argument('--num_attack', type=int, default=100)
    parser.add_argument('--num_trace_attack', type=int, default=5000)

    args = parser.parse_args()
   
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, 'diffusion')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    
    logging_file = os.path.join(args.output_dir, f'log.txt')
    
    logging.basicConfig(
        filename=logging_file, 
        encoding='utf-8', 
        level=logging.INFO, 
        format='%(message)s',
        filemode='a')
    profile(args)
    attack(args)


