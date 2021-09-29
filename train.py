#
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""
Authors: Subhojeet Pramanik

OmniNet training script. 

"""
import argparse
import os
import torch
import time
import glob
import numpy as np
import libs.omninet as omninet
from libs.utils import dataloaders as dl
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import libs.omninet.routines as r
from libs.omninet.util import ScheduledOptim
from torch.optim.adam import Adam
from libs.utils.Adamw import AdamW
import random
import sys
from tqdm import tqdm
from libs.utils.train_util import *
import re
import pickle
import json
from libs.utils.vqa.vqa import VQA
from libs.utils.vqa.vqaEval import VQAEval

from omninet_config import *

parser = argparse.ArgumentParser(description='OmniNet training script.')
parser.add_argument('n_iters', help='Number of iterations to train.')
parser.add_argument('tasks', help='List of tasks seperated by comma.')
parser.add_argument('batch_sizes', help='List of batch size for each task seperated by comma')
parser.add_argument('--n_jobs', default=1, help='Number of asynchronous jobs to run for each task.')
parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
parser.add_argument('--n_workers', default=0, type=int, help='Number of workers to load data')
parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
parser.add_argument('--save_all', action='store_true', help='true if save model files every <save_interval> steps, otherwise, save only last two')
parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
parser.add_argument('--eval_interval', help='Interval after which to evaluate on the test/val set.', default=1000)
parser.add_argument('--eval_start', default=0, type=int, help='Step after which to start evaluating on the test/val set.')
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')
parser.add_argument('--init_lr_cca', default=0.02, type=float, help='init_lr_cca')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--weight_decay_cca', default=0, type=float, help='weight decay for cca optimizer')
parser.add_argument('--weight_seed', default=1029, type=int, help='seed for model weight initialization.')
parser.add_argument('--data_seed', default=68, type=int, help='seed for dataloaders.')
parser.add_argument('--data_path', default='/files/yxue/research/allstate/data', help='data path')
parser.add_argument('--structured_folder', default=None, type=str, help='path to structured data.')
parser.add_argument('--conf_type', default='default', help='Choose confurigation types')
parser.add_argument('--model_save_path', default='/out/test', help='path to save the model.')
# parser.add_argument('--save_best', action='store_true', help='True if only save the model on validation.')
#parser.add_argument('--cca_caches', default=['spatial', 'temporal', 'structured'], help='caches ')
parser.add_argument('--no_temporal_encoder', action='store_true', help='True if no temporal encoder.')
parser.add_argument('--not_sa_on_whole_cache', action='store_true', help='True if apply self attention on samples instead of whole cache.')
parser.add_argument('--dropout_p', default=0.1, type=float, help='dropout rate for cca stream on spatial cache')
parser.add_argument('--dropout_t', default=0.1, type=float, help='dropout rate for cca stream on temporal cache')
parser.add_argument('--dropout_s', default=0.1, type=float, help='dropout rate for cca stream on structured cache')
parser.add_argument('--drop_path_rate', default=0., type=float, help='drop_path rate')
parser.add_argument('--sa_drop_path_rate', default=0., type=float, help='drop_path rate in self attention in encoder')
parser.add_argument('--more_dropout', action='store_true', help='true if apply more dropouts than vanilla Omninet')
parser.add_argument('--dropout', default=0.1, type=float, help='default dropout rate')
parser.add_argument('--se_dropout', default=0.1, type=float, help='dropout rate for structured entity periph')
parser.add_argument('--temp_enc_n_layers', default=6, type=int, help='number of temporal encoder layers.')
parser.add_argument('--temp_enc_n_heads', default=8, type=int, help='number of temporal encoder heads.')
parser.add_argument('--dec_n_layers', default=6, type=int, help='number of decoder layers.')
parser.add_argument('--dec_n_heads', default=8, type=int, help='number of decoder heads.')
parser.add_argument('--cca_n_layers', default=6, type=int, help='number of cca layers.')
parser.add_argument('--cca_n_heads', default=8, type=int, help='number of cca heads.')
parser.add_argument('--sa_n_layers', default=6, type=int, help='number of cca self attention layers.')
parser.add_argument('--sa_n_heads', default=8, type=int, help='number of cca self attention heads.')
parser.add_argument('--psa_n_layers', default=6, type=int, help='number of cca self attention layers for spatial stream.')
parser.add_argument('--psa_n_heads', default=8, type=int, help='number of cca self attention heads for spatial stream.')
parser.add_argument('--cca_hidden_dim', default=4096, type=int, help='cca d_inner.')
parser.add_argument('--patch_size', default=None, type=int, help='patch size')
parser.add_argument('--patch_stride', default=None, type=int, help='patch stride')
parser.add_argument('--max_clip_len', default=16, type=int, help='max length of video inputs')
parser.add_argument('--max_patches_h', default=7, type=int, help='max height of patches')
parser.add_argument('--max_patches_w', default=7, type=int, help='max width of patches')
parser.add_argument('--image_feature_dim', default=None, type=int, help='image feature dim produced by the image peripheral')
parser.add_argument('--image_feature_map_layer', default=None, type=int, help='num of blocks in the image peripheral')
parser.add_argument('--no_BOS_EOS', action='store_true', help='true if tokenize sentence without BOS and EOS')
parser.add_argument('--greedy_only', action='store_true', help='true if not train with target')
parser.add_argument('--optim', default='adam', type=str, help='type of optimizer')
parser.add_argument('--save_processed_img', action='store_true', help='true if save processed images')
parser.add_argument('--cca_caches', default=['spatial','temporal','structured'], nargs='+', type=str)
parser.add_argument('--cca_streams', default=None, nargs='+', type=str)
parser.add_argument('--pos_emb_streams', default=None, nargs='+', type=str)
parser.add_argument('--patch_pos', action='store_true', help='true if use learnable position embeddings in self attn')
parser.add_argument('--no_patch_emb_pos', action='store_true', help='true if use NO position embeddings for patches')
parser.add_argument('--default_attn_blocks', action='store_true', help='true if use default attn blocks in CCA')
parser.add_argument('--one_hot_only', action='store_true', help='true if only use one-hot encoded structured data')
parser.add_argument('--entity_only', action='store_true', help='true if only use entity encoded structured data')
parser.add_argument('--entity_pretrained_emb_path', default=None, type=str, help='pretrained embedding path')
parser.add_argument('--tune_steps', default=None, type=int, help='number of iterations to run during Ray tune')
parser.add_argument('--l1reg', default=None, type=float, help='l1 regularization factor')
parser.add_argument('--no_patch', action='store_true', help='true if deactivate patching')
parser.add_argument('--freeze_layers_before_CCA', action='store_true', help='true if freeze layer before CCA blocks')
parser.add_argument('--freeze_layers_all', action='store_true', help='true if freeze all omninet layers')
parser.add_argument('--freeze_layers_before_temporal_encoder_layer6', action='store_true', help='true if freeze before temporal_encoder layer 6')
parser.add_argument('--use_vit_mlp', action='store_true', help='true if use vit mlp in TemporalEncoder')
parser.add_argument('--switch_cca_omninet_freq', default=0, type=int, help='frequence of freezing/unfreezing cca/omninet layers')
parser.add_argument('--two_ques_file', default=None, type=str, help='file name of two_ques data if performing two_ques experiments on vqa, where we randomly add an irrelavant question to each question.')
# parser.add_argument('--two_ques_separate', action='store_true', help='true if encode two questions separately')
parser.add_argument('--aug_text_file', default=None, type=str, help='file name of augmented text')
parser.add_argument('--save_cca_attn', action='store_true', help='true if save cca attention scores')

args = parser.parse_args()

data_path = args.data_path
coco_images = os.path.join(data_path, 'coco/train_val')
caption_dir = os.path.join(data_path, 'coco')
vqa_dir = os.path.join(data_path, 'vqa')
socialiq_dir = os.path.join(data_path, 'socialiq')
socialiq_video_folder = 'vision/videos_1fps_640-360_resized'

structured_path = None
num_cat_dict = {}
if args.structured_folder is not None:
    structured_path = os.path.join(data_path, args.structured_folder, 'data.pkl')
    with open(os.path.join(data_path, args.structured_folder, 'num_cat_dict.pkl'), 'rb') as f:
        num_cat_dict = pickle.load(f)

hmdb_data_dir = data_path
hmdb_process_dir = os.path.join(data_path, 'hmdbprocess')
penn_data_dir = os.path.join(data_path, 'penn')

# coco_images = 'data/coco/train_val'
# caption_dir = 'data/coco'
# vqa_dir = 'data/vqa'
# model_save_path = 'checkpoints'
# hmdb_data_dir='data/hmdb'
# hmdb_process_dir='data/hmdbprocess'
# penn_data_dir='data/penn'

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(int(args.weight_seed))

def set_config(config, conf_type='default'):
    config[0]['use_temporal_encoder'] = not args.no_temporal_encoder
    config[0]['use_vit_mlp'] = args.use_vit_mlp
    config[0]['dropout'] = args.dropout
    config[0]['more_dropout'] = args.more_dropout
    config[0]['temporal_n_layers'] = args.temp_enc_n_layers
    config[0]['temporal_n_heads'] = args.temp_enc_n_heads
    config[0]['decoder_n_layers'] = args.dec_n_layers
    config[0]['decoder_n_heads'] = args.dec_n_heads
    if args.patch_size is not None:
        config[0]['patch_sizes'] = (args.patch_size, args.patch_size)
    config[0]['patch_stride'] = args.patch_stride
    config[0]['max_patches_h'] = args.max_patches_h
    config[0]['max_patches_w'] = args.max_patches_w
    if args.image_feature_dim is not None:
        config[1]['image_feature_dim'] = args.image_feature_dim
    if args.image_feature_map_layer is not None:
        config[1]['image_feature_map_layer'] = args.image_feature_map_layer
    config[1]['no_BOS_EOS'] = args.no_BOS_EOS
    config[1]['num_cat_dict'] = num_cat_dict
    config[1]['one_hot_only'] = args.one_hot_only
    if 'cca' in conf_type:
        config[0]['dropout_p'] = args.dropout_p
        config[0]['dropout_s'] = args.dropout_s
        config[0]['dropout_t'] = args.dropout_t
        config[0]['drop_path_rate'] = args.drop_path_rate
        config[0]['sa_drop_path_rate'] = args.sa_drop_path_rate
        config[0]['cca_n_layers'] = args.cca_n_layers
        config[0]['cca_n_heads'] = args.cca_n_heads
        config[0]['sa_n_layers'] = args.sa_n_layers
        config[0]['sa_n_heads'] = args.sa_n_heads
        config[0]['psa_n_layers'] = args.psa_n_layers
        config[0]['psa_n_heads'] = args.psa_n_heads
        config[0]['cca_hidden_dim'] = args.cca_hidden_dim
        config[0]['cca_caches'] = args.cca_caches
        config[0]['cca_streams'] = args.cca_streams
        config[0]['pos_emb_streams'] = args.pos_emb_streams
        config[0]['default_attn_blocks'] = args.default_attn_blocks
        config[0]['use_patch'] = not args.no_patch
        config[0]['save_cca_attn'] = args.save_cca_attn
        config[0]['patch_pos'] = args.patch_pos
        config[0]['patch_emb_pos'] = not args.no_patch_emb_pos
        config[0]['max_clip_len'] = args.max_clip_len
        config[0]['sa_on_whole_cache'] = not args.not_sa_on_whole_cache
    if 'struct' in conf_type:
        config[1]['entity_pretrained_emb_path'] = args.entity_pretrained_emb_path
        config[1]['se_dropout'] = args.se_dropout

def one_hot_encoding(a, c):
    return np.hstack([one_hot(a[:,i], c[i]) for i in range(a.shape[1])])

def one_hot(a, c):
    b = np.zeros((a.size, c))
    b[np.arange(a.size), a]=1
    return b

def print_log(s, fn):
    with open(fn, 'a') as f:
        print(s, file=f)

def write_attn(out_dir,attn):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
   
    ckpts = glob.glob(os.path.join(out_dir, '*'))
    batches = [int(os.path.basename(c)) for c in ckpts]
    if len(batches) == 0:
        last_b = 0
    else:
        last_b = max(batches)
    with open(os.path.join(out_dir,str(last_b+1)), 'wb') as f:
        pickle.dump(attn, f)

def find_name(names, pattens):
    for p in pattens:
        if p in names:
            return True
    return False

def layer_switch(model, layers, freeze=False):
    for name, param in model.named_parameters():
        if find_name(name, layers):
            param.requires_grad = False if freeze else True

layers_periph = ['image_input_perph.output_fc', 'english_language_perph.embed_layer', 'english_language_perph.output', 'german_language_perph.embed_layer', 'german_language_perph.output', 'control_peripheral']
layers_encode = ['input_spatial_proj','inpcont_input_proj', 'input_temporal_proj', 'temporal_encoder.layer_stack']
layers_decode = ['decoder.layer_stack', 'decoder.output_fc', 'emb_decoder_proj', 'output_embs', 'output_clfs', 'cont_decoder_proj']
layers_cca = ['cca'] 

layers_before_cca = layers_periph + layers_encode
layers_omninet = layers_periph + layers_encode + layers_decode #['input_spatial_proj','inpcont_input_proj', 'input_temporal_proj', 'temporal_encoder', 'image_input_perph', 'english_language_perph', 'german_language_perph', 'control_peripheral', 'decoder', 'emb_decoder_proj', 'output_embs']
layers_before_temporal_encoder_l6 = layers_periph + layers_encode[:-1] + ['cnp.temporal_encoder.layer_stack.%s.'%li for li in range(6)]

def train(shared_model, task, batch_size, train_steps, gpu_id, start,  restore,  counter, barrier=None, save_interval=None,
          eval_interval=None, log=True, eval_first=False):
    log_dir = 'logs/%s' % task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (log == True):
        summary_writer = SummaryWriter(log_dir)
    # Create local model
     
    # torch.manual_seed(int(random.random() * 1000))
    if gpu_id>0:
        if args.conf_type == 'default':
            config = defaultconf()
        elif args.conf_type == 'default_struct':
            config = default_struct()
        elif args.conf_type == 'cca':
            config = cca_config()
        elif args.conf_type == 'cca_struct':
            config = cca_struct_config()
        set_config(config, args.conf_type)
        model = omninet.OmniNet(gpu_id=gpu_id, config=config)
        model=model.cuda(gpu_id)
    else:
        #For GPU 0, use the shared model always
        model=shared_model

    freeze_layer_names = []
    if args.freeze_layers_before_CCA:
        freeze_layer_names = layers_before_cca 
    if args.freeze_layers_all:
        freeze_layer_names = layers_omninet 
    if args.freeze_layers_before_temporal_encoder_layer6:
        freeze_layer_names = layers_before_temporal_encoder_l6 

    layer_switch(model, freeze_layer_names, freeze=True)
    # for name, param in model.named_parameters():
    #     if find_name(name, freeze_layer_names):
    #         param.requires_grad = False

    if task == 'caption':
        DL,val_dl = dl.coco_cap_batchgen(caption_dir=caption_dir, image_dir=coco_images,
                                  num_workers=args.n_workers,
                                  batch_size=batch_size)
        
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=args.init_lr)
    elif task == 'socialiq':
        DL, val_dl, test_dl = dl.social_iq_batchgen(data_dir=socialiq_dir, video_folder=socialiq_video_folder, num_workers=args.n_workers, batch_size=batch_size, clip_len=args.max_clip_len, data_seed=int(args.data_seed+restore))
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
            512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr)

    elif task == 'vqa':
        vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        with open(vocab_file,'rb') as f:
                ans_to_id,id_to_ans=pickle.loads(f.read())
        vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)

        DL,val_dl,val_non_ma_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=args.n_workers, batch_size=batch_size, data_seed=int(args.data_seed+restore), structured_path=structured_path, two_ques_file=args.two_ques_file, aug_text_file=args.aug_text_file)
        # DL,val_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=args.n_workers, batch_size=batch_size, data_seed=int(args.data_seed+restore), structured_path=structured_path)
        if args.optim == 'adam':
            if args.switch_cca_omninet_freq != 0:
                layer_switch(shared_model, layers_cca, freeze=True)
            omni_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        elif args.optim == 'adamw':
            omni_optimizer = ScheduledOptim(
                AdamW(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        optimizer = omni_optimizer
        if os.path.exists(os.path.join(args.model_save_path,str(restore),'optimizer.pth')):
            optimizer.restore(args.model_save_path, restore)
        if args.switch_cca_omninet_freq != 0:
            layer_switch(shared_model, layers_cca, freeze=False)
            layer_switch(shared_model, layers_omninet, freeze=True)
            cca_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay_cca),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr_cca,name='cca_optimizer')
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'cca_optimizer.pth')):
                cca_optimizer.restore(args.model_save_path, restore)
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'omni_on.pkl')):
                with open(os.path.join(args.model_save_path,str(restore),'omni_on.pkl'), 'rb') as f:
                    omni_on = pickle.load(f)
            if not omni_on:
                optimizer = cca_optimizer
               
    elif task == 'hmdb':
        DL,val_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=args.n_workers,batch_size=batch_size,
                                   test_batch_size=int(batch_size/4),
                                   clip_len=16)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'penn':
        DL,val_dl,test_dl=dl.penn_dataloader(penn_data_dir,batch_size=batch_size,
                                             test_batch_size=int(batch_size/2),num_workers=args.n_workers,vocab_file='conf/penn_vocab.json')
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=args.init_lr)
    
    if task != 'vqa' and os.path.exists(os.path.join(args.model_save_path,str(restore),'optimizer.pth')):
        optimizer.restore(args.model_save_path, restore)
    # if restore != 0:
        # if args.save_best:
        #     optimizer.restore(args.model_save_path, 'last/0')
        # else:

        # tr_dl.dataset.resume_on()
        # val_dl.dataset.resume_on()
        # for i in range(1, restore+2):
        #     if evaluating(log, eval_interval, i):
        #         if i == (restore+1) and eval_first:
        #             continue
        #         for b in val_dl:
        #             pass
        #         continue
        #     batch = next(DL)
        # val_dl.dataset.resume_off()
        # tr_dl.dataset.resume_off()

    if restore == 0:
        print('num of parameters:', sum(p.numel() for p in shared_model.parameters() if p.requires_grad))
        print_log('num of parameters:{}'.format(sum(p.numel() for p in shared_model.parameters() if p.requires_grad)), args.model_save_path+'.log')
        print_log('num of parameters:{}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)), args.model_save_path+'.log')
        print_log('\n'.join([name for name,p in shared_model.named_parameters() if p.requires_grad]), args.model_save_path+'_trainable_params.log')

    if os.path.exists(args.model_save_path + '/best/acc.pkl'):
        with open(args.model_save_path + '/best/acc.pkl', 'rb') as f:
            acc = pickle.load(f)
        best_val_acc = acc['best_val_acc']
    else:
        best_val_acc = 0
    
    log_str = ''

    model=model.train()
    if args.switch_cca_omninet_freq != 0:
        log_str += 'switch omninet to: %s\n'%omni_on
        print('switch omninet to: %s'%omni_on)

    for i in range(start+1, train_steps):
        if args.switch_cca_omninet_freq != 0 and i % args.switch_cca_omninet_freq == 0:
            omni_on = not omni_on
            optimizer = omni_optimizer if omni_on else cca_optimizer
            layer_switch(model,layers_omninet,freeze=not omni_on)
            layer_switch(model,layers_cca,freeze=omni_on)
            log_str += 'switch omninet to: %s\n'%omni_on
            print('switch omninet to: %s'%omni_on)
            # print_log('switch omninet to: %s\n'%omni_on, args.model_save_path+'_debugging.log')
            # print_log('optimizer: %s\n'%optimizer.name, args.model_save_path+'_debugging.log')
            # print_log('\n'.join([name for name,p in shared_model.named_parameters() if p.requires_grad]), args.model_save_path+'_debugging.log')
            # print_log('-'*100+'\n', args.model_save_path+'_debugging.log')

        if args.tune_steps is not None and i - start > args.tune_steps:
            break
        model.zero_grad()
        optimizer.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()
        if task == 'caption':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    if gpu_id>=0:
                        imgs=imgs.cuda(device=gpu_id)
                    captions = b['cap']
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    _,loss,acc = r.image_caption(model, imgs, targets=captions, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, COCO validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
            captions = batch['cap']
            _, loss,acc = r.image_caption(model, imgs, targets=captions)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, Caption Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'socialiq':
            if (log and eval_interval is not None and i % eval_interval == 0 and i >= args.eval_start):
                if i == (start+1) and not eval_first:
                    continue
                start_time = time.time()
                model = model.eval()
                val_loss = 0
                val_acc=0
                n_correct = 0
                n_total = 0
                log_str += '-'*100 + '\nEvaluation step\n'
                print('-'*100 + '\nEvaluation step')
                for b in val_dl:
                    imgs = b['videos']
                    labels = b['labels']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)
                    questions = b['ques']
                    answers = b['ans']

                    pred, loss, acc = r.socialiq(model, imgs, questions, answers, targets=labels,mode='val',return_str_preds=True, greedy_only=args.greedy_only)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc += acc
                    bs = labels.shape[0]
                    n_correct += acc * bs
                    n_total += bs
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                val_acc1=n_correct/n_total
                summary_writer.add_scalar('Val_loss', val_loss, step)

                log_str += 'Step %d, SIQ validation loss: %f, Accuracy %f %%\n' % (step, val_loss,val_acc1)
                log_str += '-'*100 + '\n' + 'Evaluation takes: {:.8f}s\n'.format(time.time() - start_time)
                print('Step %d, SIQ validation loss: %f, Accuracy %f %%, Accuracy1 %f %%' % (step, val_loss,val_acc,val_acc1))
                print('-'*100 )

                print('Evaluation takes: {:.8f}s'.format(time.time() - start_time))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iteration = step-1
                    log_str += 'best_iteration:{}\n'.format(best_iteration)
                    print('best_iteration:{}'.format(best_iteration))

                    shared_model.save(args.model_save_path, 'best/0')
                    optimizer.save(args.model_save_path, 'best/0')

                    with open(args.model_save_path + '/best/acc.pkl', 'wb') as f:
                        pickle.dump({'best_val_acc': best_val_acc, 'best_iteration': best_iteration}, f)

                print_log(log_str, args.model_save_path+'.log')
                log_str = ''

                model = model.train()
                continue
            batch = next(DL)
            imgs = batch['videos']
            labels = batch['labels']
            if gpu_id >= 0:
                imgs = imgs.cuda(device=gpu_id)
                labels = labels.cuda(device=gpu_id)
            questions = batch['ques']
            answers = batch['ans']
            _, loss,acc = r.socialiq(model, imgs, questions, answers, targets=labels, greedy_only=args.greedy_only)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            log_str += 'Step %d, SIQ Loss: %f, Accuracy:  %f %%\n' % (step, loss,acc)
            print('Step %d, SIQ Loss: %f, Accuracy:  %f %%' % (step, loss,acc))

        elif task == 'vqa':
            if (log and eval_interval is not None and i % eval_interval == 0 and i >= args.eval_start):
                if i == (start+1) and not eval_first:
                    continue
                model = model.eval()
                val_loss = 0
                val_acc=0
                val_l1_loss_struct = 0
                predictions = []
                ans = []
                ans_str = []
                ques_ids = []
                # print('-' * 100)
                # print('Evaluation step')
                log_str += '-'*100 + '\nEvaluation step\n'
                for b in tqdm(val_dl):
                    imgs = b['img']
                    answers=b['ans']
                    if len(b['struct']) != 0:
                        struct = b['struct'].long() if not args.one_hot_only else None
                        struct_one_hot = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
                    else:
                        struct = None
                        struct_one_hot = None
                    # struct=b['struct'] if len(b['struct']) != 0 else None
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        answers=answers.cuda(device=gpu_id)
                        if struct_one_hot is not None:
                            struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                        if struct is not None:
                            struct = struct.cuda(device=gpu_id)
                    questions= b['ques']
                    # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
                    #     questions = [q.split('? ') for q in b['ques']]
                    #     questions = [[q[0]+'?',q[1]] for q in questions]
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    vqa_out = r.vqa(model, imgs, questions,targets=answers,structured=struct,structured_one_hot=struct_one_hot,mode='val',return_str_preds=True, greedy_only=args.greedy_only)
                    pred, loss,acc,l1_loss_struct = vqa_out[:4]
                    if len(vqa_out) == 5:
                        cca_attns = vqa_out[4]
                        pt_attn = cca_attns['pt'].detach().cpu().numpy()
                        tp_attn = cca_attns['tp'].detach().cpu().numpy()
                        ps_attn = cca_attns['ps'].detach().cpu().numpy() if cca_attns['ps'] is not None else None
                        ts_attn = cca_attns['ts'].detach().cpu().numpy() if cca_attns['ts'] is not None else None
                        ques_id = b['ques_id']
                        vocab_file=os.path.join('conf/vqa_vocab.pkl')
                        with open(vocab_file,'rb') as f:
                            ans_to_id,id_to_ans=pickle.loads(f.read())
                        ytrue = torch.reshape(answers, [-1]).cpu().numpy()
                        preds = pred.detach().cpu().squeeze().tolist()
                        preds_str = [id_to_ans[pr] for pr in preds]
                        ytrue_str = [id_to_ans[yt] for yt in ytrue]
                        save_encoding_ques_ids = [21671000, 63525001, 341393003, 206922001, 443713002, 339705011, 349321003, 268539002, 563603000, 493724001, 288229001]
                        save_encoding = False
                        for qid in ques_id:
                            if qid in save_encoding_ques_ids:
                                save_encoding = True
                                break
                        if save_encoding:
                            write_attn(args.model_save_path+'_cca_attns_val_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue, 'temporal_cache': cca_attns['t_cache'].detach().cpu().numpy(), 'spatial_cache': cca_attns['sp_cache'].detach().cpu().numpy()})
                        else:
                            write_attn(args.model_save_path+'_cca_attns_val_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                    val_loss += float(loss.detach().cpu().numpy())
                    if args.l1reg is not None:
                        struct_periph_l1_loss = sum([torch.sum(torch.abs(p)) for n,p in shared_model.named_parameters() if 'struct_periph' in n])
                        val_l1_loss_struct += args.l1reg*(float(struct_periph_l1_loss.detach().cpu().numpy()) + float(l1_loss_struct.detach().cpu().numpy()))
                    val_acc+=acc
                    ans += b['ans'].squeeze().tolist()
                    ans_str += b['ans_str']
                    ques_ids += b['ques_id']
                    predictions += pred.detach().cpu().squeeze().tolist()
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)

                non_ma_predictions = []
                non_ma_ans = []
                non_ma_ans_str = []
                non_ma_ques_ids = []
                all_simple_acc = 0
                vqa_eval_overall_acc = 0
                vqa_eval_simple_acc = 0
                for b in tqdm(val_non_ma_dl):
                    imgs = b['img']
                    answers=b['ans']
                    if len(b['struct']) != 0:
                        struct = b['struct'].long() if not args.one_hot_only else None
                        struct_one_hot = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
                    else:
                        struct = None
                        struct_one_hot = None
                    # struct=b['struct'] if len(b['struct']) != 0 else None
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        answers=answers.cuda(device=gpu_id)
                        if struct_one_hot is not None:
                            struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                        if struct is not None:
                            struct = struct.cuda(device=gpu_id)
                    questions= b['ques']
                    # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
                    #     questions = [q.split('? ') for q in b['ques']]
                    #     questions = [[q[0]+'?',q[1]] for q in questions]
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    vqa_out = r.vqa(model, imgs, questions,targets=answers,structured=struct,structured_one_hot=struct_one_hot,mode='val',return_str_preds=True, greedy_only=args.greedy_only)
                    pred, loss,acc,l1_loss_struct = vqa_out[:4]
                    if len(vqa_out) == 5:
                        cca_attns = vqa_out[4]
                        pt_attn = cca_attns['pt'].detach().cpu().numpy()
                        tp_attn = cca_attns['tp'].detach().cpu().numpy()
                        ps_attn = cca_attns['ps'].detach().cpu().numpy() if cca_attns['ps'] is not None else None
                        ts_attn = cca_attns['ts'].detach().cpu().numpy() if cca_attns['ts'] is not None else None
                        ques_id = b['ques_id']
                        vocab_file=os.path.join('conf/vqa_vocab.pkl')
                        with open(vocab_file,'rb') as f:
                            ans_to_id,id_to_ans=pickle.loads(f.read())
                        ytrue = torch.reshape(answers, [-1]).cpu().numpy()
                        preds = pred.detach().cpu().squeeze().tolist()
                        preds_str = [id_to_ans[pr] for pr in preds]
                        ytrue_str = [id_to_ans[yt] for yt in ytrue]
                        write_attn(args.model_save_path+'_cca_attns_val_non_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                    non_ma_predictions += pred.detach().cpu().squeeze().tolist()
                    non_ma_ans += b['ans'].squeeze().tolist()
                    non_ma_ans_str += b['ans_str']
                    non_ma_ques_ids += b['ques_id']
                
                all_predictions = predictions + non_ma_predictions
                all_ans = ans + non_ma_ans
                all_ans_str = ans_str + non_ma_ans_str
                all_ques_ids = ques_ids + non_ma_ques_ids
                all_simple_acc = 100*np.sum(np.array(all_predictions)==np.array(all_ans))/len(all_predictions)

                result_json = []
                for j in range(len(all_ques_ids)):
                    result_json.append({'question_id':all_ques_ids[j],'answer':id_to_ans[all_predictions[j]]})

                if not os.path.exists(os.path.join(args.model_save_path,'res')):
                    os.makedirs(os.path.join(args.model_save_path,'res'))
                with open(os.path.join(args.model_save_path, 'res/vqa_prediction.json'), 'w') as outfile:
                    json.dump(result_json, outfile)

                vqa_predictions = []
                vqaRes=vqa.loadRes(os.path.join(args.model_save_path, 'res/vqa_prediction.json'),vqa_val_ques)
                vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
                with open(os.path.join(args.model_save_path, 'res/vqa_prediction.json'), 'r') as f:
                    json_ans = json.load(f)
                for j in json_ans:
                    vqa_predictions.append(j['answer'])
                vqa_predictions=np.array(vqa_predictions)
                # print('vqa_eval_simple_accuracy:', np.sum(vqa_predictions==np.array(all_ans))/vqa_predictions.shape[0])
                vqa_eval_simple_acc = 100*np.sum(vqa_predictions==np.array(all_ans_str))/vqa_predictions.shape[0]
                # evaluate results
                vqaEval.evaluate()
                # print('Overall Accuracy is: %.02f\n' %(vqaEval.accuracy['overall']))
                vqa_eval_overall_acc = vqaEval.accuracy['overall']

                # all_simple_acc = 0
                # vqa_eval_simple_acc = 0
                # vqa_eval_overall_acc = 0

                if args.l1reg is None:
                    # print('Step %d, VQA validation loss: %f, Accuracy %f %%, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%' % (step, val_loss,val_acc,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc))
                    # print('-' * 100)
                    log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%\n' % (step, val_loss,val_acc,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc)
                    log_str += '-'*100 + '\n'
                else:
                    # print('Step %d, VQA validation loss: %f, Accuracy %f %%, L1 loss struct: %f, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%' % (step, val_loss,val_acc,val_l1_loss_struct,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc))
                    # print('-' * 100)
                    log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%, L1 loss struct: %f, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%\n' % (step, val_loss,val_acc,val_l1_loss_struct,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc)
                    log_str += '-'*100 + '\n'

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iteration = step-1
                    # print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    shared_model.save(args.model_save_path, 'best/0')
                    optimizer.save(args.model_save_path, 'best/0')

                    with open(args.model_save_path + '/best/acc.pkl', 'wb') as f:
                        pickle.dump({'best_val_acc': best_val_acc, 'best_iteration': best_iteration}, f)

                print_log(log_str, args.model_save_path+'.log')
                log_str = ''

                model = model.train()
                continue
            # if args.tune_steps is not None and i % 50 == 0:
            #     print_log(log_str, args.model_save_path+'.log')
            #     log_str = ''

            batch = next(DL)
            cur_epoch = step // eval_interval
            if args.save_processed_img:
                img_save_dir = '/scratch1/yxuea/data/processed_vqa_images/%s'%cur_epoch
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                for j, ques_id in enumerate(batch['ques_id']):
                    with open(os.path.join(img_save_dir,'%s.npy'%ques_id), 'wb') as f:
                        np.save(f, batch['img'][j].numpy()) 
            if len(batch['struct']) != 0:
                struct = batch['struct'].long() if not args.one_hot_only else None
                struct_one_hot = torch.as_tensor(one_hot_encoding(batch['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
            else:
                struct = None
                struct_one_hot = None
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
                answers = batch['ans'].cuda(device=gpu_id)
                if struct_one_hot is not None:
                    struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                if struct is not None:
                    struct = struct.cuda(device=gpu_id)
                # struct = batch['struct'].cuda(device=gpu_id) if len(batch['struct']) != 0 else None
            else:
                imgs = batch['img']
                answers = batch['ans']
                # struct = batch['struct'] if len(batch['struct']) != 0 else None
            questions = batch['ques']                       
            # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
            #     questions = [q.split('? ') for q in batch['ques']]
            #     questions = [[q[0]+'?',q[1]] for q in questions]
            #     if i == 10:
            #         print(questions[0])
            #         print(questions[-1])
            # print('questions[0]:', questions[0])
            # print('answers[0]', answers[0])
            # start_time = time.time()
            vqa_out = r.vqa(model, imgs, questions, structured=struct, structured_one_hot=struct_one_hot, targets=answers, greedy_only=args.greedy_only)
            pred, loss,acc,l1_loss_struct = vqa_out[:4]
            if len(vqa_out) == 5:
                cca_attns = vqa_out[4]
                pt_attn = cca_attns['pt'].detach().cpu().numpy()
                tp_attn = cca_attns['tp'].detach().cpu().numpy()
                ps_attn = cca_attns['ps'].detach().cpu().numpy() if cca_attns['ps'] is not None else None
                ts_attn = cca_attns['ts'].detach().cpu().numpy() if cca_attns['ts'] is not None else None
                ques_id = batch['ques_id']
                vocab_file=os.path.join('conf/vqa_vocab.pkl')
                with open(vocab_file,'rb') as f:
                    ans_to_id,id_to_ans=pickle.loads(f.read())
                ytrue = torch.reshape(answers, [-1]).cpu().numpy()
                preds=pred.argmax(-1)
                preds=torch.reshape(preds,[-1]).cpu().numpy()
                preds_str = [id_to_ans[pr] for pr in preds]
                ytrue_str = [id_to_ans[yt] for yt in ytrue]
                write_attn(args.model_save_path+'_cca_attns_tr', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                write_attn(args.model_save_path+'_cca_attns_tr', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                if i - start > 5:
                    raise Exception('a few training attn saved. stopped training.')
            if args.l1reg is not None:
                struct_periph_l1_loss = sum([torch.sum(torch.abs(p)) for n,p in shared_model.named_parameters() if 'struct_periph' in n])
                loss += args.l1reg * (struct_periph_l1_loss + l1_loss_struct)
            loss.backward()
            # end_time = time.time()
            # log_str += 'forward/backward pass takes {:.8f}s\n'.format(end_time - start_time)
            # print('forward/backward pass takes {:.8f}s'.format(end_time - start_time))
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            if args.l1reg is None:
                # print('Step %d, VQA Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
                log_str += 'Step %d, VQA Loss: %f, Accuracy:  %f %%\n' % (step, loss,acc)
            else:
                # print('Step %d, VQA Loss: %f, Accuracy:  %f %%, L1 loss struct: %f' % (step, loss,acc,args.l1reg * (struct_periph_l1_loss + l1_loss_struct)))
                log_str += 'Step %d, VQA Loss: %f, Accuracy:  %f %%, L1 loss struct: %f\n' % (step, loss,acc,args.l1reg * (struct_periph_l1_loss + l1_loss_struct))
        elif task=='hmdb':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    vid,labels = b
                    if gpu_id >= 0:
                        vid = vid.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)    
                    _, loss,acc = r.hmdb(model, vid,targets=labels, mode='val')
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, HMDB validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            vid,labels = next(DL)
            if gpu_id >= 0:
                vid = vid.cuda(device=gpu_id)
                labels = labels.cuda(device=gpu_id)    
            _, loss,acc = r.hmdb(model, vid,targets=labels,return_str_preds=True)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, HMDB Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'penn':
            if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(test_dl):
                    en = b['text']
                    targets = b['tokens']
                    pad_id=b['pad_id']
                    pad_mask=b['pad_mask']
                    if gpu_id>=0:
                        targets=targets.to(gpu_id)
                        pad_mask=pad_mask.to(gpu_id)
                    _,loss,acc = r.penn(model, en, target_pad_mask=pad_mask,
                                        pad_id=pad_id,targets=targets, mode='val',return_str_preds=True)
                    loss=loss.detach()
                    val_loss += float(loss.cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, PENN validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            en = batch['text']
            targets = batch['tokens']
            pad_id=batch['pad_id']
            pad_mask=batch['pad_mask']
            if gpu_id>=0:
                targets=targets.to(gpu_id)
                pad_mask=pad_mask.to(gpu_id)
            _, loss,acc = r.penn(model, en, pad_id=pad_id, targets=targets,target_pad_mask=pad_mask)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, PENN Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        # End Calculate loss
        if gpu_id>0:
            ensure_shared_grads(model, shared_model, gpu_id)
        optimizer.step()
        # Save model
        if (save_interval != None and (i + 1) % save_interval == 0):
            # if not args.save_best:
            if step > 2*save_interval and not args.save_all:
                os.rename(os.path.join(args.model_save_path,str(int(step-2*save_interval))),
                    os.path.join(args.model_save_path,str(step)))

            shared_model.save(args.model_save_path, step)
            optimizer.save(args.model_save_path, step)
    
            print_log(log_str, args.model_save_path+'.log')
            log_str = ''

            # if args.save_best and ((i+1)//save_interval + 1) * save_interval >= train_steps:
            #     shared_model.save(args.model_save_path, 'last/0')
            #     optimizer.save(args.model_save_path, 'last/0')

        sys.stdout.flush()

if __name__ == '__main__':
    mp.set_start_method('spawn',force=True)
    n_iters = int(args.n_iters)
    n_jobs = int(args.n_jobs)
    tasks=args.tasks
    batch_sizes=args.batch_sizes
    save_interval = int(int(args.save_interval) / n_jobs)
    eval_interval = int(int(args.eval_interval) / n_jobs)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if args.restore_last == True:
        ckpts = glob.glob(os.path.join(args.model_save_path, '*'))
        iters = [int(os.path.basename(c)) for c in ckpts if os.path.basename(c) not in ['best','res']]
        if len(iters) != 0:
            restore = max(iters)
        else:
            restore = -1
    else:
        restore = int(args.restore)
    tasks=tasks.split(',')
    tasks=[t.strip() for t in tasks]
    batch_sizes=batch_sizes.split(',')
    batch_sizes=[int(b.strip()) for b in batch_sizes]

    if len(tasks)!=len(batch_sizes):
        raise Exception('Number of tasks provided does not match the number of batch sizes provided.')

    n_gpus = int(args.n_gpus)
    n_tasks = len(tasks) * n_jobs

    if args.conf_type == 'default':
        config = defaultconf()
    elif args.conf_type == 'default_struct':
        config = default_struct()
    elif args.conf_type == 'cca':
        config = cca_config()
    elif args.conf_type == 'cca_struct':
        config = cca_struct_config()
    set_config(config, args.conf_type)

    shared_model = omninet.OmniNet(gpu_id=0, config=config)

    eval_first = False
    if restore != -1:
        # if args.save_best:
        #     shared_model.restore(args.model_save_path, 'last/0')
        # else:
        shared_model.restore(args.model_save_path, restore)
        if os.path.exists(args.model_save_path+'.log'):
            with open(args.model_save_path + '.log', 'r') as f:
                log = f.read()
            if len(re.findall('Step %s,'%(restore+1), log)) == 0:
                eval_first = True
    else:
        restore=0

    if restore == 0:
        print(config)
        print(args)
        print_log(config, args.model_save_path+'.log')
        print_log(args, args.model_save_path+'.log')
        
    shared_model=shared_model.to(0)
    shared_model.share_memory()
    counters = [Counter(restore) for i in range(len(tasks))]
    barrier = mp.Barrier(n_tasks)
    start = int(restore / n_jobs)
    # Declare training processes for multi-gpu hogwild training
    processes = []
    for i in range(n_tasks):
        #If more than one GPU is used, use first GPU only for model sharing
        if n_gpus>1:
            gpu_id=i%n_gpus
        else:
            gpu_id=0
        process = mp.Process(target=train, args=(shared_model, tasks[i % len(tasks)], batch_sizes[i % len(tasks)],
                                                 int(n_iters / n_jobs),
                                                 gpu_id, start, restore, counters[i % len(tasks)], barrier,
                                                 (save_interval if i == 0 else None),
                                                 (eval_interval if i < len(tasks) else None),
                                                 (True if i < len(tasks) else False),
                                                 eval_first))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
