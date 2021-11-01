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

OmniNet API

"""
import torch
import os
import torch.nn as nn

from .peripherals import *
from .util import *
from .cnp import CNP

class OmniNet(nn.Module):

    def __init__(self, config=None, gpu_id=-1,dropout=None):
        super(OmniNet, self).__init__()
        if config is None:
            cc, pc, d = self.__defaultconf__()
        else:
            cc, pc, d = config
        if dropout is not None:
            cc['dropout']=dropout
            pc['dropout']=dropout
        self.gpu_id = gpu_id
        tasks = {'PENN': pc['penn_output_classes'], 'HMDB':pc['hmdb_output_classes'],
                 'IMAGE_CAPTION':pc['english_language_output_vocab'],'VQA':pc['vqa_output_vocab'],
                 'SIQ':pc['SIQ_output_classes'],'VG':pc['VG_output_classes']}
        self.cnp = CNP(tasks,conf=cc,domains=d, gpu_id=gpu_id)
        
        self.image_input_perph = ImageInputPeripheral(output_dim=cc['input_dim'], 
                                                      feature_dim=pc['image_feature_dim'],
                                                      feature_map_layer=pc['image_feature_map_layer'],         
                                                      dropout=pc['dropout'],freeze_layers=True)
        self.english_language_perph = LanguagePeripheral(vocab_size=pc['english_language_input_vocab'],
                                                                     embed_dim=pc['english_language_input_embed'],
                                                                     output_dim=cc['input_dim'],
                                                                     lang='en', no_BOS_EOS=pc['no_BOS_EOS'],
                                                                     gpu_id=gpu_id,dropout=pc['dropout'])
        self.german_language_perph = LanguagePeripheral(vocab_size=pc['german_language_input_vocab'],
                                                                    embed_dim=pc['german_language_input_embed'],
                                                                    output_dim=cc['input_dim'],
                                                                    lang='de', no_BOS_EOS=pc['no_BOS_EOS'],
                                                                    gpu_id=gpu_id)
        raw_struct_dim=sum(pc['num_cat_dict'].values())+pc['num_conti'] 
        if raw_struct_dim != 0:
            self.struct_periph = StructuredPeripheral(output_dim=cc['structured_dim'], raw_struct_dim=raw_struct_dim,dropout=pc['dropout'])
        if len(pc['num_cat_dict']) != 0 and not pc['one_hot_only']:
            self.struct_entity_periph = StructuredEntityPeripheral(output_dim=cc['structured_dim'], num_cat_dict=pc['num_cat_dict'], pretrained_path=pc['entity_pretrained_emb_path'], dropout=pc['se_dropout']) #, gpu_id=gpu_id)

    def reset(self,batch_size):
        self.cnp.reset(batch_size)

    def encode_videos(self,videos,domain='IMAGE'):
        video_encodings,h,w = self.image_input_perph.encode(videos)
        video_encodings = self.cnp.encode_with_patch(video_encodings,(h,w))
        self.cnp.encode(video_encodings,domain=domain)

    def encode_images(self,images,domain='IMAGE'):
        image_encodings,h,w = self.image_input_perph.encode(images)
        image_encodings = self.cnp.encode_with_patch(image_encodings,(h,w))
        self.cnp.encode(image_encodings,domain=domain)
    
    def encode_englishtexts(self,texts,domain='ENGLISH',sa=False):
        sent_encodings,input_pad_mask=self.english_language_perph.embed_sentences(texts)
        self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain, sa=sa)
    
    def encode_structured(self, structured_one_hot, structured=None, domain='STRUCT'):
        if structured_one_hot is not None:
            structured_one_hot = self.struct_periph.encode(structured_one_hot)
        if structured is not None:
            structured = self.struct_entity_periph.encode(structured)
        return self.cnp.encode_structured(structured, structured_one_hot, domain=domain)
        #return self.cnp.encode_structured(None, struct_one_encoding, domain=domain)

    def cross_cache_attention(self):
        return self.cnp.cross_cache_attention()

    def decode_from_targets(self,task,targets,target_pad_mask=None):
        return self.cnp.decode(task, targets=targets,pad_mask=target_pad_mask)
    
    def decode_greedy(self,task, num_steps):
        return self.cnp.decode(task, targets=None, num_steps=num_steps)

    def save(self, checkpoint_dir, iterations):
        save_dir = os.path.join(checkpoint_dir, str(iterations))
        try:
            os.stat(save_dir)
        except:
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pth'))
        print('Model saved, iterations: {}'.format(iterations))

    def rename_mha_checkpoint(self, pretrained_dict):
        return {k.replace('mha', 'slf_attn'): v for k, v in pretrained_dict.items()}

    def restore(self, checkpoint_dir, iterations):
        save_dir = os.path.join(checkpoint_dir, str(iterations), 'model.pth')
        pretrained_dict=self.rename_mha_checkpoint(torch.load(save_dir))
        model_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict,strict=False)
        print('Restored existing model with iterations: {}'.format(iterations))
    
    def restore_file(self, file):
        pretrained_dict=self.rename_mha_checkpoint(torch.load(file))
        model_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict,strict=False)
    
    @staticmethod
    def __defaultconf__():
        """
        The default confurigation as specified in the original paper

        """

        cnp_conf = {
            'input_dim':512,
            'control_dim':32,
            'output_dim':512,
            'spatial_dim':512,
            'temporal_dim':512,
            'temporal_n_layers':6,
            'temporal_n_heads':8,
            'temporal_d_k':64,
            'temporal_d_v':64,
            'temporal_hidden_dim':2048,
            'decoder_dim':512,
            'decoder_n_layers':6,
            'decoder_n_heads':8,
            'decoder_d_k':64,
            'decoder_d_v':64,
            'decoder_hidden_dim':2048,
            'max_seq_len':500,
            'output_embedding_dim':300,
            'dropout':0.1}
        perph_conf = {
            'german_language_input_vocab': 25000,
            'german_language_input_embed': 300,
            'english_language_input_vocab': 25000,
            'english_language_input_embed': 300,
            'english_language_output_vocab': 25000,
            'german_language_output_vocab': 25000,
            'dropout': 0.1 ,
            'vqa_output_vocab':3500,
            'hmdb_output_classes':52,
            'penn_output_classes':48
        }

        domains = ['ENGLISH','GERMAN','IMAGE']

        return cnp_conf, perph_conf, domains
