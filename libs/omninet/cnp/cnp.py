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

OmniNet Central Neural Processor implementation

"""

from .Layers import *
from ..util import *
from torch.nn.functional import log_softmax, softmax

# import os
# import pickle
# import glob

# def write_attn(out_dir,attn):
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#    
#     ckpts = glob.glob(os.path.join(out_dir, '*'))
#     batches = [int(os.path.basename(c)) for c in ckpts]
#     if len(batches) == 0:
#         last_b = 0
#     else:
#         last_b = max(batches)
#     with open(os.path.join(out_dir,str(last_b+1)), 'wb') as f:
#         pickle.dump(attn, f)
         
class CNP(nn.Module):

    def __init__(self,tasks,conf=None,domains=['EMPTY'],gpu_id=-1):
        super(CNP, self).__init__()
        default_conf=self.__defaultconf__()
        if(conf!=None):
            for k in conf.keys():
                if k not in conf:
                    raise ValueError("The provided configuration does not contain %s"%k)
        else:
            conf=default_conf
        #Load the Confurigation
        self.gpu_id=gpu_id
        self.input_dim=conf['input_dim']
        self.control_dim=conf['control_dim']
        self.output_dim=conf['output_dim']
        self.structured_dim=conf['structured_dim']
        self.spatial_dim=conf['spatial_dim']
        self.temporal_dim=conf['temporal_dim']
        self.temporal_n_layers=conf['temporal_n_layers']
        self.temporal_n_heads=conf['temporal_n_heads']
        self.temporal_d_k=conf['temporal_d_k']
        self.temporal_d_v=conf['temporal_d_v']
        self.temporal_hidden_dim=conf['temporal_hidden_dim']
        self.decoder_dim=conf['decoder_dim']
        self.decoder_n_layers=conf['decoder_n_layers']
        self.decoder_n_heads=conf['decoder_n_heads']
        self.decoder_d_k=conf['decoder_d_k']
        self.decoder_d_v=conf['decoder_d_v']
        self.decoder_hidden_dim=conf['decoder_hidden_dim']
        self.max_seq_len=conf['max_seq_len']
        self.output_embedding_dim=conf['output_embedding_dim']
        self.dropout=conf['dropout']
        self.more_dropout=conf['more_dropout']
        self.batch_size=-1 #Uninitilized CNP memory

        self.use_temporal_encoder=conf['use_temporal_encoder']
        self.use_cca=conf['use_cca']
        self.use_patch=conf['use_patch'] if 'use_patch' in conf else False

        # Prepare the task lists and various output classifiers and embeddings
        if isinstance(tasks, dict):
            self.task_clflen = list(tasks.values())
            self.task_dict = {t: i for i, t in enumerate(tasks.keys())}
        else:
            raise ValueError('Tasks must be of type dict containing the tasks and output classifier dimension')

        self.output_clfs = nn.ModuleList([nn.Linear(self.output_dim, t) for t in self.task_clflen])
        #Use one extra to define padding
        self.output_embs = nn.ModuleList([nn.Embedding(t+1,self.output_embedding_dim,padding_idx=t) for t in self.task_clflen])

        #Initialize the various sublayers of the CNP
        control_states=domains+list(tasks.keys())
        self.control_peripheral=ControlPeripheral(self.control_dim,control_states,gpu_id=gpu_id)
        if self.use_temporal_encoder:
            self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len,self.temporal_n_layers,
                                                     self.temporal_n_heads,self.temporal_d_k,self.temporal_d_v,
                                                    self.temporal_dim,self.temporal_hidden_dim,vit_mlp=conf['use_vit_mlp'], dropout=self.dropout,
                                                     gpu_id=self.gpu_id)
        if self.use_patch:
            # TODO: add necessary configs
            self.patch_embedding = PatchEmbedding(self.input_dim, conf['patch_sizes'], conf['max_clip_len'], conf['max_patches_h'], conf['max_patches_w'], self.spatial_dim, stride=conf['patch_stride'], pos_emb=conf['patch_emb_pos'], dropout=self.dropout, gpu_id=self.gpu_id)
        if self.use_cca:
            self.cca = CrossCacheAttention(conf['cca_caches'], conf['cca_n_layers'], conf['sa_n_layers'], conf['psa_n_layers'],
                                                self.spatial_dim, self.structured_dim, self.temporal_dim, 
                                                conf['cca_hidden_dim'], conf['cca_n_heads'], conf['sa_n_heads'], conf['psa_n_heads'], conf['cca_d_k'], conf['cca_d_v'], 
                                                conf['default_attn_blocks'], conf['use_vit_mlp'], 
                                                cca_streams=conf['cca_streams'], pos_emb_streams=conf['pos_emb_streams'], dropout_p=conf['dropout_p'], dropout_s=conf['dropout_s'], dropout_t=conf['dropout_t'], drop_path_rate=conf['drop_path_rate'], sa_drop_path_rate=conf['sa_drop_path_rate'], return_attns=conf['save_cca_attn'], patch_pos=conf['patch_pos'], max_clip_len=conf['max_clip_len'], max_patches_h=conf['max_patches_h'], max_patches_w=conf['max_patches_w'], sa_on_whole_cache=conf['sa_on_whole_cache'], gpu_id=self.gpu_id)
        self.decoder=Decoder(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        #Initialize the various CNP caches as empty
        self.spatial_cache=None
        self.temporal_cache=None
        self.structured_cache=None
        self.decoder_cache=None
        self.temporal_spatial_link=[]
        self.pad_cache=None    #Used to store the padding values so that it can be used later in enc dec attn
        self.structured_one_encoding=None
        self.structured_logits=None

        #Various projection layers
        self.spatial_pool=nn.AdaptiveAvgPool1d(1)
        self.inpcont_input_proj=nn.Linear(self.input_dim+self.control_dim,self.input_dim)
        self.input_spatial_proj=nn.Linear(self.input_dim,self.spatial_dim)
        self.input_temporal_proj=nn.Linear(self.input_dim,self.temporal_dim)
        self.emb_decoder_proj=nn.Linear(self.output_embedding_dim,self.decoder_dim)
        self.cont_decoder_proj=nn.Linear(self.control_dim,self.decoder_dim)
        
        self.combined_logit_proj=nn.Linear(self.structured_dim+self.output_dim,self.output_dim)
        #freeze layers

        self.proj_dropout = nn.Dropout(self.dropout)        
        
    def decode(self,task,targets=None,num_steps=100,recurrent_steps=1,pad_mask=None,beam_width=1):
        if targets is not None:
            b,t=targets.shape
            #Use teacher forcing to generate predictions. the graph is kept in memory during this operation.
            if (len(targets.shape) != 2 or targets.shape[0] != self.batch_size):
                raise ValueError(
                    "Target tensor must be of shape (batch_size,length of sequence).")
            if task not in self.task_dict.keys():
                raise ValueError('Invalid task %s'%task)
            dec_inputs=self.output_embs[self.task_dict[task]](targets)
            dec_inputs=self.emb_decoder_proj(dec_inputs)
            if self.more_dropout:
                dec_inputs=self.proj_dropout(dec_inputs)
            control=self.control_peripheral(task,(self.batch_size))
            control=control.unsqueeze(1)
            control=self.cont_decoder_proj(control)
            if self.more_dropout:
                dontrol=self.proj_dropout(control)
            dec_inputs=torch.cat([control,dec_inputs],1)
            # Get output from decoder
            #Increase the length of the pad_mask to match the size after adding the control vector
            if pad_mask is not None:
                pad_extra=torch.zeros((b,1),device=self.gpu_id,dtype=pad_mask.dtype)
                pad_mask=torch.cat([pad_extra,pad_mask],1)
            logits,=self.decoder(dec_inputs,self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                 self.pad_cache,
                                 recurrent_steps=recurrent_steps,pad_mask=pad_mask)

            # if self.structured_cache is not None:
            #     structured_logits,_ = self.structured_decoder(self.structured_cache)
            #     structured_logits = self.structured_logits.expand(-1,t+1,-1)
            #     logits = self.combined_logit_proj(torch.cat([logits, structured_logits], 2))

            l1_loss = None                
            if self.structured_logits is not None:
                # print('decode structured_logits not None')
                structured_logits = self.structured_logits.expand(-1,t+1,-1)
                structured_logits = torch.cat([logits, structured_logits], 2)
                logits = self.combined_logit_proj(structured_logits)
                if self.more_dropout:
                    logits = self.proj_dropout(logits)
                l1_loss = torch.sum(torch.abs(self.combined_logit_proj.weight[:,self.structured_dim:])) 

            #Predict using the task specific classfier
            predictions=self.output_clfs[self.task_dict[task]](logits)
            predictions=predictions[:,0:t,:]
            return log_softmax(predictions,dim=2), l1_loss
        else:
            control = self.control_peripheral(task, (self.batch_size))
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)
            if self.more_dropout:
                control = self.proj_dropout(control)
            dec_inputs=control
            
            for i in range(num_steps-1):
                logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                       self.pad_cache,
                                       recurrent_steps=recurrent_steps)
                prediction = self.output_clfs[self.task_dict[task]](logits)
                prediction=prediction[:,-1,:].unsqueeze(1)
                prediction=log_softmax(prediction,dim=2).argmax(-1)
                prediction=self.output_embs[self.task_dict[task]](prediction)
                prediction = self.emb_decoder_proj(prediction).detach()
                if self.more_dropout:
                    prediction = self.proj_dropout(prediction)
                if beam_width>1:
                    p=torch.topk(softmax(prediction),beam_width)
                    
                dec_inputs=torch.cat([dec_inputs,prediction],1)
            logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,                                     self.pad_cache,recurrent_steps=recurrent_steps)

            # if self.structured_cache is not None:
            #     structured_logits,_ = self.structured_decoder(self.structured_cache)
            #     structured_logits = self.structured_logits.expand(-1,num_steps,-1)
            #     logits = self.combined_logit_proj(torch.cat([logits, structured_logits], 2))
               
            l1_loss = None
            if self.structured_logits is not None:
                # print('decode structured_logits not None')
                structured_logits = self.structured_logits.expand(-1,t+1,-1)
                structured_logits = torch.cat([logits, structured_logits], 2)
                logits = self.combined_logit_proj(structured_logits)
                if self.more_dropout:
                    logits = self.proj_dropout(logits)
                
                l1_loss = torch.sum(torch.abs(self.combined_logit_proj.weight[:,self.structured_dim:])) 
            
            predictions = self.output_clfs[self.task_dict[task]](logits)
            return log_softmax(predictions,dim=2), l1_loss

        

    def encode(self,input,pad_mask=None,domain='EMPTY',recurrent_steps=1,sa=False):
        if (len(input.shape)!=4):
            raise Exception('Invalid input dimensions.')
        b,t,s,f=list(input.size())
        self.temporal_spatial_link.append((t,s))
        if b != self.batch_size:
            raise Exception('Input batch size does not match.')
        #Spatial encode. Spatial encodes encodes both spatial and time dimension features together
        control_vecs = self.control_peripheral(domain, (b, t, s))
        input = torch.cat([input, control_vecs], 3)
        input=self.inpcont_input_proj(input)
        if self.more_dropout:
            input = self.proj_dropout(input)
        #Project the spatial data, into the query dimension and add it to the existing cache
        if s>1:
            spatial_f=torch.reshape(input,[b,t*s,f])
            spatial_f=self.input_spatial_proj(spatial_f)
            if self.more_dropout:
                spatial_f=self.proj_dropout(spatial_f)
            if self.spatial_cache is None:
                self.spatial_cache=spatial_f
            else:
                self.spatial_cache=torch.cat([self.spatial_cache,spatial_f],1)

        #Feed the time features. First AVGPool the spatial features.
        temp_data=input.transpose(2,3).reshape(b*t,f,s)
        temp_data=self.spatial_pool(temp_data).reshape(b,t,f)
        temp_data=self.input_temporal_proj(temp_data)
        if self.more_dropout:
            temp_data=self.proj_dropout(temp_data)
        #Create a control state and concat with the temporal data
        #Add data to temporal cache
        if self.use_temporal_encoder:
            temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)
        elif sa:
            temp_data = self.cca.cca_stream(self.cca.sa_t, temp_data, None, 't', pad_mask_q=pad_mask, pad_mask_k=pad_mask)[0]
       
        if self.temporal_cache is None:
            self.temporal_cache=temp_data
        else:
            self.temporal_cache=torch.cat([self.temporal_cache,temp_data],1)

        #Add pad data to pad cache
        if pad_mask is None:
            pad_mask=torch.zeros((b,t),device=self.gpu_id,dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache=pad_mask
        else:
            self.pad_cache=torch.cat([self.pad_cache,pad_mask],1)
            

    def encode_structured(self,cat_encodings,one_encoding,domain='EMPTY'):
        if one_encoding is not None:
            self.structured_one_encoding = one_encoding.unsqueeze(1)
            self.structured_logits = self.structured_one_encoding
            # print('set self.structured_logits to one_hot_encodings')
        if cat_encodings is not None:
            if one_encoding is not None:
                struct_encodings = torch.cat([cat_encodings, self.structured_one_encoding], 1)
            else:
                struct_encodings = cat_encodings
            b,n,f = list(struct_encodings.size())
            control_vecs = self.control_peripheral(domain, (b, n))
            struct_encodings = torch.cat([struct_encodings, control_vecs], 2)
            structured_f=self.inpcont_input_proj(struct_encodings) # learns cross modality information?
            if self.more_dropout:
                structured_f = self.proj_dropout(structured_f)
            if self.structured_cache is None:
                self.structured_cache=structured_f
            else:
                self.structured_cache=torch.cat([self.structured_cache,structured_f],1)

    def encode_with_patch(self, input, img_size):
        if self.use_patch:
            return self.patch_embedding(input, img_size)
        return input

    def cross_cache_attention(self):
        if self.use_cca:
            cca_out = self.cca(self.spatial_cache, 
                               self.temporal_cache, 
                               self.structured_cache,
                               self.pad_cache,
                               self.temporal_spatial_link)
            self.spatial_cache = cca_out[0]
            self.temporal_cache = cca_out[1]
            self.structured_cache = cca_out[2]
            if cca_out[3] is not None:
                # print('set self.structured_logits to cca out')
                self.structured_logits = cca_out[3]
            # if not self.use_temporal_encoder:
            #     n_spatial = sum([t for t,s in self.temporal_spatial_link if s > 1])
            #     temporal_cache_t,=self.temporal_encoder(self.temporal_cache[:,n_spatial:],pad_mask=self.pad_cache[:,n_spatial:],recurrent_steps=1)
            #     self.temporal_cache = torch.cat([self.temporal_cache[:,:n_spatial], temporal_cache_t], 1)
            if len(cca_out) == 5:
                return cca_out[4]

    def clear_spatial_cache(self):
        self.spatial_cache=None

    def clear_temporal_cache(self):
        self.temporal_raw_cache=None
        self.temporal_cache=None

    def clear_structured_cache(self):
        self.structured_cache=None

    def reset(self,batch_size=1):
        self.attn_scores=[]
        self.batch_size=batch_size
        self.temporal_spatial_link=[]
        self.pad_cache=None
        self.pad_cache=None
        self.clear_spatial_cache()
        self.clear_temporal_cache()
        self.clear_structured_cache()
    
    @staticmethod
    def __defaultconf__():
        conf={
            'input_dim':128,
            'control_dim':32,
            'output_dim':128,
            'spatial_dim':128,
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
            'max_seq_len':1000,
            'output_embedding_dim':300,
            'dropout':0.1
        }
        return conf

class ControlPeripheral(nn.Module):
    """
        A special peripheral used to help the CNP identify the data domain or specify the context of
        the current operation.

    """
    def __init__(self, control_dim, control_states, gpu_id=-1):
        """
            Accepts as input control states as list of string. The control states are sorted before id's
            are assigned
        """
        super(ControlPeripheral, self).__init__()
        self.control_dim = control_dim
        self.gpu_id = gpu_id
        self.control_dict = {}
        for i, c in enumerate(control_states):
            self.control_dict[c] = i
        self.control_embeddings=nn.Embedding(len(control_states)+1,self.control_dim)

    def forward(self, control_state, shape=()):
        if self.gpu_id>=0:
            control_ids = torch.ones(shape, dtype=torch.long,device=self.gpu_id)*self.control_dict[control_state]
        else:
            control_ids = torch.ones(shape, dtype=torch.long)*self.control_dict[control_state]
        return self.control_embeddings(control_ids)

class TemporalCacheEncoder(nn.Module):
    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, vit_mlp=False, dropout=0.1,gpu_id=-1):

        super().__init__()

        n_position = len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, vit_mlp=vit_mlp, dropout=dropout)
            for _ in range(n_layers)])
        self.gpu_id=gpu_id

    def forward(self, src_seq, return_attns=False,recurrent_steps=1, pad_mask=None):

        enc_slf_attn_list = []
        b,t,_=src_seq.shape

        if self.gpu_id >= 0:
            src_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            src_pos = torch.arange(1, t + 1).repeat(b, 1)
        # -- Forward
        enc_output = src_seq + self.position_enc(src_pos)
        enc_output = self.dropout_emb(enc_output)
        if pad_mask is not None:
            slf_attn_mask=get_attn_key_pad_mask(pad_mask,src_seq)
        else:
            slf_attn_mask=None
        non_pad_mask=get_non_pad_mask(src_seq,pad_mask)
        for i in range(recurrent_steps):
            for enc_layer in self.layer_stack:
                enc_output, enc_slf_attn = enc_layer(
                    enc_output,non_pad_mask,slf_attn_mask=slf_attn_mask)

                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class PatchEmbedding(nn.Module):
    ''' 
    Decompose spatial cache elements into patches 
    and add position embeddings
    '''

    def __init__(self, patch_dim, patch_sizes, len_max_frames, len_max_patches_h, len_max_patches_w, d_in, stride=None, pos_emb=True, dropout=0.1, gpu_id=-1):
        super(PatchEmbedding, self).__init__()

        # self.num_patches = (image_size[0] // patch_sizes[0]) * (image_size[1] // patch_sizes[1])
        # patch_dim = d_in * patch_sizes[0] * patch_sizes[1]
        # self.to_patch_embedding_linear = nn.Linear(patch_dim, d_in)
        self.patching = False
        if not (patch_sizes[0] == 1 and patch_sizes[1] == 1):
            if stride is None:
                self.to_patch_embedding = nn.Conv2d(d_in, patch_dim, kernel_size=patch_sizes, stride=patch_sizes)
            else:
                self.to_patch_embedding = nn.Conv2d(d_in, patch_dim, kernel_size=patch_sizes, stride=stride)
            self.patching = True
        
        self.pos_emb = pos_emb
        if pos_emb:
            print('PatchEmbedding initialize pos emb')
            self.patch_pos_h_emb = nn.Embedding(len_max_patches_h, d_in//4)
            self.patch_pos_w_emb = nn.Embedding(len_max_patches_w, d_in//4)
            self.frame_pos_emb = nn.Embedding(len_max_frames, d_in//2)

        self.pos_drop = nn.Dropout(dropout)
        self.gpu_id = gpu_id

    def forward(self, img_seq, img_size=None):
        # print('PatchEmbedding')
        b, t, s, f = img_seq.shape
        # print(img_seq.shape)
        if img_size is None:
            h = w = int(s**0.5)
        else:
            h, w = img_size
        # print(h,w)
        # print(img_size)

        # ph, pw = self.patch_sizes
        # s = ts // t
        # h = w = int(s ** 0.5)
        
        # # img_seq = rearrange(img_seq, 'b t (h w) f -> b t h w f', h=h, w=w)
        # # img_seq = rearrange(img_seq, 'b t (h ph) (w pw) f -> b (t h w) (ph pw f)', ph = ph, pw = pw)
        # if ph*pw > 1:
        #     img_seq = self.to_patch_embedding_linear(img_seq)
        # img_seq = rearrange(img_seq, 'b t (h w) f -> (b t) f h w', h=h, w=w)
        img_seq = img_seq.reshape(-1,h,w,f).permute(0,3,1,2) # 'b t (h w) f -> (b t) f h w', h=h, w=w)
        if self.patching:
            patch_seq = self.to_patch_embedding(img_seq)
        else:
            patch_seq = img_seq
        _, _, n_patches_h, n_patches_w = patch_seq.shape
        # print(patch_seq.shape)

        # # patch_seq = rearrange(patch_seq, '(b t) f h w -> b (t h w) f', t=t)
        # patch_seq = rearrange(patch_seq, '(b t) f h w -> b t h w f', t=t)
        patch_seq = patch_seq.reshape(b,t,f,n_patches_h,n_patches_w).permute(0,1,3,4,2)
        # print('patch_seq', patch_seq.shape)

        if not hasattr(self, 'pos_emb') or self.pos_emb:
            # n_patches = patch_seq.shape[1]//t
            # ppos = self.patch_pos_emb(torch.arange(n_patches, device=self.gpu_id))
            ppos_h = self.patch_pos_h_emb(torch.arange(n_patches_h, device=self.gpu_id))
            ppos_w = self.patch_pos_w_emb(torch.arange(n_patches_w, device=self.gpu_id))
            # print('ppos_h', ppos_h.shape)
            # print('ppos_w', ppos_w.shape)
            ppos_h = ppos_h.unsqueeze(1).expand(n_patches_h, n_patches_w, ppos_h.shape[-1])
            ppos_w = ppos_w.expand(n_patches_h, n_patches_w, ppos_w.shape[-1])
            ppos = torch.cat([ppos_h, ppos_w], 2)
            # print('ppos', ppos.shape)
            ppos = ppos.repeat(t,1,1,1)
            # print('ppos', ppos.shape)

            fpos = self.frame_pos_emb(torch.arange(t, device=self.gpu_id))
            # print('fpos', fpos.shape)
            pos_dim = fpos.shape[1]
            n_patches = n_patches_h * n_patches_w
            fpos = fpos.unsqueeze(2).expand(t,pos_dim,n_patches).transpose(1,2).reshape(t,n_patches_h,n_patches_w,pos_dim)
            # print('fpos', fpos.shape)

            xpos = torch.cat([fpos,ppos],3)
            # print('xpos', xpos.shape)
            # print('patch_seq', patch_seq.shape)
            patch_seq += xpos

        # # patch_seq = rearrange(patch_seq, 'b (t p) f -> b t p f', p=n_patches)
        # patch_seq = rearrange(patch_seq, 'b t h w f -> b t (h w) f')
        patch_seq = patch_seq.reshape(b,t,-1,f)
        
        patch_seq = self.pos_drop(patch_seq)
        
        return patch_seq

class CrossCacheAttention(nn.Module):
    def __init__(self, cache_names, n_layers, sa_n_layers, psa_n_layers, d_p, d_s, d_t, d_inner, n_head, sa_n_head, psa_n_head, d_k, d_v, default_attn_blocks=False, vit_mlp=False, cca_streams=None, pos_emb_streams=None, dropout_p=0.1, dropout_s=0.1, dropout_t=0.1, dropout=0.1, drop_path_rate=0., sa_drop_path_rate=0., return_attns=False, patch_pos=False, max_clip_len=16, max_patches_h=7, max_patches_w=7, sa_on_whole_cache=True, gpu_id=-1):
        super(CrossCacheAttention, self).__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.sa_n_layers = sa_n_layers
        self.sa_n_head = sa_n_head
        self.psa_n_layers = psa_n_layers
        self.psa_n_head = psa_n_head
        self.d_p = d_p
        self.d_s = d_s
        self.d_t = d_t
        self.d_inner = d_inner
        self.d_k = d_k
        self.d_v = d_v
        self.default_attn_blocks = default_attn_blocks
        self.vit_mlp = vit_mlp
        self.dropout_p = dropout_p
        self.dropout_s = dropout_s
        self.dropout_t = dropout_t
        self.drop_path_rate = drop_path_rate
        self.sa_drop_path_rate = sa_drop_path_rate
        # self.more_dropout = more_dropout
        self.return_attns = return_attns
        self.cache_names = cache_names
        self.patch_pos = patch_pos
        self.sa_on_whole_cache = sa_on_whole_cache
        self.gpu_id = gpu_id

        self.cache_symbols = []

        if 'spatial' in cache_names:
            self.cache_symbols.append('p')
        if 'temporal' in cache_names:
            self.cache_symbols.append('t')
        if 'structured' in cache_names:
            self.cache_symbols.append('s')
        
        if cca_streams is None:
            self.streams = [''.join(list(perm)) for perm in list(permutations(self.cache_symbols, 2))]
            # # remove streams toward structured
            # self.streams = [stream_name for stream_name in self.streams if stream_name not in ['sp','st']]
            self.streams.extend(self.cache_symbols)
        else:
            self.streams = cca_streams

        self.cca_p_with_t = self.get_network('pt')
        self.cca_p_with_s = self.get_network('ps')
        self.cca_t_with_p = self.get_network('tp')
        self.cca_t_with_s = self.get_network('ts')
        self.cca_s_with_p = self.get_network('sp')
        self.cca_s_with_t = self.get_network('st')
        self.sa_s = self.get_network('s')
        self.sa_p = self.get_network('p')
        self.sa_t = self.get_network('t')

        self.p_proj = nn.Linear(2*d_p, d_p) if 'pt' in self.streams and 'ps' in self.streams else None
        self.t_proj = nn.Linear(2*d_t, d_t) if 'tp' in self.streams and 'ts' in self.streams else None
        self.s_proj = nn.Linear(2*d_s, d_s) if 'sp' in self.streams and 'st' in self.streams else None
        # if self.p_proj is not None or self.t_proj is not None or self.s_proj is not None:
        self.proj_dropout = nn.Dropout(dropout)

        self.pos_emb_streams = pos_emb_streams

        n_position = 501 #len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc_t = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, self.d_t, padding_idx=0),
            freeze=True)
        if not patch_pos:
            self.position_enc_p = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table((max_patches_h*max_patches_w+1), d_p, padding_idx=0),
                freeze=True)
        else:
            self.position_enc_p = PatchEmbedding(d_p, (1,1), max_clip_len, max_patches_h, max_patches_w, d_p, pos_emb=conf['patch_emb_pos'], dropout=dropout, gpu_id=gpu_id)

    def get_network(self, cca_type='t'):
        if cca_type not in self.streams:
            print('cca_type', cca_type, 'not in streams:', self.streams)
            return None
        
        d_inner = self.d_inner
        dpr = self.drop_path_rate
        if cca_type == 'tp':
            d_model, attn_dropout = (self.d_t, self.d_p), self.dropout_t
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 'ts':
            d_model, attn_dropout = (self.d_t, self.d_s), self.dropout_t
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 'sp':
            d_model, attn_dropout = (self.d_s, self.d_p), self.dropout_s
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 'st':
            d_model, attn_dropout = (self.d_s, self.d_t), self.dropout_s
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 'pt':
            d_model, attn_dropout = (self.d_p, self.d_t), self.dropout_p
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 'ps':
            d_model, attn_dropout = (self.d_p, self.d_s), self.dropout_p
            n_layers, n_head = self.n_layers, self.n_head
        elif cca_type == 's':
            d_model, attn_dropout = self.d_s, self.dropout_s
            n_layers, n_head = self.sa_n_layers, self.sa_n_head
            dpr = self.sa_drop_path_rate
        elif cca_type == 'p':
            d_model, attn_dropout = self.d_p, self.dropout_p
            n_layers, n_head = self.psa_n_layers, self.psa_n_head
            d_inner = 2048
            dpr = self.sa_drop_path_rate
        elif cca_type == 't':
            d_model, attn_dropout = self.d_t, self.dropout_t
            n_layers, n_head = self.sa_n_layers, self.sa_n_head
            d_inner = 2048
            dpr = self.sa_drop_path_rate
        else:
            raise Exception('Unknow cca network type.')

        return TransformerEncoderBlock(n_layers=n_layers, 
                                       d_model=d_model, 
                                       d_inner=d_inner, 
                                       n_head=n_head,
                                       d_k=self.d_k, 
                                       d_v=self.d_v, 
                                       default_attn_blocks=self.default_attn_blocks,
                                       vit_mlp=self.vit_mlp,
                                       dropout=attn_dropout,
                                       drop_path_rate=dpr)

    def cca_pos_emb(self, cache, cache_symbol, stream_name, position_enc, dropout_emb, temporal_spatial_link=[(1,49),(-1,1)]):
        k = '%s-%s'%(stream_name,cache_symbol)
        if k not in self.pos_emb_streams:
            # print('cca_pos_emb stream name not found', cache_symbol, stream_name, k)
            return cache
        
        # print('cca_pos_emb', k)
        if cache_symbol == 't':
            position_enc = self.position_enc_t
        elif cache_symbol == 'p':
            position_enc = self.position_enc_p
        else:
            raise Exception('unknown cache_symbol for position encoding.')

        # print('cca_pos_emb add pos emb for', k) 
        if cache_symbol == 'p' and hasattr(self, 'patch_pos') and self.patch_pos: 
            # print('cca_pos_emb patch_pos')
            cache_lst = []
            cursor = 0
            b,_,f = cache.shape
            for t,s in temporal_spatial_link:
                if s > 1:
                    cache_lst.append(position_enc(cache[:,cursor:s].reshape(b,t,s,f)).reshape(b,t*s,f))
                    cursor += s
            cache = torch.cat(cache_lst, dim=1)
        else:
            # print('cca_pos_emb sine')
            b,t,_=cache.shape
            if self.gpu_id >= 0:
                src_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
            else:
                src_pos = torch.arange(1, t + 1).repeat(b, 1)
            cache = cache + position_enc(src_pos)
        
        # cache = dropout_emb(cache)
        return cache

    def cca_stream(self, net, cache_alpha, cache_beta, stream_name, pad_mask_q=None, pad_mask_k=None):
        # print('cca_stream', stream_name)
        if net is None or cache_alpha is None:
            return None,None

        if pad_mask_k is not None:
            attn_mask = get_attn_key_pad_mask(pad_mask_k, cache_alpha)
        else:
            attn_mask = None
        non_pad_mask=get_non_pad_mask(cache_alpha,pad_mask_q)

        if stream_name[0] == 't':
            position_enc = self.position_enc_t
        elif stream_name[0] == 'p':
            position_enc = self.position_enc_p
        else:
            raise Exception('unknown cache_symbol for position encoding.')

        cache_alpha = self.cca_pos_emb(cache_alpha, stream_name[0], stream_name, position_enc, self.dropout_emb)

        if cache_beta is None:
            return net(cache_alpha, non_pad_mask, attn_mask=attn_mask)

        cache_beta = self.cca_pos_emb(cache_beta, stream_name[1], stream_name, position_enc, self.dropout_emb)
        return net(cache_alpha, non_pad_mask, enc_input_k=cache_beta, enc_input_v=cache_beta, attn_mask=attn_mask)

    def combine_stream(self, stream_proj, stream_out1, stream_out2, target_cache):
        if stream_out1 is not None and stream_out2 is not None:
            # print('combine_stream combines')
            stream_out = torch.cat([stream_out1, stream_out2], 2)
            stream_out = self.proj_dropout(stream_proj(stream_out))
            return stream_out

        if stream_out1 is not None:
            return stream_out1

        if stream_out2 is not None:  
            return stream_out2
        
        return target_cache

    def forward(self, spatial_cache, temporal_cache, structured_cache, pad_cache, temporal_spatial_link):
        n_spatial = sum([t for t,s in temporal_spatial_link if s > 1])
        temporal_lst = [t for t,s in temporal_spatial_link if s == 1]
        real_pad_cache = pad_cache[:,n_spatial:]
        real_temporal_cache = temporal_cache[:,n_spatial:]
        # print('cca_p_with_t')
        spatial_temporal_cache, spatial_temporal_attn = self.cca_stream(self.cca_p_with_t, 
                                                    spatial_cache, 
                                                    real_temporal_cache,
                                                    'pt',
                                                    pad_mask_k=real_pad_cache)
           
        # try:
        #     write_attn('/scratch1/yxuea/out/cca_spatial_temporal_attn_ignore_t0', spatial_temporal_attn.detach().cpu().numpy())
        # except:
        #     pass
        # print('cca_p_with_s')
        spatial_structured_cache, spatial_structured_attn = self.cca_stream(self.cca_p_with_s, 
                                                      spatial_cache, 
                                                      structured_cache,
                                                      'ps')
        # print('combine pt ps')
        spatial_cross_cache = self.combine_stream(self.p_proj, 
                                                  spatial_temporal_cache, 
                                                  spatial_structured_cache, 
                                                  spatial_cache)
        
        if self.sa_p is not None and spatial_cross_cache is not None:
            # print('sa_p')
            spatial_cross_cache = self.cca_stream(self.sa_p, spatial_cross_cache, None, 'p')[0]

        # print('cca_t_with_p')
        temporal_spatial_cache, temporal_spatial_attn = self.cca_stream(self.cca_t_with_p, 
                                                    real_temporal_cache, 
                                                    spatial_cache,
                                                    'tp',
                                                    pad_mask_q=real_pad_cache)
        # try:
        #     write_attn('/scratch1/yxuea/out/cca_temporal_spatial_attn_ignore_t0', temporal_spatial_attn.detach().cpu().numpy())
        # except:
        #     pass
        # print('cca_t_with_s')
        temporal_structured_cache, temporal_structured_attn = self.cca_stream(self.cca_t_with_s, 
                                                       real_temporal_cache, 
                                                       structured_cache,
                                                       'ts',
                                                       pad_mask_q=real_pad_cache)
        # print('combine tp ts')
        temporal_cross_cache = self.combine_stream(self.t_proj, 
                                                   temporal_spatial_cache, 
                                                   temporal_structured_cache, 
                                                   real_temporal_cache)
 
        if self.sa_t is not None and temporal_cross_cache is not None:
            if len(temporal_lst) == 1 or self.sa_on_whole_cache:
                temporal_cross_cache = self.cca_stream(self.sa_t, temporal_cross_cache, None, 't', pad_mask_q=real_pad_cache, pad_mask_k=real_pad_cache)[0]
            else:
                temporal_cache_lst = []
                cursor = 0
                for len_t in temporal_lst:
                    temporal_cache_lst.append(self.cca_stream(self.sa_t, temporal_cross_cache[cursor:(cursor+len_t)], None, 't', pad_mask_q=real_pad_cache[cursor:(cursor+len_t)], pad_mask_k=real_pad_cache[cursor:(cursor+len_t)])[0])
                    cursor += len_t
                temporal_cross_cache = torch.cat(temporal_cache_lst, 1)
            

        temporal_cross_cache = torch.cat([temporal_cache[:,:n_spatial], temporal_cross_cache], 1)

        # print('cca_s_with_t')
        structured_temporal_cache, _ = self.cca_stream(self.cca_s_with_t, 
                                                        structured_cache, 
                                                        real_temporal_cache,
                                                        'st',
                                                        pad_mask_k=real_pad_cache)
        # print('cca_s_with_p')
        structured_spatial_cache, _ = self.cca_stream(self.cca_s_with_p, 
                                                      structured_cache, 
                                                      spatial_cache, 'sp')
        # print('combine sp st')
        structured_cross_cache = self.combine_stream(self.s_proj, 
                                                    structured_temporal_cache, 
                                                    structured_spatial_cache, 
                                                    structured_cache)
        # print('structured_cross_cache',structured_cross_cache)
        if self.sa_s is not None and structured_cross_cache is not None:
            # print('sa_s')
            structured_logits = self.cca_stream(self.sa_s, structured_cross_cache, None, 's')[0]
            if structured_logits is not None:
                structured_logits = structured_logits[:,-1:,:]
        else:
            structured_logits = None
            # print('skip sa_s')

        if self.return_attns:
            return spatial_cross_cache, temporal_cross_cache, structured_cross_cache, structured_logits, {'pt': spatial_temporal_attn, 'tp': temporal_spatial_attn, 't_cache': temporal_cache, 'sp_cache': spatial_cache, 'ps': spatial_structured_attn, 'ts': temporal_structured_attn}
        return spatial_cross_cache, temporal_cross_cache, structured_cross_cache, structured_logits

class Decoder(nn.Module):

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, temporal_dim, spatial_dim,output_dim,dropout=0.1,gpu_id=-1):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, temporal_dim,
                         spatial_dim,dropout=dropout,gpu_id=gpu_id)
            for _ in range(n_layers)])
        self.output_fc=nn.Linear(d_model,output_dim)
        self.gpu_id=gpu_id

    def forward(self, dec_inputs, spatial_cache, temporal_cache,temporal_spatial_link,
                pad_cache,
                pad_mask=None,return_attns=False,recurrent_steps=1):

        # -- Forward
        b,t,_=dec_inputs.shape
        if self.gpu_id >= 0:
            dec_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            dec_pos = torch.arange(1, t + 1).repeat(b, 1)
        dec_outputs = dec_inputs + self.position_enc(dec_pos)
        slf_attn_mask_subseq=get_subsequent_mask((b,t),self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad=get_attn_key_pad_mask(pad_mask,dec_inputs)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask=slf_attn_mask_subseq
        #Run all the layers of the decoder building the prediction graph
        dec_enc_attn_mask=get_attn_key_pad_mask(pad_cache, dec_inputs)
        non_pad_mask=get_non_pad_mask(dec_inputs,pad_mask)
        for i in range(recurrent_steps):
            for dec_layer in self.layer_stack:
                dec_outputs, attns = dec_layer(dec_outputs,temporal_cache, spatial_cache,temporal_spatial_link,
                                               non_pad_mask,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_mask)
        dec_outputs=self.output_fc(dec_outputs)
        if return_attns:
            return dec_outputs,attns
        return dec_outputs,


