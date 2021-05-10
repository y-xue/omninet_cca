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
from einops import rearrange, repeat

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
        self.batch_size=-1 #Uninitilized CNP memory

        self.use_cca=conf['use_cca']

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
        self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len,self.temporal_n_layers,
                                                     self.temporal_n_heads,self.temporal_d_k,self.temporal_d_v,
                                                    self.temporal_dim,self.temporal_hidden_dim,dropout=self.dropout,
                                                     gpu_id=self.gpu_id)
        if self.use_cca:
            # TODO: add necessary configs
            self.patch_embedding = PatchEmbedding(self.input_dim, conf['patch_sizes'], conf['max_clip_len'], conf['max_patches_h'], conf['max_patches_w'], self.spatial_dim, gpu_id=self.gpu_id)
            self.cca = CrossCacheAttention(conf['cca_caches'], conf['cca_n_layers'], 
                                                self.spatial_dim, self.structured_dim, self.temporal_dim, 
                                                conf['cca_hidden_dim'], conf['cca_n_head'], conf['cca_d_k'], conf['cca_d_v'], 
                                                dropout_p=conf['dropout_p'], dropout_s=conf['dropout_s'], dropout_t=conf['dropout_t'])
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
            control=self.control_peripheral(task,(self.batch_size))
            control=control.unsqueeze(1)
            control=self.cont_decoder_proj(control)
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
                
            if self.structured_logits is not None:
                structured_logits = self.structured_logits.expand(-1,t+1,-1)
                logits = self.combined_logit_proj(torch.cat([logits, structured_logits], 2))
            
            #Predict using the task specific classfier
            predictions=self.output_clfs[self.task_dict[task]](logits)
            predictions=predictions[:,0:t,:]
            return log_softmax(predictions,dim=2)
        else:
            control = self.control_peripheral(task, (self.batch_size))
            control = control.unsqueeze(1)
            control = self.cont_decoder_proj(control)
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
                if beam_width>1:
                    p=torch.topk(softmax(prediction),beam_width)
                    
                dec_inputs=torch.cat([dec_inputs,prediction],1)
            logits, = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,                                     self.pad_cache,recurrent_steps=recurrent_steps)

            # if self.structured_cache is not None:
            #     structured_logits,_ = self.structured_decoder(self.structured_cache)
            #     structured_logits = self.structured_logits.expand(-1,num_steps,-1)
            #     logits = self.combined_logit_proj(torch.cat([logits, structured_logits], 2))
               
            if self.structured_logits is not None:
                structured_logits = self.structured_logits.expand(-1,t+1,-1)
                logits = self.combined_logit_proj(torch.cat([logits, structured_logits], 2))
            
            predictions = self.output_clfs[self.task_dict[task]](logits)
            return log_softmax(predictions,dim=2)

        

    def encode(self,input,pad_mask=None,domain='EMPTY',recurrent_steps=1):
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
        #Project the spatial data, into the query dimension and add it to the existing cache
        if s>1:
            spatial_f=torch.reshape(input,[b,t*s,f])
            spatial_f=self.input_spatial_proj(spatial_f)
            if self.spatial_cache is None:
                self.spatial_cache=spatial_f
            else:
                self.spatial_cache=torch.cat([self.spatial_cache,spatial_f],1)

        #Feed the time features. First AVGPool the spatial features.
        temp_data=input.transpose(2,3).reshape(b*t,f,s)
        temp_data=self.spatial_pool(temp_data).reshape(b,t,f)
        temp_data=self.input_temporal_proj(temp_data)
        #Create a control state and concat with the temporal data
        #Add data to temporal cache
        temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)
       
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
        self.structured_one_encoding = one_encoding.unsqueeze(1)
        struct_encodings = torch.cat([cat_encodings, self.structured_one_encoding], dim=1)
        control_vecs = self.control_peripheral(domain, (b, n))
        struct_encodings = torch.cat([struct_encodings, control_vecs], 3)
        structured_f=self.inpcont_input_proj(struct_encodings) # learns cross modality information?
        if self.structured_cache is None:
            self.structured_cache=structured_f
        else:
            self.structured_cache=torch.cat([self.structured_cache,structured_f],1)

    def encode_with_patch(self, input, img_size):
        if self.use_cca:
            return self.patch_embedding(input, img_size)
        return input

    def cross_cache_attention(self):
        if self.use_cca:
            cca_out = self.cca(self.spatial_cache, 
                               self.temporal_cache, 
                               self.structured_cache,
                               self.pad_cache)
            self.spatial_cache = cca_out[0]
            self.temporal_cache = cca_out[1]
            self.structured_cache = cca_out[2]
            self.structured_logits = cca_out[3]

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
            d_model, d_inner, dropout=0.1,gpu_id=-1):

        super().__init__()

        n_position = len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
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

    def __init__(self, patch_dim, patch_sizes, len_max_frames, len_max_patches_h, len_max_patches_w, d_in, gpu_id=-1):
        super(PatchEmbedding, self).__init__()

        # self.num_patches = (image_size[0] // patch_sizes[0]) * (image_size[1] // patch_sizes[1])
        # patch_dim = d_in * patch_sizes[0] * patch_sizes[1]
        # self.to_patch_embedding_linear = nn.Linear(patch_dim, d_in)

        self.to_patch_embedding = nn.Conv2d(d_in, patch_dim, kernel_size=patch_sizes, stride=patch_sizes)
        
        self.patch_pos_h_emb = nn.Embedding(len_max_patches_h, d_in//4)
        self.patch_pos_w_emb = nn.Embedding(len_max_patches_w, d_in//4)
        self.frame_pos_emb = nn.Embedding(len_max_frames, d_in//2)

        self.gpu_id = gpu_id

    def forward(self, img_seq, img_size):
        # print('PatchEmbedding')
        b, t, s, f = img_seq.shape
        # print(img_seq.shape)
        h, w = img_size

        # ph, pw = self.patch_sizes
        # s = ts // t
        # h = w = int(s ** 0.5)
        
        # img_seq = rearrange(img_seq, 'b t (h w) f -> b t h w f', h=h, w=w)
        # img_seq = rearrange(img_seq, 'b t (h ph) (w pw) f -> b (t h w) (ph pw f)', ph = ph, pw = pw)
        # if ph*pw > 1:
        #     img_seq = self.to_patch_embedding_linear(img_seq)
        img_seq = rearrange(img_seq, 'b t (h w) f -> (b t) f h w', h=h, w=w)
        patch_seq = self.to_patch_embedding(img_seq)
        _, _, n_patches_h, n_patches_w = patch_seq.shape

        # patch_seq = rearrange(patch_seq, '(b t) f h w -> b (t h w) f', t=t)
        patch_seq = rearrange(patch_seq, '(b t) f h w -> b t h w f', t=t)
        # print('patch_seq', patch_seq.shape)

        # n_patches = patch_seq.shape[1]//t
        # ppos = self.patch_pos_emb(torch.arange(n_patches, device=self.gpu_id))
        ppos_h = self.patch_pos_h_emb(torch.arange(n_patches_h, device=self.gpu_id))
        ppos_w = self.patch_pos_w_emb(torch.arange(n_patches_w, device=self.gpu_id))
        # print('ppos_h', ppos_h.shape)
        # print('ppos_w', ppos_w.shape)
        ppos_h = ppos_h.unsqueeze(1).expand(n_patches_h, n_patches_w, ppos_h.shape[-1])
        ppos_w = ppos_w.expand(n_patches_h, n_patches_w, ppos_w.shape[-1])
        ppos = torch.cat([ppos_h, ppos_w], dim=2)
        # print('ppos', ppos.shape)
        ppos = ppos.repeat(t,1,1,1)
        # print('ppos', ppos.shape)

        fpos = self.frame_pos_emb(torch.arange(t, device=self.gpu_id))
        # print('fpos', fpos.shape)
        pos_dim = fpos.shape[1]
        n_patches = n_patches_h * n_patches_w
        fpos = fpos.unsqueeze(2).expand(t,pos_dim,n_patches).transpose(1,2).reshape(t,n_patches_h,n_patches_w,pos_dim)
        # print('fpos', fpos.shape)

        xpos = torch.cat([fpos,ppos],dim=3)
        # print('xpos', xpos.shape)
        # print('patch_seq', patch_seq.shape)
        patch_seq += xpos

        # patch_seq = rearrange(patch_seq, 'b (t p) f -> b t p f', p=n_patches)
        patch_seq = rearrange(patch_seq, 'b t h w f -> b t (h w) f')
        
        return patch_seq

class CrossCacheAttention(nn.Module):
    def __init__(self, cache_names, n_layers, d_p, d_s, d_t, d_inner, n_head, d_k, d_v, dropout_p=0.1, dropout_s=0.1, dropout_t=0.1):
        super(CrossCacheAttention, self).__init__()
        self.n_layers = n_layers
        self.d_p = d_p
        self.d_s = d_s
        self.d_t = d_t
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_p = dropout_p
        self.dropout_s = dropout_s
        self.dropout_t = dropout_t
        self.cache_names = cache_names

        self.cache_symbols = []

        if 'spatial' in cache_names:
            self.cache_symbols.append('p')
        if 'temporal' in cache_names:
            self.cache_symbols.append('t')
        if 'structured' in cache_names:
            self.cache_symbols.append('s')
        
        self.streams = [''.join(list(perm)) for perm in list(permutations(self.cache_symbols, 2))]
        # # remove streams toward structured
        # self.streams = [stream_name for stream_name in self.streams if stream_name not in ['sp','st']]
        self.streams.extend(self.cache_symbols)

        self.cca_p_with_t = self.get_network('pt')
        self.cca_p_with_s = self.get_network('ps')
        self.cca_t_with_p = self.get_network('tp')
        self.cca_t_with_s = self.get_network('ts')
        self.cca_s_with_p = self.get_network('sp')
        self.cca_s_with_t = self.get_network('st')
        self.sa_s = self.get_network('s')

        self.p_proj = nn.Linear(2*self.d_p, d_p)
        self.t_proj = nn.Linear(2*self.d_t, d_t)
        self.s_proj = nn.Linear(2*self.d_s, d_s)

    def get_network(self, cca_type='t'):
        if cca_type not in self.streams:
            return None

        if cca_type == 'pt':
            d_model, attn_dropout = (self.d_t, self.d_p), self.dropout_t
        elif cca_type == 'st':
            d_model, attn_dropout = (self.d_t, self.d_s), self.dropout_t
        elif cca_type == 'ps':
            d_model, attn_dropout = (self.d_s, self.d_p), self.dropout_s
        elif cca_type == 'ts':
            d_model, attn_dropout = (self.d_s, self.d_t), self.dropout_s
        elif cca_type == 'tp':
            d_model, attn_dropout = (self.d_p, self.d_t), self.dropout_p
        elif cca_type == 'sp':
            d_model, attn_dropout = (self.d_p, self.d_s), self.dropout_p
        elif cca_type == 's':
            d_model, attn_dropout = self.d_s, self.dropout_s
        else:
            raise Exception('Unknow cca network type.')

        return TransformerEncoderBlock(n_layers=self.n_layers, 
                                       d_model=d_model, 
                                       d_inner=self.d_inner, 
                                       n_head=self.n_head,
                                       d_k=self.d_k, 
                                       d_v=self.d_v, 
                                       dropout=attn_dropout)

    def cca_stream(self, net, cache_alpha, cache_beta, pad_mask_q=None, pad_mask_k=None):
        if cache_alpha is None:
            return None,None

        if cache_beta is None:
            return cache_alpha,None

        if pad_mask_k is not None:
            attn_mask = get_attn_key_pad_mask(pad_mask_k, cache_alpha)
        else:
            attn_mask = None
        non_pad_mask=get_non_pad_mask(cache_alpha,pad_mask_q)

        return net(cache_alpha, non_pad_mask, enc_input_k=cache_beta, enc_input_v=cache_beta, attn_mask=attn_mask)

    def combine_stream(self, stream_proj, stream_out1, stream_out2, target_cache):
        if stream_out1 is not None and stream_out2 is not None:
            return stream_proj(torch.cat([stream_out1, stream_out2], dim=2))
        
        if stream_out1 is not None:
            return stream_out1
        
        if stream_out2 is not None:
            return stream_out2
        
        return target_cache

    def forward(self, spatial_cache, temporal_cache, structured_cache, pad_cache):
        spatial_temporal_cache, _ = self.cca_stream(self.cca_p_with_t, 
                                                    spatial_cache, 
                                                    temporal_cache,
                                                    pad_mask_k=pad_cache)
        spatial_structured_cache, _ = self.cca_stream(self.cca_p_with_s, 
                                                      spatial_cache, 
                                                      structured_cache)
        spatial_cross_cache = self.combine_stream(self.p_proj, 
                                                  spatial_temporal_cache, 
                                                  spatial_structured_cache, 
                                                  spatial_cache)

        temporal_spatial_cache, _ = self.cca_stream(self.cca_t_with_p, 
                                                    temporal_cache, 
                                                    spatial_cache,
                                                    pad_mask_q=pad_cache)
        temporal_structured_cache, _ = self.cca_stream(self.cca_t_with_s, 
                                                       temporal_cache, 
                                                       structured_cache,
                                                       pad_mask_q=pad_cache)
        temporal_cross_cache = self.combine_stream(self.t_proj, 
                                                   temporal_spatial_cache, 
                                                   temporal_structured_cache, 
                                                   temporal_cache)

        structured_temporal_cache, _ = self.cca_stream(self.cca_s_with_t, 
                                                        structured_cache, 
                                                        temporal_cache,
                                                        pad_mask_k=pad_cache)
        structured_spatial_cache, _ = self.cca_stream(self.cca_s_with_p, 
                                                      structured_cache, 
                                                      spatial_cache)
        structured_cross_cache = self.combine_stream(self.p_proj, 
                                                    structured_temporal_cache, 
                                                    structured_spatial_cache, 
                                                    structured_cache)
        if structured_cross_cache is not None:
            structured_logits = self.sa_s(structured_cross_cache, non_pad_mask=None)[0][-1] # use the last output for prediction
        else:
            structured_logits = None

        # cross_cache_lst = []
        # if 'pt' in self.streams:
        #     spatial_temporal_cache, _ = self.cca_p_with_t(spatial_cache, temporal_cache, temporal_cache)
        #     cross_cache_lst.append(spatial_temporal_cache)
        # if 'ps' in self.streams:
        #     spatial_structured_cache, _ = self.cca_p_with_s(spatial_cache, structured_cache, structured_cache)
        #     cross_cache_lst.append(spatial_structured_cache)
        # spatial_cross_cache = self.torch.cat(cross_cache_lst, dim=2)
        # if len(cross_cache_lst) == 2:
        #     spatial_cross_cache = self.p_proj(spatial_cross_cache)


        # cross_cache_lst = []
        # if 'tp' in self.streams:
        #     temporal_spatial_cache, _ = self.cca_t_with_p(temporal_cache, spatial_cache, spatial_cache)
        #     cross_cache_lst.append(temporal_spatial_cache)
        # if 'ts' in self.streams:
        #     temporal_structured_cache, _ = self.cca_t_with_s(temporal_cache, structured_cache, structured_cache)
        #     cross_cache_lst.append(temporal_structured_cache)
        # if len(cross_cache_lst) != 0:
        #     temporal_cross_cache = torch.cat(cross_cache_lst, dim=2)

        # cross_cache_lst = []
        # if 'st' in self.streams:
        #     structured_temporal_cache, _ = self.cca_s_with_t(structured_cache, temporal_cache, temporal_cache)
        #     cross_cache_lst.append(structured_temporal_cache)
        # if 'sp' in self.streams:
        #     structured_spatial_cache, _ = self.cca_s_with_p(structured_cache, spatial_cache, spatial_cache)
        #     cross_cache_lst.append(structured_spatial_cache)
        # if len(cross_cache_lst) != 0:
        #     structured_cross_cache = torch.cat(cross_cache_lst, dim=2)

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


