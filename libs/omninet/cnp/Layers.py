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

OmniNet temporal Encoder and spatio-temporal decoder layer

"""
import torch.nn as nn
import torch
from libs.omninet.cnp.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from itertools import permutations



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model_alpha, d_inner, dropout=dropout)

    def forward(self, enc_input,non_pad_mask, enc_input_k=None, enc_input_v=None, attn_mask=None):
        if enc_input_k is None and enc_input_v is None:
            enc_output, enc_attn = self.mha(
                enc_input, enc_input, enc_input,mask=attn_mask)
        else:
            enc_output, enc_attn = self.mha(
                enc_input, enc_input_k, enc_input_v,mask=attn_mask)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        return enc_output, enc_attn

class TransformerEncoderBlock(nn.Module):
    ''' Compose with multiple EncoderLayer '''

    def __init__(self, n_layers, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(d_model, 
                d_inner, n_head, d_k, d_v, dropout=dropout))

    def forward(self, enc_input, non_pad_mask, enc_input_k=None, enc_input_v=None, attn_mask=None):
        for layer in self.layers:
            if enc_input_k is not None and enc_input_v is not None:
                enc_input, enc_attn = layer(enc_input, non_pad_mask, enc_input_k, enc_input_v, attn_mask=attn_mask)
            else:
                enc_input, enc_attn = layer(enc_input, non_pad_mask, attn_mask=attn_mask)

        return enc_input, enc_attn

class CrossCacheAttentionLayer(nn.Module):
    def __init__(self, cache_names, n_layers, d_p, d_s, d_t, d_inner, n_head, d_k, d_v, dropout_p=0.1, dropout_s=0.1,, dropout_t=0.1):
        super(CrossCacheAttentionLayer, self).__init__()
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

    def cca_stream(self, net, cache_alpha, cache_beta):
        if cache_alpha is None:
            return None,None

        if cache_beta is None:
            return cache_alpha,None

        return net(cache_alpha, cache_beta, cache_beta)

    def combine_stream(self, stream_proj, stream_out1, stream_out2, target_cache):
        if stream_out1 is not None and stream_out2 is not None:
            return stream_proj(self.torch.cat([stream_out1, stream_out2], dim=2))
        
        if stream_out1 is not None:
            return stream_out1
        
        if stream_out2 is not None:
            return stream_out2
        
        return target_cache

    def forward(self, spatial_cache, temporal_cache, structured_cache):
        spatial_temporal_cache, _ = self.cca_stream(self.cca_p_with_t, 
                                                    spatial_cache, 
                                                    temporal_cache)
        spatial_structured_cache, _ = self.cca_stream(self.cca_p_with_s, 
                                                      spatial_cache, 
                                                      structured_cache)
        spatial_cross_cache = self.combine_stream(self.p_proj, 
                                                  spatial_temporal_cache, 
                                                  spatial_structured_cache, 
                                                  spatial_cache)

        temporal_spatial_cache, _ = self.cca_stream(self.cca_t_with_p, 
                                                    temporal_cache, 
                                                    spatial_cache)
        temporal_structured_cache, _ = self.cca_stream(self.cca_t_with_s, 
                                                       temporal_cache, 
                                                       structured_cache)
        temporal_cross_cache = self.combine_stream(self.t_proj, 
                                                   temporal_spatial_cache, 
                                                   temporal_structured_cache, 
                                                   temporal_cache)

        structured_temporal_cache, _ = self.cca_stream(self.cca_s_with_t, 
                                                        structured_cache, 
                                                        temporal_cache)
        structured_spatial_cache, _ = self.cca_stream(self.cca_s_with_p, 
                                                      structured_cache, 
                                                      spatial_cache)
        structured_cross_cache = self.combine_stream(self.p_proj, 
                                                    structured_temporal_cache, 
                                                    structured_spatial_cache, 
                                                    structured_cache)
        structured_logits = self.sa_s(structured_cross_cache)[0][-1] # use the last output for prediction

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


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, temporal_dim, spatial_dim,dropout=0.1,gpu_id=-1):
        super(DecoderLayer, self).__init__()
        self.gpu_id=gpu_id
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.temporal_cache_attn = MultiHeadAttention(n_head, temporal_dim, d_k, d_v, dropout=dropout)
        self.temporal_proj=nn.Linear(d_model,temporal_dim)
        self.spatial_proj=nn.Linear(temporal_dim,spatial_dim)
        self.spatial_cache_attn = MultiHeadAttention(n_head, spatial_dim, d_k, d_v, dropout=dropout)
        self.spat_dec_proj = nn.Linear(spatial_dim,d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


    def forward(self, dec_input,temporal_cache, spatial_cache,temporal_spatial_link,non_pad_mask,slf_attn_mask=None,dec_enc_attn_mask=None):
        #First attend the output encodings on itself
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input,mask=slf_attn_mask)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        #Attend hidden states on the temporal cache
        dec_temp=self.temporal_proj(dec_output)

        dec_temp, dec_temp_attn = self.temporal_cache_attn(
            dec_temp, temporal_cache, temporal_cache,mask=dec_enc_attn_mask)
        if non_pad_mask is not None: dec_temp*=non_pad_mask
        # Attend hidden states on the spatial cache
        dec_spat=self.spatial_proj(dec_temp)
        dec_spat_attn=None
        if spatial_cache is not None:
            # Process the spatial cache and add the respective weightings
            spatial_gate = []
            idx_start = 0
            for l in temporal_spatial_link:
                t, s = l
                if s > 1:
                    temp_sel = dec_temp_attn[:, :, :, idx_start:idx_start + t]
                    b, nh, dq, t = temp_sel.shape
                    temp_sel = temp_sel.unsqueeze(4).expand(b, nh, dq, t, s).transpose(3, 4)
                    temp_sel = temp_sel.reshape(b, nh, dq, t * s)
                    spatial_gate.append(temp_sel)
                idx_start = idx_start + t
            spatial_gate = torch.cat(spatial_gate, dim=3)
            dec_spat,dec_spat_attn=self.spatial_cache_attn(dec_spat,spatial_cache,spatial_cache,
                                                           k_gate=spatial_gate)
            if non_pad_mask is not None: dec_spat*=non_pad_mask
               
        dec_output=self.spat_dec_proj(dec_spat)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        return dec_output,[dec_slf_attn,dec_spat_attn,dec_temp_attn]