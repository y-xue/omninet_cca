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
from ..util import *
from libs.omninet.cnp.SubLayers import MultiHeadAttention, PositionwiseFeedForward, Mlp
from itertools import permutations
from libs.omninet.cnp.util import DropPath


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, vit_mlp=False, dropout=0.1, drop_path_rate=0.):
        super(EncoderLayer, self).__init__()
        print('EncoderLayer drop_path', drop_path_rate)
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        if isinstance(d_model, tuple):
            d_model = d_model[0]
        self.vit_mlp = vit_mlp
        if vit_mlp:
            self.pos_ffn = Mlp(d_model, d_inner, dropout=dropout)
        else:
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.drop_path = DropPath(drop_path_rate) # if drop_path > 0. else nn.Identity()

    def forward(self, enc_input,non_pad_mask, enc_input_k=None, enc_input_v=None, slf_attn_mask=None):
        # print('EncoderLayer forward')
        if enc_input_k is None and enc_input_v is None:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input,mask=slf_attn_mask,res=True, drop_path=self.drop_path)
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input_k, enc_input_v,mask=slf_attn_mask,res=True, drop_path=self.drop_path)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        if self.vit_mlp:
            enc_output = self.pos_ffn(enc_output, res=True, drop_path=self.drop_path)
        else:
            enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        return enc_output, enc_slf_attn

class CCAEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, drop_path_rate=0.):
        super(CCAEncoderLayer, self).__init__()
        print('CCAEncoderLayer')
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        if isinstance(d_model, tuple):
            d_model_alpha, _ = d_model
        else:
            d_model_alpha = d_model
        self.pos_ffn = Mlp(d_model_alpha, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model_alpha)
        self.layer_norm2 = nn.LayerNorm(d_model_alpha)
        self.drop_path = DropPath(drop_path_rate) # if drop_path > 0. else nn.Identity()

    def forward(self, enc_input,non_pad_mask, enc_input_k=None, enc_input_v=None, slf_attn_mask=None):
        # print('CCAEncoderLayer forward')
        x = enc_input
        enc_input = self.layer_norm1(x)
        if enc_input_k is None and enc_input_v is None:
            enc_output, enc_attn = self.slf_attn(
                enc_input, enc_input, enc_input,mask=slf_attn_mask)
        else:
            enc_output, enc_attn = self.slf_attn(
                enc_input, enc_input_k, enc_input_v,mask=slf_attn_mask)
        x = x + self.drop_path(enc_output)
        x = x + enc_output
        if non_pad_mask is not None: x*=non_pad_mask

        x = x + self.drop_path(self.pos_ffn(self.layer_norm3(x)))
        x = x + self.pos_ffn(self.layer_norm2(x))
        if non_pad_mask is not None: x*=non_pad_mask
        return x, enc_attn

class TransformerEncoderBlock(nn.Module):
    ''' Compose with multiple CCAEncoderLayer '''

    def __init__(self, n_layers, d_model, d_inner, n_head, d_k, d_v, default_attn_blocks=False, vit_mlp=False, dropout=0.1, drop_path_rate=0.):
        super(TransformerEncoderBlock, self).__init__()
        self.default_attn_blocks = default_attn_blocks
        if not default_attn_blocks:
            if isinstance(d_model, tuple):
                self.layer_norm_alpha = nn.LayerNorm(d_model[0])
                self.layer_norm_beta = nn.LayerNorm(d_model[1])
            else:
                self.layer_norm_alpha = nn.LayerNorm(d_model)
                self.layer_norm_beta = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            if default_attn_blocks:
                self.layers.append(EncoderLayer(d_model, 
                    d_inner, n_head, d_k, d_v, vit_mlp=vit_mlp, dropout=dropout, drop_path_rate=dpr[i]))
            else:
                self.layers.append(CCAEncoderLayer(d_model, 
                    d_inner, n_head, d_k, d_v, dropout=dropout, drop_path_rate=dpr[i]))

    def forward(self, enc_input, non_pad_mask, enc_input_k=None, enc_input_v=None, attn_mask=None):
        if not self.default_attn_blocks:
            if enc_input_k is not None and enc_input_v is not None and self.layer_norm_beta is not None:
                enc_input_k = self.layer_norm_beta(enc_input_k)
                enc_input_v = self.layer_norm_beta(enc_input_v)

        for layer in self.layers:
            if enc_input_k is not None and enc_input_v is not None:
                enc_input, enc_attn = layer(enc_input, non_pad_mask, enc_input_k, enc_input_v, slf_attn_mask=attn_mask)
            else:
                enc_input, enc_attn = layer(enc_input, non_pad_mask, slf_attn_mask=attn_mask)
        if not self.default_attn_blocks:
            enc_output = self.layer_norm_alpha(enc_input)
        else:
            enc_output = enc_input
        return enc_output, enc_attn

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
            dec_input, dec_input, dec_input,mask=slf_attn_mask, res=True)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        #Attend hidden states on the temporal cache
        dec_temp=self.temporal_proj(dec_output)

        dec_temp, dec_temp_attn = self.temporal_cache_attn(
            dec_temp, temporal_cache, temporal_cache,mask=dec_enc_attn_mask, res=True)
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
                                                           k_gate=spatial_gate, res=True)
            if non_pad_mask is not None: dec_spat*=non_pad_mask
               
        dec_output=self.spat_dec_proj(dec_spat)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        return dec_output,[dec_slf_attn,dec_spat_attn,dec_temp_attn]
