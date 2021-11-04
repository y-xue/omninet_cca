import numpy as np
import torch

def load_vit_weights():
    name_map = {
        'Transformer/encoderblock_%s/LayerNorm_0/bias': 'cnp.cca.sa_p.layers.%s.slf_attn.layer_norm.bias',
        'Transformer/encoderblock_%s/LayerNorm_0/scale': 'cnp.cca.sa_p.layers.%s.slf_attn.layer_norm.weight',
        'Transformer/encoderblock_%s/LayerNorm_2/bias': 'cnp.cca.sa_p.layers.%s.pos_ffn.layer_norm.bias',
        'Transformer/encoderblock_%s/LayerNorm_2/scale': 'cnp.cca.sa_p.layers.%s.pos_ffn.layer_norm.weight',
        'Transformer/encoderblock_%s/MlpBlock_3/Dense_0/bias': 'cnp.cca.sa_p.layers.%s.pos_ffn.w_1.bias',
        'Transformer/encoderblock_%s/MlpBlock_3/Dense_0/kernel': 'cnp.cca.sa_p.layers.%s.pos_ffn.w_1.weight',
        'Transformer/encoderblock_%s/MlpBlock_3/Dense_1/bias': 'cnp.cca.sa_p.layers.%s.pos_ffn.w_2.bias',
        'Transformer/encoderblock_%s/MlpBlock_3/Dense_1/kernel': 'cnp.cca.sa_p.layers.%s.pos_ffn.w_2.weight',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/key/bias': 'cnp.cca.sa_p.layers.%s.slf_attn.w_ks.bias',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/key/kernel': 'cnp.cca.sa_p.layers.%s.slf_attn.w_ks.weight',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/query/bias': 'cnp.cca.sa_p.layers.%s.slf_attn.w_qs.bias',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/query/kernel': 'cnp.cca.sa_p.layers.%s.slf_attn.w_qs.weight',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/value/bias': 'cnp.cca.sa_p.layers.%s.slf_attn.w_vs.bias',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/value/kernel': 'cnp.cca.sa_p.layers.%s.slf_attn.w_vs.weight',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/bias': 'cnp.cca.sa_p.layers.%s.slf_attn.fc.bias',
        'Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/kernel': 'cnp.cca.sa_p.layers.%s.slf_attn.fc.weight',
        'Transformer/posembed_input/pos_embedding': 'cnp.cca.position_enc_p.weight',
    }
    pretrained_path = '/scratch1/yxuea/data/vit/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    w = np.load(pretrained_path)

    state_dict = {}
    for layer in range(12):
        for k in name_map:
            if 'pos_embedding' in k:
                state_dict[name_map[k]] = torch.from_numpy(w[k].reshape(-1,w[k].shape[-1]))
                continue

            param = w[k%layer]
            if 'key/bias' in k or 'query/bias' in k or 'value/bias' in k:
                state_dict[name_map[k]%layer] = torch.from_numpy(param.reshape(-1))
            elif 'key/kernel' in k or 'query/kernel' in k or 'value/kernel' in k:
                state_dict[name_map[k]%layer] = torch.from_numpy(param.reshape(param.shape[0],-1)) 
            elif 'out/kernel' in k:
                state_dict[name_map[k]%layer] = torch.from_numpy(param.reshape(-1, param.shape[-1])) 
            else:
                state_dict[name_map[k]%layer] = torch.from_numpy(param)
    return state_dict
    # torch.save(state_dict, )
