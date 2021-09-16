def defaultconf():
    """
    The default confurigation as specified in the original paper

    """

    cnp_conf = {
        'input_dim':512,
        'control_dim':32,
        'output_dim':512,
        'spatial_dim':512,
        'temporal_dim':512,
        'structured_dim':512,
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
        'use_cca':False,
        'use_temporal_encoder':True,
        'use_vit_mlp':False,
        'dropout':0.1,
        'more_dropout':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'image_feature_dim': 2048,
        'image_feature_map_layer': 4,
        'dropout': 0.1 ,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'SIQ_output_classes':2,
        'no_BOS_EOS': False,
        'num_conti': 0,
        'num_cat_dict': {},
        'one_hot_only': True
    }

    domains = ['ENGLISH','GERMAN','IMAGE']

    return cnp_conf, perph_conf, domains

def default_struct():
    cnp_conf, perph_conf, domains = defaultconf()
    domains.append('STRUCT')
    perph_conf['entity_pretrained_emb_path'] = None
    perph_conf['se_dropout'] = 0.1
    return cnp_conf, perph_conf, domains
    
def cca_config():
    cnp_conf, perph_conf, domains = defaultconf()
    cnp_conf['max_clip_len'] = 16
    cnp_conf['max_patches_h'] = 7
    cnp_conf['max_patches_w'] = 7
    cnp_conf['patch_sizes'] = (2,2)
    cnp_conf['patch_stride'] = None
    cnp_conf['use_cca'] = True
    cnp_conf['use_patch'] = True
    cnp_conf['use_temporal_encoder'] = False
    cnp_conf['cca_caches'] = ['spatial', 'temporal', 'structured']
    cnp_conf['cca_streams'] = None
    cnp_conf['pos_emb_streams'] = None
    cnp_conf['cca_n_layers'] = 6
    cnp_conf['cca_n_heads'] = 8
    cnp_conf['sa_n_layers'] = 6
    cnp_conf['sa_n_heads'] = 8
    cnp_conf['psa_n_layers'] = 6
    cnp_conf['psa_n_heads'] = 8
    cnp_conf['cca_hidden_dim'] = 4096
    cnp_conf['cca_d_k'] = 64
    cnp_conf['cca_d_v'] = 64
    cnp_conf['default_attn_blocks'] = False
    cnp_conf['dropout_p'] = 0.1
    cnp_conf['dropout_s'] = 0.1
    cnp_conf['dropout_t'] = 0.1
    cnp_conf['drop_path_rate'] = 0.
    cnp_conf['structured_dim'] = 512

    perph_conf['image_feature_dim'] = 1024
    perph_conf['image_feature_map_layer'] = 3

    return cnp_conf, perph_conf, domains

def cca_struct_config():
    cnp_conf, perph_conf, domains = cca_config()
    domains.append('STRUCT')
    perph_conf['entity_pretrained_emb_path'] = None
    perph_conf['se_dropout'] = 0.1
    return cnp_conf, perph_conf, domains

def cca_d256_config():
    cnp_conf, perph_conf, domains = cca_config()
    cnp_conf['input_dim'] = 256
    cnp_conf['output_dim'] = 256
    cnp_conf['spatial_dim'] = 256
    cnp_conf['temporal_dim'] = 256
    cnp_conf['structured_dim'] = 256
    cnp_conf['temporal_n_layers'] = 3
    cnp_conf['temporal_n_heads'] = 4
    cnp_conf['temporal_hidden_dim'] = 1024
    cnp_conf['decoder_dim'] = 256 
    cnp_conf['decoder_n_layers'] = 3
    cnp_conf['decoder_n_heads'] = 4
    cnp_conf['decoder_hidden_dim'] = 1024

    cnp_conf['cca_n_layers'] = 3
    cnp_conf['cca_n_heads'] = 4
    cnp_conf['cca_hidden_dim'] = 1024

    return cnp_conf, perph_conf, domains
