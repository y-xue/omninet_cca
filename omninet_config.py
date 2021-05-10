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
        'caa':False,
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

def cca_config():
    cnp_conf, perph_conf, domains = defaultconf()
    cnp_conf['max_clip_len'] = 16
    cnp_conf['max_patches'] = 49
    cnp_conf['caa'] = True
    cnp_conf['caa_caches'] = ['spatial', 'temporal', 'structured']
    cnp_conf['cca_n_layers'] = 6
    cnp_conf['cca_n_head'] = 8
    cnp_conf['cca_hidden_dim'] = 4096
    cnp_conf['cca_d_k'] = 64
    cnp_conf['cca_d_v'] = 64
    cnp_conf['dropout_p'] = 0.1
    cnp_conf['dropout_s'] = 0.1
    cnp_conf['dropout_t'] = 0.1
    cnp_conf['structured_dim'] = 512

    return cnp_conf, perph_conf, domains
