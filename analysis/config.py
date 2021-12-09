import yaml
import configargparse


def parse_config():

    p = configargparse.ArgParser() 
    
    p.add('--c','--config',required=False,is_config_file=True,help='path to config file')
    p.add('--test_csv',help='test dataset (.csv)' )
    p.add('--val_csv',help='validation dataset (.csv)')
    p.add('--train_csv',help='train dataset (.csv)')
    p.add('--competitors_results',help='competitors results (.csv)')
    p.add('--bioseq2seq_results',help='bioseq2seq results (.csv)')
    p.add('--EDC_results',help='EDC results (.csv)')
    p.add('--best_seq_self_attn',help='best bioseq2seq self-attention (.self_attn)')
    p.add('--best_EDC_self_attn',help='best EDC self-attention (.self_attn)')
    p.add('--best_seq_EDA',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_EDA',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_seq_IG',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_IG',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_seq_MDIG',help='best bioseq2seq MDIG (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_MDIG',help='best EDC MDIG (.ig)',type=yaml.safe_load)
    
    return p.parse_known_args()


def add_file_list(info_dict,label_field):

    p = info_dict['prefix']
    s = info_dict['suffix']
    labels = info_dict[label_field]
    file_list =  [f'{p}{v}{s}' for v in labels]
    info_dict['path_list'] = file_list
    return info_dict
