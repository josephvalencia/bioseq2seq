import os
import argparse
import torch
import numpy as np
from math import log, floor

from bioseq2seq.inputters import str2reader
from bioseq2seq.translate import Translator, GNMTGlobalScorer
from bioseq2seq.bin.models import restore_seq2seq_model
from bioseq2seq.utils.logging import init_logger, logger
from bioseq2seq.bin.data_utils import iterator_from_fasta
from bioseq2seq.inputters.corpus import maybe_fastafile_open
from bioseq2seq.transforms.transform import Transform

class NoOp(Transform):
    '''Hack for using translate_dynamic with no transforms'''
    def apply(self, example, is_train=False, stats=None, **kwargs):
        return example

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--save_EDA",help="Whether to save encoder-decoder attention", action="store_true")

    # translate required args
    parser.add_argument("--input",help="FASTA file for translation")
    parser.add_argument("--output_name","--o", default="translation",help="Name of file for saving predicted translations")
    parser.add_argument("--checkpoint","--c",help="Model checkpoint (.pt)")
    parser.add_argument("--max_tokens",type=int,default=9000,help="Max number of tokens in prediction batch")
    parser.add_argument("--mode",default="bioseq2seq",help="Inference mode. One of bioseq2seq|EDC")
    parser.add_argument("--rank",type=int,help="Rank of process",default=0)
    parser.add_argument("--num_gpus",type=int,help="Number of available GPU machines",default=0)

    # translate optional args
    parser.add_argument("--beam_size","--b",type=int, default=1, help ="Beam size for decoding")
    parser.add_argument("--n_best",type=int, default=1, help="Number of beam hypotheses to list")
    parser.add_argument("--max_decode_len",type=int, default=1, help="Maximum length of protein decoding")
    parser.add_argument("--attn_save_layer",type=int,default=-1,help="If --save_attn flag is used, which layer of EDA to save")
    return parser.parse_args()


def human_format(number):
    
    units = ['','K','M','G','T','P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def run_helper(rank,model,vocab,args):

    init_logger()
    
    scorer = GNMTGlobalScorer(alpha=0.0, 
                                beta=0.0, 
                                length_penalty="avg", 
                                coverage_penalty="none")
    
    gpu = rank if args.num_gpus > 0 else -1
    if gpu != -1:
        device = "cuda:{}".format(rank) 
        model.to(device)

    src_reader = str2reader["text"]
    tgt_reader = str2reader["text"]

    if not os.path.isdir(args.output_name):
        os.mkdir(args.output_name)
    
    input_filename = os.path.split(args.input)[-1].replace('.fasta','').replace('.fa','')
    outfile = open(f'{args.output_name}/{input_filename}_preds.txt','w')
    attnfile = None

    if args.save_EDA:
        model.decoder.attn_save_layer = args.attn_save_layer
        attnfile = open(f'{args.output_name}/EDA_layer{args.attn_save_layer}.npz','wb')
    
    src_text_field = vocab['src'].base_field
    src_vocab = src_text_field.vocab
    tgt_text_field = vocab['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    
    translator = Translator(model=model, 
                            fields=vocab,
                            out_file=outfile,
                            attn_file=attnfile,
                            src_reader=src_reader, 
                            tgt_reader=tgt_reader, 
                            global_scorer=scorer,
                            gpu=gpu,
                            beam_size=args.beam_size,
                            n_best=args.n_best,
                            max_length=args.max_decode_len)
    
    stride = args.num_gpus if args.num_gpus > 0 else 1
    offset = rank if stride > 1 else 0

    src = []
    tscripts = []
    
    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            if (i % stride) == offset:
                seq = str(record.seq)
                seq_whitespace =  ' '.join([c for c in seq])
                src.append(seq_whitespace.encode('utf-8'))
                tscripts.append(record.id)

    translator.translate_dynamic(src=src,
                        src_feats=None,
                        ids=tscripts,
                        transform=NoOp(opts={}),
                        batch_size=args.max_tokens,
                        batch_type="tokens",
                        attn_debug=args.save_EDA)
    outfile.close()
    print(f'predictions saved at {outfile}')
    if attnfile is not None:
        attnfile.close()

def translate_from_checkpoint(args):

    device = 'cpu'
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    
    logger.info("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
    if not options is None:
        for k,v in vars(options).items():
            logger.info(k,v)
    
    model = restore_seq2seq_model(checkpoint,device,options)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# trainable parameters = {human_format(num_params)}")
    model.eval()

    vocab = checkpoint['vocab']
    if args.num_gpus > 1:
        logger.info('Translating on {} GPUs'.format(args.num_gpus))
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(model,vocab,args))
    elif args.num_gpus > 0:
        logger.info('Translating on single GPU'.format(args.num_gpus))
        run_helper(args.rank,model,vocab,args)
    else:
        logger.info('Translating on single CPU'.format(args.num_gpus))
        run_helper(args.rank,model,vocab,args)

if __name__ == "__main__":

    args = parse_args()
    translate_from_checkpoint(args)
