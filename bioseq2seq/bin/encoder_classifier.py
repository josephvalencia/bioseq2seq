import os
import argparse
import pandas as pd
import random
import torch
import numpy as np
import tqdm
import time
from datetime import datetime
from math import log, floor
from sklearn.metrics import accuracy_score

from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split , filter_by_length
from bioseq2seq.bin.models import make_transformer_seq2seq , make_transformer_classifier

def parse_args():
    """Parse required and optional configuration arguments.""" 
    
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--input",'--i',help = "File containing RNA to Protein dataset")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--max_tokens",type = int , default = 3000, help = "Max number of tokens in training batch")
    parser.add_argument("--max_len_transcript", type = int, default = 1000, help = "Maximum length of transcript")
    parser.add_argument("--checkpoint", "--c", help = "Name of .pt model to initialize training")

    return parser.parse_args()

def classify(args):

    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()
    
    # determined by GPU memory
    max_tokens_in_batch = args.max_tokens

    # raw GENCODE transcript data. cols = ['ID','RNA','PROTEIN']
    if args.input.endswith(".gz"):
        dataframe = pd.read_csv(args.input,sep="\t",compression = "gzip")
    else:
        dataframe = pd.read_csv(args.input,sep="\t")

    # obtain splits. Default 80/10/10. Filter below max_len_transcript
    df_train,df_test,df_val = train_test_val_split(dataframe,args.max_len_transcript,random_seed)
    # convert to torchtext.Dataset
    train,test,val = dataset_from_df(df_train.copy(),df_test.copy(),df_val.copy(),mode='D_classify')
    device = "cuda:0"

    criterion = torch.nn.NLLLoss(ignore_index=1,reduction='sum')

    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    model = restore_transformer_model(checkpoint,device,options)

    #train_iterator = iterator_from_dataset(train,max_tokens_in_batch,device,train=False)
    valid_iterator = iterator_from_dataset(val,max_tokens_in_batch,device,train=False)
    #test_iterator = iterator_from_dataset(test,max_tokens_in_batch,device,train=False)

    save_file = args.output_name + ".preds"

    loss,accuracy = evaluate(model,valid_iterator,criterion,vocab,save_file)

    print(accuracy)

def restore_transformer_model(checkpoint,device,opts):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    model = make_transformer_classifier(n_enc=4,model_dim=128,max_rel_pos=10)
    model.load_state_dict(checkpoint['model'],strict=False)
    model.generator.load_state_dict(checkpoint['generator'])
    model.to(device = device)
    return model

def evaluate(model,iterator,criterion,vocab,save_file):

    outFile = open(save_file,'w')
    model.eval()

    preds = []
    truth = []
    running_loss = 0.0

    tgt_vocab = vocab['tgt'].vocab
    src_vocab = vocab['src'].vocab

    with torch.no_grad():
        for batch in tqdm.tqdm(iterator):
            
            src, src_lens = batch.src
            tgt = torch.squeeze(batch.tgt,dim=0)
            tgt = torch.squeeze(tgt,dim=1)
            ids = batch.id
            
            outputs = model(src,src_lens)
            loss = criterion(outputs,tgt)
            running_loss += loss

            pred = torch.max(outputs.detach().cpu(),dim=1)
            
            for j, name in enumerate(ids):
                curr_pred = tgt_vocab.itos[pred.indices[j]]
                raw = src[:,j,:].reshape(-1).tolist()
                rna = "".join([src_vocab.itos[x] for x in raw])
                gold = tgt_vocab.itos[tgt[j]]
                outFile.write("ID: {}\nRNA: {}\nPRED: {}\nGOLD: {}\n\n".format(name,rna,curr_pred,gold))

            preds.append(pred.indices)
            truth.append(tgt.detach().cpu())

    preds = np.concatenate(preds)
    truth = np.concatenate(truth)

    outFile.close()
    return running_loss,accuracy_score(truth,preds)

if __name__ == "__main__":

    args = parse_args()
    classify(args)
