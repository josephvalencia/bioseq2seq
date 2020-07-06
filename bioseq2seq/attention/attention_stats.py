import os
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import scipy

class AttentionDistribution:

    def __init__(self,tscript_name,attn,seq,cds):

        self.tscript_name = tscript_name
        self.attn = attn
        self.seq = seq
        self.cds = cds

        if not os.path.isdir("output/"+self.tscript_name):
            os.mkdir("output/"+self.tscript_name)

    def __attention_entropy__(self,attns,axis=1):
        """ Shannon entropy of attention distribution.
           Args:
               attn : (L,L) attention distribution of one head at one layer
        """
        attns = attns.detach().cpu().numpy()
        attns[attns == 0.0] = np.nan
        attns = np.ma.array(attns,mask=np.isnan(attns))
        entropy = -(np.log2(attns) * attns).sum(axis=axis)
        return entropy

    def __center_of_attention__(self,attns,axis=1):
        """ Finds index where 1/2 of cumulative weight is to the left and 1/2 to the right."""
        cumulative = torch.cumsum(attns,axis)
        mask = cumulative <= 0.5
        return mask.sum(axis = 1).detach().cpu().numpy()

    def __max_attention__(self,attns,axis=1):
        """ Finds index where maximum attention weight comes from."""
        max = torch.argmax(attns,dim=axis)
        return max.detach().cpu().numpy()

    def __max__activation(self,attns,axis=1):
        max = torch.max(attns,dim=axis)
        return max.detach().cpu().numpy()

class SelfAttentionDistribution(AttentionDistribution):

    """ Encapsulates summary functions for calculated self-attention distributions.
        Args:
            self_attn : (H,N,L,L) where H is number of self-attention heads, N is number of Transformer layers and L is input sequence length
    """
    
    def __init__(self,tscript_name,self_attn,seq,cds):

        super().__init__(tscript_name,self_attn,seq,cds)

    def summarize(self, format = "json"):

        N,H,L, _ = tuple(self.attn.shape)

        entry = {"TSCRIPT_ID": self.tscript_name}
        entry['seq'] = self.seq

        if not self.cds is None:
            entry['CDS_START'] = self.cds[0]
            entry['CDS_END'] = self.cds[1]
        else:
            entry['CDS_START'] = -1
            entry['CDS_END'] = -1

        layers = []

        for n in range(N):
            layer = n
            heads = []
            for h in range(H):

                dist = self.attn[n,h,:,:]
                entropy = [round(x,3) for x in self.__attention_entropy__(dist).tolist()]
                max_attn = [x for x in self.__max_attention__(dist).tolist()]

                head_info = {"head" : h , "h_x" : entropy,"max" : max_attn}
                heads.append(head_info)

            layer_info = {"layer":n ,"heads":heads}
            layers.append(layer_info)

        entry['layers'] = layers
        summary = json.dumps(entry)
        return summary

class EncoderDecoderAttentionDistribution(AttentionDistribution):

    def __init__(self,tscript_name,enc_dec_attn,seq,cds):
        
        super().__init__(tscript_name,enc_dec_attn,seq,cds)

    def summarize(self,position = 0,layers = [0]):

        layers = list(range(len(self.attn))) if layers is None else layers
        entry = {"TSCRIPT_ID": self.tscript_name}

        for l in layers:
            enc_dec = self.attn[l][position,:len(self.seq)].detach().cpu().tolist()
            field = "layer_{}_pos_{}".format(l,position)
            entry[field] = enc_dec

        return json.dumps(entry)

    def plot_heatmap(self):

        first_pos = self.attn[0][:,:len(self.seq)].cpu().numpy()
        name = self.tscript_name + "enc_dec_attn.png"

        ax = sns.heatmap(np.transpose(first_pos),cmap="YlGnBu")
        plt.savefig(name)
        plt.close()

    def plot_entropy(self):
        
        """ Plot nucleotide position vs Shannon entropy of attention.
            Args:
        """

        first = self.attn[0][:,:len(self.seq)]
        entropy = [round(x,3) for x in self.__attention_entropy__(first).tolist()]
        x = np.arange(len(entropy))

        name = self.tscript_name+"enc_dec_attn_entropy.png"

        plt.plot(x,entropy)
        plt.ylabel("Entropy (bits)")
        plt.xlabel("Nucleotide")
        plt.title("Enc-Dec Attention Entropy")
        plt.savefig(name)
        plt.close()