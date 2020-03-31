import os
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import json
import pprint
from matplotlib import pyplot as plt

class SelfAttentionDistribution(object):

    """ Encapsulates summary functions for calculated self-attention distributions.
        Args:
            self_attn : (H,N,L,L) where H is number of self-attention heads, N is number of Transformer layers and L is input sequence length
        """
    def __init__(self,tscript_name,self_attn,seq,cds = None):

        self.tscript_name = tscript_name
        self.bulk_attn = self_attn
        self.seq = seq
        self.cds = cds

        if not os.path.isdir("output/"+self.tscript_name):
            os.mkdir("output/"+self.tscript_name)

    def summarize(self, format = "json"):

        N,H,L, _ = tuple(self.bulk_attn.shape)

        table = {}

        entry = {"ENSEMBL_ID": self.tscript_name}
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

                dist = self.bulk_attn[n,h,:,:]

                entropy = [float('%.3e' % (x)) for x in self.__attention_entropy__(dist).tolist()]
                max_attn = [x for x in self.__max_attention__(dist).tolist()]

                head_info = {"head":h,"h_x":entropy,"max":max_attn}
                heads.append(head_info)

            layer_info = {"layer":n ,"heads":heads}
            layers.append(layer_info)

        entry['layers'] = layers
        summary = json.dumps(entry)
        return summary

    def __attention_entropy__(self,attns):
        """ Shannon entropy of attention distribution.
           Args:
               attn : (L,L) attention distribution of one head at one layer
        """
        entropy = - (torch.log2(attns) * attns).sum(axis=1)
        return entropy.detach().cpu().numpy()

    def __center_of_attention__(self,attns,axis = 1):
        """ Finds index where 1/2 of cumulative weight is to the left and 1/2 to the right.
        """
        cumulative = torch.cumsum(attns,axis)
        mask = cumulative <= 0.5
        return mask.sum(axis = 1).detach().cpu().numpy()

    def __max_attention__(self,attns,axis=1):
        """ Finds index where maximum attention weight comes from.
        """
        max = torch.argmax(attns,dim=axis)
        return max.detach().cpu().numpy()

class AttentionGradientHandler(object):

    def __init__(self, encoder_params):
        self.encoder_params = encoder_params

    def store_grads(self,layer_name,grads):
        pass
