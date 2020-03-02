import os
import torch
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class SelfAttentionDistribution(object):

    """ Encapsulates summary functions for calculated self-attention distributions.
        Args:
            self_attn : (H,N,L,L) where H is number of self-attention heads, N is number of Transformer layers and L is input sequence length
        """
    def __init__(self,tscript_name,self_attn):

        self.tscript_name = tscript_name
        self.bulk_attn = self_attn

        if not os.path.isdir("output/"+self.tscript_name):
            os.mkdir("output/"+self.tscript_name)

    def summarize(self):

        N,H,L, _ = tuple(self.bulk_attn.shape)

        table = {}

        for n in range(N):
            for h in range(H):

                head_name = "layer-{}.head-{}".format(n,h)
                print("Summarizing {}.{}".format(self.tscript_name,head_name))

                dist = self.bulk_attn[n,h,:,:]

                self.plot_max(head_name,dist)
                self.plot_center(head_name,dist)

    def __attention_entropy__(self,attns):
        """
           Args:
               attn : (L,L) attention distribution of one head at one layers
        """
        entropy = - (torch.log2(attns) * attns).sum(axis=1)
        return entropy.cpu().numpy()

    def __center_of_attention__(self,attns,axis = 1):

        cumulative = torch.cumsum(attns,axis)
        mask = cumulative <= 0.5
        return mask.sum(axis = 1).cpu().numpy()

    def __max_attention__(self,attns,axis =1):

        max = torch.argmax(attns,dim=axis)
        return max.cpu().numpy()

    def plot_entropy(self,head_name,attns):
        """
            Args:
        """
        entropy = self.__attention_entropy__(attns)

        plt.plot(range(entropy.shape[0]),entropy)
        plt.ylabel("Entropy (bits)")
        plt.xlabel("Nucleotide")
        plt.title("Attention Entropy "+self.tscript_name+"."+head_name)
        plt.savefig("output/"+self.tscript_name+"/"+head_name+"_entropy.pdf")
        plt.close()

    def plot_max(self,head_name,attns,relative = False):
        """
            Args:
        """
        max_attns = self.__max_attention__(attns)

        if relative:
            offset = max_attns - np.arange(max_attns.shape[0])
            plt.scatter(range(max_attns.shape[0]),offset)
            plt.xlabel("Nuc Index")
            plt.ylabel("Max Index")
            plt.title("Max Attention (Relative) "+self.tscript_name+"."+head_name)
            plt.savefig("output/"+self.tscript_name+"/"+head_name+"_max_relative.pdf")
        else:
            plt.scatter(range(max_attns.shape[0]),max_attns)
            plt.xlabel("Nuc Index")
            plt.ylabel("Max Index")
            plt.title("Max Attention "+self.tscript_name+"."+head_name)
            plt.savefig("output/"+self.tscript_name+"/"+head_name+"_max.pdf")
        plt.close()

    def plot_center(self,head_name,attns,relative = False):
        """
            Args:
        """
        centers = self.__center_of_attention__(attns)

        if relative:
            offset = centers - np.arange(centers.shape[0])
            plt.xlabel("Nuc Index")
            plt.ylabel("Center Index")
            plt.scatter(range(centers.shape[0]),offset)
            plt.title("Center of Attention (Relative) "+self.tscript_name+"."+head_name)
            plt.savefig("output/"+self.tscript_name+"/"+head_name+"_center_relative.pdf")
        else:
            plt.scatter(range(centers.shape[0]),centers)
            plt.xlabel("Nuc Index")
            plt.ylabel("Center Index")
            plt.title("Center of Attention "+self.tscript_name+"."+head_name)
            plt.savefig("output/"+self.tscript_name+"/"+head_name+"_center.pdf")
        plt.close()

    def plot_heatmap(self,head_name,attns):

        df = pd.DataFrame.from_records(attns.cpu().numpy())
        ax = sns.heatmap(df,cmap="Blues")
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.title("Attention Heatmap "+self.tscript_name+"."+head_name)
        plt.savefig("output/"+self.tscript_name+"/"+head_name+"_heatmap.pdf")
        plt.close()

class AttentionGradientHandler(object):

    def __init__(self, encoder_params):
        self.encoder_params = encoder_params

    def store_grads(self,layer_name,grads):
        pass
