import os,sys
import re
from matplotlib import pyplot as plt

def plot_attentions(parent):
    
    plt.figure(figsize=(6.4, 2.4))
    plt.style.use('seaborn')

    enc_dec_attn_list = [parent+x for x in os.listdir(parent) if x.startswith("NC") and x.endswith(".enc_dec_attns")]
    enc_dec_attn_list = [x for x in enc_dec_attn_list if "layer3" in x]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for fh in sorted(enc_dec_attn_list):
        x_vals = []
        y_vals = []
        markers = []
        prefix = os.path.split(fh)[1]
        with open(fh) as inFile:
            for l in inFile:
                fields = l.rstrip().split("\t")
                x_vals.append(fields[0])
                y_vals.append(float(fields[1]))

        info = re.search("layer(\d)head(\d)",fh)
        layer = int(info.group(1))
        head = int(info.group(2))
        
        label = str(layer) if head == 0 else None

        if layer < 3:
            plt.plot(y_vals,label=label,color=colors[layer],alpha=0.5)
        else:
            plt.plot(y_vals,label=label,color=colors[layer])

    
    plt.ylabel("Attention")
    plt.xlabel("Position")
    plt.title("All Attentions")
    plt.legend(title="Layer")
    plt.tight_layout()
    #plt.savefig("all_layer_attentions.pdf")
    plt.savefig("layer3attentions.pdf")
    plt.close()

if __name__ == "__main__":

    plot_attentions(sys.argv[1])




