import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import setup_fonts,parse_config, build_output_dir
import numpy as np

def run_checks_ISM(args,prefix,output_dir):

    bio = f"{args.best_BIO_DIR}/{prefix}.{args.reference_class}.{args.position}.ISM_mutation_scores.csv"
    edc = f"{args.best_EDC_DIR}/{prefix}.{args.reference_class}.{args.position}.ISM_mutation_scores.csv"
    
    bio_df = pd.read_csv(bio)
    bio_df['model'] = ['bioseq2seq' for _ in bio_df['delta'].tolist()]
    
    edc_df = pd.read_csv(edc)
    edc_df['model'] = ['EDC' for _ in edc_df['delta'].tolist()]
    df = pd.concat([bio_df,edc_df])
    # dummy var to reduce whitespace between violins 
    df['dummy'] = ['hack' for _ in range(len(df))] 
    stop_codons = ['TAG','TAA','TGA'] 
    df['is_stop_original'] = [True if x in stop_codons else False for x in df['original'].tolist()] 
    df['is_stop_mutated'] = [True if x in stop_codons else False for x in df['mutated'].tolist()] 
    order = ['bioseq2seq','EDC'] 
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(3.25,3),sharex=True)
    
    # start codons 
    disrupts_atg = (df['original'] == 'ATG') & (df['mutated'] != 'ATG')
    disrupts_start = df[(disrupts_atg) & (df['loc'] < 3)]
    g = sns.violinplot(data=disrupts_start,y='dummy',x='delta',ax=ax1,hue='model',cut=0,hue_order=order,linewidth=1,width=0.9)
    g.legend_.remove()
    ax1.set_yticks([]) 
    ax1.set_ylabel('') 
    ax1.set_xlabel(r'Start codon mutations $\Delta$S')
    sns.despine() 
    
    # early (<50 codons) nonsense mutations
    is_nonsense = (~df['is_stop_original']) & (df['is_stop_mutated'])
    early_nonsense = df[(is_nonsense) & (df['loc'] < 150)]
    g = sns.violinplot(data=early_nonsense,x='delta',y='dummy',ax=ax2,hue='model',cut=0,hue_order=order,width=0.9,linewidth=1)
    #g.legend_.remove()
    ax2.set_yticks([]) 
    ax2.set_ylabel('') 
    ax2.set_xlabel(r'Nonsense mutations first 50 codons $\Delta$S')
    sns.despine() 
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/mutations_ism_sanity.svg')
    plt.close()
    print(f'saved {output_dir}/mutations_ism_sanity.svg')

def run_checks_shuffled(args,prefix,output_dir,exclude=False):

    if exclude:
        homology = pd.read_csv("test_maximal_homology.csv")
        reduced = homology['score'] <=80
        homology = homology.loc[reduced]
        allowed = set()
        allowed.update(homology['ID'].tolist())
    storage = []
    for parent,model in zip([args.best_BIO_DIR,args.best_EDC_DIR],['bioseq2seq','EDC']):
        wildtype = np.load(f"{parent}/{prefix}.{args.reference_class}.{args.position}.logit.npz")
        dinuc_shuffled_5_prime= np.load(f"{parent}/{prefix}_2-nuc_shuffled_5-prime.{args.reference_class}.{args.position}.logit.npz")
        dinuc_shuffled_3_prime= np.load(f"{parent}/{prefix}_2-nuc_shuffled_3-prime.{args.reference_class}.{args.position}.logit.npz")
        mononuc_shuffled_5_prime= np.load(f"{parent}/{prefix}_1-nuc_shuffled_5-prime.{args.reference_class}.{args.position}.logit.npz")
        mononuc_shuffled_3_prime= np.load(f"{parent}/{prefix}_1-nuc_shuffled_3-prime.{args.reference_class}.{args.position}.logit.npz")
        shuffled_CDS = np.load(f"{parent}/{prefix}_3-nuc_shuffled_CDS.{args.reference_class}.{args.position}.logit.npz")
        
        is_coding = lambda x : x.startswith('XM') or x.startswith('NM') 
        for tscript,logit in wildtype.items():
            if exclude and tscript not in allowed:
                continue
            if is_coding(tscript):
                shuffle_ks = [2,1,2,1,3]
                names = ['5\' ' +r'UTR nt shuffled $\Delta$S',
                        '5\' ' +r'UTR nt shuffled $\Delta$S',
                        '3\' '+ r'UTR nt shuffled $\Delta$S',
                        '3\' '+ r'UTR nt shuffled $\Delta$S',
                        r'CDS codon shuffled $\Delta$S']
                datasets = [dinuc_shuffled_5_prime,
                            mononuc_shuffled_5_prime,
                            dinuc_shuffled_3_prime,
                            mononuc_shuffled_3_prime,shuffled_CDS]
                for shuffled,name,k in zip(datasets,names,shuffle_ks):
                    # find ids of shuffled equivalents
                    candidates = [x for x in shuffled.keys() if x.startswith(tscript)]
                    for c in candidates: 
                        logit_shuffled = shuffled[c]
                        diff = logit_shuffled - logit
                        # diff == 0.0 likely indicates a zero length region of interest so not shuffled 
                        if diff != 0.0:
                            # dummy var to assist plotting
                            short_region = '5-prime' 
                            entry = {'model' : model,'tscript' : c, 'label' : name, 'delta' : diff.item(),'k' : f'{k}'}
                            storage.append(entry)

    df = pd.DataFrame(storage)
    print(df) 
    order = ['bioseq2seq','EDC'] 
    #fig,axs = plt.subplots(3,1,figsize=(3.25,3),sharex=False) 
    fig,axs = plt.subplots(3,1,figsize=(3.75,3),sharex=True) 
    for ax, (label,group) in zip(axs.flat,df.groupby('label')):
        g = sns.violinplot(data=group,x='delta',y='k',ax=ax,hue='model',cut=0,hue_order=order,linewidth=1)
        #g.legend_.remove()
        if len(pd.unique(group['k'])) <= 1: 
            ax.set_yticks([]) 
            ax.set_ylabel('') 
        else:
            ax.set_ylabel('Shuffle k-mer',fontsize=8)
        ax.set_xlabel(label)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mutations_shuffled_sanity.svg')
    plt.close()
    print(f'saved {output_dir}/mutations_shuffled_sanity.svg')

if __name__ == "__main__":

    setup_fonts()
    args,unknown_args = parse_config()
    sns.set_style(style="whitegrid",rc={'font.family' : ['Helvetica']})
    output_dir = build_output_dir(args)

    prefix = "verified_test_RNA" 
    run_checks_ISM(args,prefix,output_dir)
    run_checks_shuffled(args,prefix,output_dir,exclude=True)
