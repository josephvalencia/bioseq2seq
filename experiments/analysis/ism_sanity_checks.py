import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import setup_fonts,parse_config, build_output_dir,palette_by_model
import numpy as np
import os,math

def combined_mutation_dataframe(args,prefix,output_dir,test_csv):

    models = {'seq-wt (LFN)' : f"{args.best_BIO_DIR}/{prefix}.{args.reference_class}.{args.position}.ISM_mutation_scores.csv",
            'seq-wt (CNN)' :  f"{args.best_CNN_DIR}/{prefix}.{args.reference_class}.{args.position}.ISM_mutation_scores.csv",
            'class (LFN)': f"{args.best_EDC_DIR}/{prefix}.{args.reference_class}.{args.position}.ISM_mutation_scores.csv"}

    test_df = pd.read_csv(test_csv,sep='\t')
   
    storage = []
    for model,csv in models.items():
        model_df = pd.read_csv(csv)
        model_df['Model'] = [model] * len(model_df)
        storage.append(model_df)

    df = pd.concat(storage)
    return df    
    

def plot_start_disruptions(df,ax_ticks,ax_lims,output_dir):

    # start codons
    plt.figure(figsize=(3.25,1.25))
    disrupts_atg = (df['original'] == 'ATG') & (df['mutated'] != 'ATG')
    disrupts_start = df[(disrupts_atg) & (df['loc'] < 3)]
    print('upstream ORFs') 
    print(disrupts_start.groupby('Model')['delta'].median())
    g = sns.violinplot(data=disrupts_start,y='Model',x='delta',hue='Model',cut=0,palette=palette_by_model(),linewidth=1,width=0.9,dodge=False,scale_hue=False)
    g.legend_.remove()
    g.set_xticks(ax_ticks)
    g.set_xlim(ax_lims)
    #ax.set_yticks([]) 
    #ax.set_ylabel('') 
    g.set_xlabel(r'Start codon mutations $\Delta$S')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/start_disruptions_sanity.svg')
    plt.close()
    print(f'saved {output_dir}/start_disruptions_sanity.svg')
     
def plot_nonsense_mutations(df,output_dir):

    sns.set_style(style="ticks",rc={'font.family' : ['Arial']})
    plt.figure(figsize=(3.75,2))
     
    bin_size = 50 # in codons
    stop_codons = ['TAG','TAA','TGA'] 
    df['is_stop_original'] = [True if x in stop_codons else False for x in df['original'].tolist()] 
    df['is_stop_mutated'] = [True if x in stop_codons else False for x in df['mutated'].tolist()] 
    
    binned = df['loc'] / (bin_size*3) 
    bins = [bin_size*math.floor(x) for x in binned]
    bins = [f'[{b}-\n{b+bin_size-1}]' for b in bins]
    df['binned_loc'] = bins #df['binned_loc'].apply(lambda x : 50*math.floor(x)) 
    
    # early (<50 codons) nonsense mutations
    is_nonsense = (~df['is_stop_original']) & (df['is_stop_mutated'])
    nonsense = df[(is_nonsense)]# & (df['loc'] < 150)]
    g = sns.lineplot(data=nonsense,y='delta',x='binned_loc',hue='Model',palette=palette_by_model(),err_style='bars',linewidth=1.5) 
    #g.legend_.remove()
    
    g.set_xticks(pd.unique(df['binned_loc']),fontsize=8) 
    g.set_xlabel(r'Codons')
    g.set_ylabel(r'Nonsense mutation mean $\Delta$S',fontsize=8)
    sns.despine() 
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/nonsense_mutations_ism_sanity.svg')
    plt.close()
    sns.set_style(style="whitegrid",rc={'font.family' : ['Arial']})
    print(f'saved {output_dir}/nonsense_mutations_ism_sanity.svg')

def run_checks_upstream(args,prefix,output_dir,exclude=False):

    if exclude:
        homology = pd.read_csv("test_maximal_homology.csv")
        reduced = homology['score'] <=80
        homology = homology.loc[reduced]
        allowed = set()
        allowed.update(homology['ID'].tolist())
    
    models = {'seq-wt (LFN)' : args.best_BIO_DIR,
            'seq-wt (CNN)' :  args.best_CNN_DIR,
            'class (LFN)': args.best_EDC_DIR}
    
    storage = []
    for model,parent in models.items():
        wildtype = np.load(f"{parent}/{prefix}.{args.reference_class}.{args.position}.logit.npz")
        swapped = np.load(f"{parent}/swapped_uORFs.{args.reference_class}.{args.position}.logit.npz")
        is_coding = lambda x : x.startswith('XM') or x.startswith('NM') 
        for tscript,logit in wildtype.items():
            if exclude and tscript not in allowed:
                continue
            if is_coding(tscript):
                candidates = [x for x in swapped.keys() if x.startswith(tscript)]
                for c in candidates: 
                    logit_shuffled = swapped[c]
                    diff = logit_shuffled - logit
                    # diff == 0.0 likely indicates a zero length region of interest so not shuffled 
                    if diff != 0.0:
                        # dummy var to assist plotting
                        short_region = '5-prime' 
                        entry = {'Model' : model,'tscript' : c, 'delta' : diff.item()}
                        storage.append(entry)

    df = pd.DataFrame(storage)
    print('upstream ORFs') 
    print(df.groupby('Model')['delta'].median())
    fig = plt.figure(figsize=(4.5,2)) 
    g = sns.violinplot(data=df,x='delta',y='Model',hue='Model',palette=palette_by_model(),cut=0,linewidth=1,dodge=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/upstream_swapped_sanity.svg')
    plt.close()
    print(f'saved {output_dir}/upstream_swapped_sanity.svg')

def run_checks_mutations(args,prefix,output_dir,mut_df,exclude=False):

    if exclude:
        homology = pd.read_csv("test_maximal_homology.csv")
        reduced = homology['score'] <=80
        homology = homology.loc[reduced]
        allowed = set()
        allowed.update(homology['ID'].tolist())
    storage = []
    models = {'seq-wt (LFN)' : args.best_BIO_DIR,
            'seq-wt (CNN)' :  args.best_CNN_DIR,
            'class (LFN)': args.best_EDC_DIR}
    for model,parent in models.items():
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
                        #'5\' ' +r'UTR nt shuffled $\Delta$S',
                        #'3\' '+ r'UTR nt shuffled $\Delta$S',
                        r'CDS codon shuffled $\Delta$S',
                        '3\' '+ r'UTR nt shuffled $\Delta$S']
                datasets = [dinuc_shuffled_5_prime,
                            #mononuc_shuffled_5_prime,
                            shuffled_CDS,
                            #mononuc_shuffled_3_prime,
                            dinuc_shuffled_3_prime]
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
                            entry = {'Model' : model,'tscript' : c, 'label' : name, 'delta' : diff.item(),'k' : f'{k}'}
                            storage.append(entry)

    df = pd.DataFrame(storage)
    print(df) 
    #fig,axs = plt.subplots(3,1,figsize=(3.25,3),sharex=False) 
    fig,axs = plt.subplots(3,1,figsize=(3.25,3),sharex=True) 
    for ax, (label,group) in zip(axs.flat,df.groupby('label')):
        print(label) 
        print(group.groupby('Model')['delta'].median())
        g = sns.violinplot(data=group,x='delta',y='Model',ax=ax,hue='Model',cut=0,palette=palette_by_model(),linewidth=1,dodge=False,scale_hue=False)
        g.legend_.remove()
        ''' 
        g = sns.violinplot(data=group,x='delta',y='k',ax=ax,hue='Model',cut=0,palette=palette_by_model(),linewidth=1)
        if len(pd.unique(group['k'])) <= 1: 
            ax.set_yticks([]) 
            ax.set_ylabel('') 
        else:
            ax.set_ylabel('Shuffle k-mer',fontsize=8)
        '''
        ax.set_xlabel(label)
    ticks = axs[-1].get_xticks()
    lims = axs[-1].get_xlim()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mutations_shuffled_sanity.svg')
    plt.close()
    print(f'saved {output_dir}/mutations_shuffled_sanity.svg')
    plot_start_disruptions(mut_df,ticks,lims,output_dir)

if __name__ == "__main__":

    setup_fonts()
    args,unknown_args = parse_config()
    test_csv = os.path.join(args.data_dir,args.test_prefix+'.csv')
    sns.set_style(style="whitegrid",rc={'font.family' : ['Arial']})
    output_dir = build_output_dir(args)

    prefix = "verified_test_RNA" 
    mut_df = combined_mutation_dataframe(args,prefix,output_dir,test_csv)
    plot_nonsense_mutations(mut_df,output_dir) 
    run_checks_mutations(args,prefix,output_dir,mut_df,exclude=True)
    run_checks_upstream(args,prefix,output_dir,exclude=True)
