import os

groups = [['PC','NC'],['PC','PC'],['NC','NC']]
metrics = [['max','max'],['max','min'],['max','random'],['min','random'],['random','random']]
#groups = [['PC','NC']]
#metrics = [['max','max']]

for g in groups:
    for m in metrics:

        for base in ['avg','zero']:
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            # ensure comparison groups are different
            if a != b:
                name = f'{a}_{b}'
                EDC_dir = f'new_attr/EDC_3_{name}/EDC_3_{base}_pos_test_summed_attr'
                seq2seq_dir = f'new_attr/seq2seq_3_{name}/seq2seq_3_{base}_pos_test_summed_attr'
                cmd1 = f'streme -p {seq2seq_dir}/positive_motifs.fa -n {seq2seq_dir}/negative_motifs.fa -oc {seq2seq_dir}/streme_out -rna -minw 4 -maxw 9 -pvt 1e-5 -kmer 4 -patience 0' 
                cmd2 = f'streme -p {EDC_dir}/positive_motifs.fa -n {EDC_dir}/negative_motifs.fa -oc {EDC_dir}/streme_out -rna -minw 4 -maxw 9 -pvt 1e-5 -kmer 4 -patience 0'
                os.system(cmd1)
                os.system(cmd2)
        
        ''' 
        for l in range(4):
            for h in range(8):
                a = f'{g[0]}-{m[0]}'
                b = f'{g[1]}-{m[1]}'
                # ensure comparison groups are different
                if a != b:
                    name = f'{a}_{b}'
                    EDC_dir = f'new_attr/EDC_3_{name}/EDC_3_test_layer{l}_layer{l}head{h}'
                    seq2seq_dir = f'new_attr/seq2seq_3_{name}/seq2seq_3_test_layer{l}_layer{l}head{h}'
                    cmd1 = f'streme -p {seq2seq_dir}/positive_motifs.fa -n {seq2seq_dir}/negative_motifs.fa -oc {seq2seq_dir}/streme_out -rna -minw 4 -maxw 9 -pvt 1e-5 -kmer 4 -patience 0' 
                    cmd2 = f'streme -p {EDC_dir}/positive_motifs.fa -n {EDC_dir}/negative_motifs.fa -oc {EDC_dir}/streme_out -rna -minw 4 -maxw 9 -pvt 1e-5 -kmer 4 -patience 0'
                    os.system(cmd1)
                    os.system(cmd2)
        '''
