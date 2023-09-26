#!/usr/bin/env bash
export dir="experiments/analysis/"

# Table 1 (Classification) 
python $dir/eval_classification.py --config $CONFIG
python $dir/eval_tools_homology.py experiments/competitor_results

# Table 2 (Micropeptides)
python $dir/eval_lncPEP_translation.py --config $CONFIG align > lncPEP_needle_cmds.txt
SGE_Array -c lncPEP_needle_cmds.txt
python $dir/eval_lncPEP_translation.py --config $CONFIG parse

# Fig 2 (mRNA and lncRNA translation) 
python $dir/eval_lncrna_translation.py --config $CONFIG align > needle_cmds.txt
python $dir/eval_mrna_translation.py --config $CONFIG align >> needle_cmds.txt
SGE_Array -c needle_cmds.txt
python $dir/eval_mrna_translation.py --config $CONFIG parse 
python $dir/eval_lncrna_translation.py --config $CONFIG parse 

# Fig 3, S1,S2,S3 (LFNet filters and attention metagenes)
python $dir/plot_filters.py --config $CONFIG
python $dir/nucleotide_metagene.py --config $CONFIG 

# all bin-based metagene plots Fig 4, S3,S6
python $dir/percentile_metagene.py --config $CONFIG  

# metric corr with ISM and inter-replicate corrs Fig 4,6, S5
python $dir/attr_correlations.py --config $CONFIG

# Fig 5, S4 (ISM synonymous metagenes, ISM example logos, enriched motifs)
python $dir/mutation_analysis.py --config $CONFIG
python $dir/plot_examples.py --config $CONFIG 

# shuffled variants, start and nonsense mutations Fig 4
python $dir/ism_sanity_checks.py --config $CONFIG

# Fig 5,6 motif discovery and analysis
python $dir/find_attr_spikes.py --config $CONFIG > streme_commands.txt
python $dir/find_attr_spikes.py --config $CONFIG --mask >> streme_commands.txt
python $dir/find_attr_spikes.py --config $CONFIG --mdig >> streme_commands.txt
python $dir/find_attr_spikes.py --config $CONFIG --mask --mdig >> streme_commands.txt
bash -c streme_commands.txt 
python $dir/parse_streme.py --config $CONFIG
python $dir/parse_streme.py --config $CONFIG --mask
python $dir/parse_streme.py --config $CONFIG --mdig
python $dir/parse_streme.py --config $CONFIG --mask --mdig

# ISM-tAI corr Fig 5, ISM-MDIG synon codon corr Fig 6
python $dir/codon_correlation.py --config $CONFIG
