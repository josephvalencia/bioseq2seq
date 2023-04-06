#!/usr/bin/env bash
export CONFIG="example_config.yaml"
export SCRIPT_DIR="experiments/analysis/"

# Table 1 (Classification) 
python $SCRIPT_DIR/evaluate.py --config $CONFIG
python $SCRIPT_DIR/eval_tools.py experiments/competitor_results

# Table 2 (Micropeptides)
python $SCRIPT_DIR/eval_lncPEP_translation.py --config $CONFIG align > lncPEP_needle_cmds.txt
SGE_Array -c lncPEP_needle_cmds.txt
python $SCRIPT_DIR/eval_lncPEP_translation.py --config $CONFIG parse

# Fig 2 (mRNA and lncRNA translation) 
python $SCRIPT_DIR/eval_lncrna_translation.py --config $CONFIG align > needle_cmds.txt
python $SCRIPT_DIR/eval_mrna_translation.py --config $CONFIG align >> needle_cmds.txt
SGE_Array -c needle_cmds.txt
python $SCRIPT_DIR/eval_mrna_translation.py --config $CONFIG parse 
python $SCRIPT_DIR/eval_lncrna_translation.py --config $CONFIG parse 

# Fig 3, S1,S2,S3 (LFNet filters and attention metagenes)
python $SCRIPT_DIR/plot_filters.py --config $CONFIG
python $SCRIPT_DIR/nucleotide_metagene.py --config $CONFIG 

# all bin-based metagene plots Fig 4, S3,S6
python $SCRIPT_DIR/percentile_metagene.py --config $CONFIG  

# metric corr with ISM and inter-replicate corrs Fig 4,6, S5
python $SCRIPT_DIR/attr_correlations.py --config $CONFIG

# shuffled variants, start and nonsense mutations Fig 4
python $SCRIPT_DIR/ism_sanity_checks.py --config $CONFIG

# Fig 5, S4 (ISM synonymous metagenes, ISM example logos, enriched motifs)
python $SCRIPT_DIR/mutation_analysis.py --config $CONFIG
python $SCRIPT_DIR/plot_examples.py --config $CONFIG 

# Fig 5,6 motif discovery and analysis
python $SCRIPT_DIR/find_attr_spikes.py --config $CONFIG > streme_commands.txt
python $SCRIPT_DIR/find_attr_spikes.py --config $CONFIG --mask >> streme_commands.txt
python $SCRIPT_DIR/find_attr_spikes.py --config $CONFIG --mdig >> streme_commands.txt
python $SCRIPT_DIR/find_attr_spikes.py --config $CONFIG --mask --mdig >> streme_commands.txt
SGE_Array -c streme_commands.txt 
python $SCRIPT_DIR/parse_streme.py --config $CONFIG
python $SCRIPT_DIR/parse_streme.py --config $CONFIG --mask
python $SCRIPT_DIR/parse_streme.py --config $CONFIG --mdig
python $SCRIPT_DIR/parse_streme.py --config $CONFIG --mask --mdig

# ISM-tAI corr Fig 5, ISM-MDIG synon codon corr Fig 6
python $SCRIPT_DIR/codon_correlation.py --config $CONFIG
