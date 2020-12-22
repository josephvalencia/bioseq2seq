#!/usr/bin/env bash
meme results/best_seq2seq/normed_IG_results/coding_topk_idx_motifs.fasta -dna -minw 3 -maxw 12 -mod oops -oc results/best_seq2seq/normed_IG_results/ -maxsize 350000
meme results/best_ED_classify/normed_IG_results/coding_topk_idx_motifs.fasta -dna -minw 3 -maxw 12 -mod oops -oc results/best_ED_classify/normed_IG_results/ -maxsize 350000

for l in {0..3}
do
for h in {0..7}
do
meme results/best_ED_classify/layer${l}head${h}results/coding_topk_idx_motifs.fasta -dna -minw 3 -maxw 12 -mod oops -oc results/best_ED_classify/layer${l}head${h}results/ -maxsize 350000
done
done
