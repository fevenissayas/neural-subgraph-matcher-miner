#!/bin/bash
source .venv/bin/activate
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.venv/lib/python3.7/site-packages/torch/lib python3 -m subgraph_mining.decoder \
    --dataset=data/wiki-vote.pkl \
    --model_path=ckpt/model.pt \
    --batch_size=64 \
    --max_pattern_size=5 \
    --n_trials=10 \
    --out_path=results/wiki_motifs_real.json > output_motifs.txt 2>&1
