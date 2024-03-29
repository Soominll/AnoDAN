# AnoDAN
### Anomalous gene detection using generative adversarial networks and graph neural networks for overcoming drug resistance

This repository provides relevant codes and data to find a combinatorial drug target using our framework called AnoDAN.

- AnoDAN : three code files for AnoDAN - execute Main.py first, then find a combinatorial target using anomaly_scoring_analysis.py
- Gene_expression : gene expression files for each pathway
- Pathway : data that contains genes and their interections in each pathway
- Data : list of pathways, sensitive and resistance cell lines.


## Key features
AnoDAN combines GAN and GNN to identify genes that can overcome drug resistance and to unravel the underlying mechanisms by incorporating pathway information. 
It accurately reproduces the distribution of gene expressions in the sensitive cell lines and identifies anomalous genes in the resistant cell lines.
It incorporates datasets with graphical structures using the biological pathway information of the genes to discover hidden mechanisms as well as target genes for overcoming the resistance.

## Implementation
- tensorflow_gpu>=2.1.0-rc1 (or) tensorflow>=2.1.0-rc1
- dm-sonnet>=2.0.0b0
- tensorflow_probability
- deepmind - graph_nets
- numpy, pandas, keras, umap, matplotlib, seaborn, pickle
 
After installing above requirements, you can clone this repository for usage.
