# Spectral Diffusion and Ranking Stability in Protein Graph Representation Learning

**Kenny Yu**  
Project in Mathematics - Individual Report  
University of Sydney

## Overview

This project investigates self-supervised protein representation learning using a Personalised PageRank (PPR)-based Graph Neural Network.

A two-layer MLP encodes structural node features, followed by weighted APPNP diffusion. The project studies spectral filtering, fixed-point convergence, and Top-20 ranking stability, and compares GNN-based protein rankings against Node2Vec, RWR, CADTP, and degree-based baselines.

## 1. Environment Setup

Python 3.12 is recommended.

```bash
git clone https://github.com/kenny1031/math3888-individual.git
cd math3888-individual

python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
```

## 2. Required Data

Large raw datasets and generated files are not included in the repository.

Place the required files in the following structure:

```text
data/
├── metadata/
│   ├── YEAST-mod.gaf
│   ├── go-basic.obo
│   └── group_reference/
│       ├── cadtp_all_scores_refactor.csv
│       └── go_biological_process_labels_refactor.csv
├── processed/
│   └── functional_edges_score_ge_400.csv.gz
└── raw/
    ├── 4932.protein.aliases.v12.0.txt.gz
    ├── 4932.protein.info.v12.0.txt.gz
    ├── 4932.protein.links.full.v12.0.txt.gz
    └── essential_proteins.csv
```

**Data sources:**

- **STRING v12.0:** Yeast (*Saccharomyces cerevisiae*, taxon 4932) protein information, aliases, and association scores.
- **Gene Ontology:** Yeast GAF annotations (`YEAST-mod.gaf`) and ontology (`go-basic.obo`).
- **Group reference:** Previously generated CADTP rankings and GO labels from the MATH3888 Group K project.
- **Essential proteins:** The yeast essential-protein reference dataset used in the group project.

The graph is constructed using a STRING confidence threshold of **700**. The `score_ge_400` file is an intermediate input; the final threshold is applied during data preparation.

## 3. Reproduce Results

Run the following scripts from the repository root, using the configuration in `configs/default.yaml`.

**Step 1: Prepare data**

```bash
python scripts/prepare_data.py
```

Constructs the weighted protein network, extracts the HLJ1 connected component, and prepares structural node features.

**Step 2: Validate diffusion operators (optional)**

```bash
python scripts/test_diffusion.py
```

Checks graph operators and numerical properties of restarted diffusion.

**Step 3: Train GNN models**

```bash
python scripts/train_gnn.py
```

Trains self-supervised GNN models with propagation depths $K=0,2,4,8,16$.

**Step 4: Evaluate rankings**

```bash
python scripts/evaluate_aligned.py
```

Compares GNN rankings with the group baselines using GO Precision@20 and degree/distance-matched randomisation.

**Step 5: Analyse convergence and ranking stability**

```bash
python scripts/analyse_fixed_encoder.py
```

Uses the fixed $K=4$ encoder to investigate diffusion convergence, Dirichlet energy, fixed-point ranking margins, and Top-20 stability.

## 4. Outputs

Generated results are saved to the project's `outputs/` and `figures/` directories, including:

- GNN training and ranking evaluation results.
- Fixed-point convergence diagnostics.
- Spectral response and ranking stability figures.

## 5. Reproducibility Notes

The experiments use a continuation probability of $\alpha=0.85$, 64-dimensional embeddings, and random seed 3888 for GNN training.

All GNN models are trained without GO labels. GO annotations are used only for downstream evaluation.

For mathematical derivations, experimental interpretation, and limitations, please refer to the accompanying individual report.