#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lut_plain_test.py
=================
Quick end-to-end smoke test for the streamlined Elion + Plapt pipeline.

Folder layout assumed::

    project_root/
    +-- elion/
    ¦   +-- data/
    ¦   ¦   +-- chembl_22_clean_1576904_sorted_std_final.smi
    ¦   +-- Estimators.py
    ¦   +-- Generator.py
    +-- plapt/
    ¦   +-- models/
    ¦   ¦   +-- affinity_predictor.onnx
    ¦   +-- plapt.py
    +-- lut_plain_test.py      <-- this file
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# ---------------------------------------------------------------------------
# 0.  Ensure the project root is on sys.path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import sascorer  # noqa: E402  (optional dependency)
except ImportError:  # pragma: no cover
    sascorer = None

# ---------------------------------------------------------------------------
# 1.  Locate data/model files
# ---------------------------------------------------------------------------

ELION_DATA = ROOT / "elion" / "data" / "chembl_22_clean_1576904_sorted_std_final.smi"
Plapt_ONNX = ROOT / "plapt" / "models" / "affinity_predictor.onnx"

if not ELION_DATA.exists():
    raise FileNotFoundError(f"SMILES data file not found: {ELION_DATA}")
if not Plapt_ONNX.exists():
    raise FileNotFoundError(f"ONNX model file not found: {Plapt_ONNX}")

# ---------------------------------------------------------------------------
# 2.  Generator + reward configuration
# ---------------------------------------------------------------------------

SEED_SMI = "O=c1c(O)c(-c2ccccc2O)oc2c(O)c(O)ccc12"

generator_cfg = {
    "name": "ReLeaSE",
    "seed_smi": SEED_SMI,
    "batch_size": 32,
    "max_iterations": 2,
    "data_path": str(ELION_DATA),  # <-- relative-safe path
}

SOD1_SEQ = (
    "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTS"
    "AGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVV"
    "HEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ"
)

reward_cfg = {
    "PLAPTReward": {
        "name": "PLAPTReward",
        "target_sequence": SOD1_SEQ,
        "threshold": 1.0,
        "threshold_limit": 1.0,
        "threshold_step": 0.0,
        "optimize": True,
        "prediction_module_path": Plapt_ONNX,
    }
}

control_cfg = {
    "run_type": "bias_generator",
    "verbosity": 1,
    "history_file": "bias_history.csv",
    "comment": "Pipeline smoke test",
    "restart": False,
}

# ---------------------------------------------------------------------------
# 3.  Import project modules
# ---------------------------------------------------------------------------

from elion.Generator import Generator  # noqa: E402
from elion.Estimators import Estimators  # noqa: E402
from plapt.plapt import Plapt  # noqa: E402

# ---------------------------------------------------------------------------
# 4.  Instantiate pipeline pieces
# ---------------------------------------------------------------------------

gen = Generator(generator_cfg).generator
props = Estimators(reward_cfg)
plapt = Plapt(prediction_module_path=Plapt_ONNX,
              device=("cuda" if torch.cuda.is_available() else "cpu"),
              use_tqdm=False)

print("Device:", "GPU" if torch.cuda.is_available() else "CPU")

# ---------------------------------------------------------------------------
# 5.  Bias generator (stub) and sample molecules
# ---------------------------------------------------------------------------

print("Biasing generator …")
gen.bias_generator(control_cfg, props)

def sample(g, n: int, temperature: float = 1.2) -> List[str]:
    try:
        return g.generate_smis(n_smis=n, temperature=temperature)
    except TypeError:  # fallback for stub implementation
        pool = g.generate_smis()
        random.shuffle(pool)
        return pool[:n]

RAW_N = 120
smiles_pool = sample(gen, RAW_N)

valid_smi = [s for s in smiles_pool if Chem.MolFromSmiles(s)]
unique_smi = list(dict.fromkeys(valid_smi))
if not unique_smi:
    print("No valid SMILES produced; exiting.")
    sys.exit(0)

print("Molecules passed to Plapt:", len(unique_smi))

# ---------------------------------------------------------------------------
# 6.  Score with Plapt and basic chemistry metrics
# ---------------------------------------------------------------------------

aff_res = plapt.score_candidates(SOD1_SEQ, unique_smi)
aff = [r["neg_log10_affinity_M"] for r in aff_res]

def qed(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol) if mol else 0.0

def sa(smiles: str) -> float:
    if not sascorer:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0
    raw = sascorer.calculateScore(mol)
    raw = max(1, min(raw, 10))
    return (10 - raw) / 9.0

def lipinski(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0
    v = 0
    if Descriptors.NumHDonors(mol) > 5: v += 1
    if Descriptors.NumHAcceptors(mol) > 10: v += 1
    if Descriptors.MolWt(mol) > 500: v += 1
    if Descriptors.MolLogP(mol) > 5: v += 1
    return 1.0 - v / 4.0

qed_scores = [qed(s) for s in unique_smi]
sa_scores = [sa(s) for s in unique_smi]
lip_scores = [lipinski(s) for s in unique_smi]

a_min, a_max = min(aff), max(aff)
norm_aff = [
    (a - a_min) / (a_max - a_min) if a_max > a_min else 0.5
    for a in aff
]

final_scores = [
    0.6 * a + 0.1 * q + 0.1 * s + 0.2 * l
    for a, q, s, l in zip(norm_aff, qed_scores, sa_scores, lip_scores)
]

ranked = sorted(
    zip(unique_smi, final_scores), key=lambda x: x[1], reverse=True
)

# ---------------------------------------------------------------------------
# 7.  Show top hits
# ---------------------------------------------------------------------------

TOP_N = min(15, len(ranked))
print(f"\nTop {TOP_N} molecules:")
for idx, (smi, score) in enumerate(ranked[:TOP_N], 1):
    print(f"{idx:2d}. {smi}   reward={score:.3f}")
