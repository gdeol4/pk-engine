# coding: utf-8
import importlib.util
import pathlib
from typing import List

import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

try:
    import sascorer
except ImportError:
    sascorer = None

# find plapt model
_plapt_root = pathlib.Path(
    importlib.util.find_spec("plapt").submodule_search_locations[0]
)
_PLAPT_MODEL = _plapt_root / "models" / "affinity_predictor.onnx"

from plapt.plapt import Plapt

# helper functions
def _affinity(plapt_obj, seq: str, smis: List[str]) -> List[float]:
    try:
        res = plapt_obj.score_candidates(seq, smis)
        return [r["neg_log10_affinity_M"] for r in res]
    except Exception:
        return [0.0] * len(smis)

def _qed(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    return QED.qed(m) if m else 0.0

def _sa(smi: str) -> float:
    if sascorer is None:
        return 0.0
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0
    raw = sascorer.calculateScore(m)
    raw = max(1.0, min(raw, 10.0))
    return (10.0 - raw) / 9.0

def _lipinski(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0
    v = 0
    if Descriptors.NumHDonors(m)   > 5: v += 1
    if Descriptors.NumHAcceptors(m)>10: v += 1
    if Descriptors.MolWt(m)        >500: v += 1
    if Descriptors.MolLogP(m)      > 5: v += 1
    return 1.0 - v / 4.0

class MultiObjectiveReward:
    rew_coeff      = 1.0
    max_reward     = 1.0
    optimize       = True

    threshold       = 0.0
    threshold_limit = 1.0
    threshold_step  = 0.0
    thresh_limit    = threshold_limit
    thresh_step     = threshold_step
    converged       = False

    def __init__(
        self,
        *_,
        target_sequence: str = "",
        optimize: bool = True,
        threshold: float = 0.0,
        threshold_limit: float = 1.0,
        threshold_step: float = 0.0,
        **__
    ):
        if target_sequence == "":
            raise ValueError("target_sequence required")

        self.target_seq      = target_sequence
        self.optimize        = bool(optimize)
        self.threshold       = float(threshold)
        self.threshold_limit = float(threshold_limit)
        self.threshold_step  = float(threshold_step)
        self.thresh_limit    = self.threshold_limit
        self.thresh_step     = self.threshold_step
        self.converged       = False

        self.plapt = Plapt(
            prediction_module_path=str(_PLAPT_MODEL),
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_tqdm=False,
        )

    # ReLeaSE adapter
    def predict(self, mols):
        smis = [Chem.MolToSmiles(m) if m else "" for m in mols]
        return self(smis)

    @staticmethod
    def reward(values):
        return values

    def update_threshold(self, avg_val: float):
        if not self.optimize or self.converged:
            return
        if avg_val >= self.threshold_limit:
            self.converged = True
        else:
            self.threshold = min(self.threshold + self.threshold_step,
                                 self.threshold_limit)
            self.converged = self.threshold >= self.threshold_limit - 1e-9
    # Elion calls this every policy-iteration to drive convergence
    def check_and_adjust_property_threshold(self, values):
        """
        Required by Estimators.check_and_adjust_thresholds().
        It receives the list of predicted values for this property
        and decides whether to move the threshold.

        We simply forward the average value to update_threshold().
        """
        if not values:                     # empty list guard
            return
        avg_val = sum(values) / len(values)
        self.update_threshold(avg_val)

    def __call__(self, smiles: List[str]) -> List[float]:
        aff = _affinity(self.plapt, self.target_seq, smiles)
        qed = [_qed(s) for s in smiles]
        sa  = [_sa(s)  for s in smiles]
        lip = [_lipinski(s) for s in smiles]

        mn, mx = min(aff), max(aff)
        if mx > mn:
            aff_n = [(a - mn) / (mx - mn) for a in aff]
        else:
            aff_n = [0.5] * len(aff)

        return [
            0.6 * a + 0.1 * q + 0.1 * s + 0.2 * l
            for a, q, s, l in zip(aff_n, qed, sa, lip)
        ]

class PLAPTReward(MultiObjectiveReward):
    pass
