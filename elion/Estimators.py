# elion/Estimators.py
# ------------------------------------------------------------------
# Molecular‑property predictors & rewards manager (flat layout)
# ------------------------------------------------------------------

import importlib
from typing import Dict, List

import rdkit.Chem as Chem


class Estimators:
    """
    Collects property objects, converts predictions to rewards and
    aggregates a TOTAL reward per molecule.

    Each property class must expose:
        • predict(mols)  -> List[float]
        • reward(values) -> List[float]
        • rew_coeff, max_reward, optimize   (attributes)
        • check_and_adjust_property_threshold(values)   (optional)
    """

    # --------------------------------------------------------------

    def __init__(self, properties_cfg: Dict[str, Dict]):
        self.properties: Dict[str, object] = {}
        self.n_mols: int = 0
        self.all_converged: bool = False

        for prop in properties_cfg:
            print("-" * 80)
            print(f"Loading Property: {prop.upper()}")
            # ➊  FLAT‑PATH IMPORT
            module = importlib.import_module(f"elion.properties.{prop}")
            cls = getattr(module, prop)
            self.properties[prop] = cls(prop, **properties_cfg[prop])
            print(f"Done Loading Property: {prop.upper()}")

        # Maximum possible reward per molecule
        self.max_reward = sum(
            cls.rew_coeff * cls.max_reward for cls in self.properties.values()
        )
        print("-" * 80)
        print("Done reading properties.")
        print(
            f"The maximum possible reward per molecule is: "
            f"{self.max_reward:6.2f}"
        )
        print("Note: Maximum rewards only consider properties being optimized.")
        print("=" * 80)
        print()

    # --------------------------------------------------------------
    # 1.  Prediction helpers
    # --------------------------------------------------------------

    def estimate_properties(self, mols: List[Chem.Mol]) -> Dict[str, List[float]]:
        pred = {prop: [] for prop in self.properties}
        self.n_mols = len(mols)

        for prop, cls in self.properties.items():
            pred[prop] = cls.predict(mols)
        return pred

    # --------------------------------------------------------------
    # 2.  Reward helpers
    # --------------------------------------------------------------

    def estimate_rewards(self, predictions: Dict[str, List[float]]) -> Dict[str, List[float]]:
        rew: Dict[str, List[float]] = {}
        for prop, cls in self.properties.items():
            if cls.optimize:
                _values = predictions[prop]
                if len(_values) != self.n_mols:
                    raise ValueError(
                        f"{prop}: expected {self.n_mols} values, "
                        f"got {len(_values)}"
                    )
                rew[prop] = cls.reward(_values)
            else:
                rew[prop] = [0.0] * self.n_mols
        rew["TOTAL"] = self.total_reward(rew)
        return rew

    def total_reward(self, rewards: Dict[str, List[float]]) -> List[float]:
        total_rew = []
        for mol in range(self.n_mols):
            total = 0.0
            for prop, cls in self.properties.items():
                if cls.optimize:
                    total += rewards[prop][mol] * cls.rew_coeff
            total_rew.append(total)
        return total_rew

    # --------------------------------------------------------------
    # 3.  Threshold adjustment
    # --------------------------------------------------------------

    def check_and_adjust_thresholds(self, predictions: Dict[str, List[float]]):
        self.all_converged = True
        for prop, cls in self.properties.items():
            if cls.optimize:
                cls.check_and_adjust_property_threshold(predictions[prop])
                if not cls.converged:
                    self.all_converged = False

    # --------------------------------------------------------------
    # 4.  End‑to‑end convenience:  SMILES → reward
    # --------------------------------------------------------------

    # ➋  ACCEPT AND IGNORE EXTRA ARGS  (keeps RL loop happy)
    def smiles_reward_pipeline(self, smis: List[str], *_, **__) -> List[float]:
        """
        1) SMILES → RDKit mols
        2) properties → rewards → TOTAL
        """
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        predictions = self.estimate_properties(mols)
        rewards = self.estimate_rewards(predictions)
        return rewards["TOTAL"]
