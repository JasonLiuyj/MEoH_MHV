from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    decs: np.ndarray
    objs: np.ndarray
    cons: np.ndarray | None = None
    fes: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.decs = np.asarray(self.decs, dtype=float)
        self.objs = np.asarray(self.objs, dtype=float)
        if self.cons is not None:
            self.cons = np.asarray(self.cons, dtype=float)
        if self.fes is not None:
            self.fes = np.asarray(self.fes, dtype=float)

    def __len__(self) -> int:
        return int(self.objs.shape[0])

