from __future__ import annotations

from algorithms.dynamic_algorithms import (
    DTAEA,
    ArchiveSPEA2SDE,
    ArchiveNSGAII,
    KTDMOEA,
    LEC,
    RestartNSGAII,
    RestartSPEA2SDE,
    STA,
    TrustArchiveNSGAII,
    TrustArchiveSPEA2SDE,
)
from algorithms.random_search import RandomSearch


class AlgorithmNotPortedError(NotImplementedError):
    pass


PORTED_ALGORITHMS = {
    "DTAEA": DTAEA,
    "ArchiveSPEA2SDE": ArchiveSPEA2SDE,
    "ArchiveNSGAII": ArchiveNSGAII,
    "KTDMOEA": KTDMOEA,
    "LEC": LEC,
    "STA": STA,
    "RestartNSGAII": RestartNSGAII,
    "RestartSPEA2SDE": RestartSPEA2SDE,
    "TrustArchiveNSGAII": TrustArchiveNSGAII,
    "TrustArchiveSPEA2SDE": TrustArchiveSPEA2SDE,
    "RandomSearch": RandomSearch,
}

MATLAB_ALGORITHMS_PENDING = set()


def create_algorithm(name: str, seed: int, pf_count: int = 5000, progress_interval: int = 25):
    if name in PORTED_ALGORITHMS:
        algorithm = PORTED_ALGORITHMS[name](seed=seed, pf_count=pf_count)
        algorithm.progress_interval = progress_interval
        return algorithm
    if name in MATLAB_ALGORITHMS_PENDING:
        raise AlgorithmNotPortedError(
            f"{name} has not been ported to pure Python yet. "
            "Use RandomSearch for pipeline testing, or port this algorithm into algorithms/ first."
        )
    raise KeyError(f"Unknown algorithm: {name}")
