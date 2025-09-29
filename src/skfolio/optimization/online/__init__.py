"""Online optimization algorithms, benchmarks, and regret utilities."""

from skfolio.optimization.online._base import FTRLProximal, FTRLStrategy
from skfolio.optimization.online._benchmark import BCRP, CRP, UCRP
from skfolio.optimization.online._regret import RegretType, regret

__all__ = [
    "BCRP",
    "CRP",
    "UCRP",
    "FTRLProximal",
    "FTRLStrategy",
    "RegretType",
    "regret",
]
