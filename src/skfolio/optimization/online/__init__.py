"""Online optimization algorithms, benchmarks, and regret utilities."""

from skfolio.optimization.online._benchmark import BCRP, CRP, UCRP, BestStock
from skfolio.optimization.online._loser import (
    FTLStrategy,
    FollowTheLoser,
)
from skfolio.optimization.online._regret import RegretType, regret
from skfolio.optimization.online._winner import FTWStrategy, FollowTheWinner

__all__ = [
    "BCRP",
    "BestStock",
    "CRP",
    "FollowTheLoser",
    "FollowTheWinner",
    "FTLStrategy",
    "FTWStrategy",
    "regret",
    "RegretType",
    "UCRP",
]
