"""Online optimization algorithms, benchmarks, and regret utilities."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

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
