"""Datasets module."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.datasets._base import (
    load_factors_dataset,
    load_ftse100_dataset,
    load_nasdaq_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
    load_sp500_index,
)
from skfolio.datasets._relatives import (
    load_cmc20_relatives_dataset,
    load_djia_relatives_dataset,
    load_msci_relatives_dataset,
    load_nyse_o_relatives_dataset,
    load_sp500_relatives_dataset,
    load_tse_relatives_dataset,
)

__all__ = [
    "load_cmc20_relatives_dataset",
    "load_djia_relatives_dataset",
    "load_factors_dataset",
    "load_ftse100_dataset",
    "load_msci_relatives_dataset",
    "load_nasdaq_dataset",
    "load_nyse_o_relatives_dataset",
    "load_sp500_dataset",
    "load_sp500_implied_vol_dataset",
    "load_sp500_index",
    "load_sp500_relatives_dataset",
    "load_tse_relatives_dataset",
]
