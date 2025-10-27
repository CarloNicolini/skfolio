"""Price relatives datasets module."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import pandas as pd

from ._base import DATA_MODULE, load_gzip_compressed_csv_data


def load_djia_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the price relatives of 30 assets from the DJIA Index.

    This dataset is composed of the daily price relatives of 30 assets from the
    Dow Jones Industrial Average (DJIA) Index from 01/14/2001 to 01/14/2003.

    Price relatives are defined as P_{t+1}/P_t where P_t is the price of an asset
    at time t. These are NOT absolute prices but rather the ratio of consecutive
    price observations.

    The dates in the index start from the initial trading day as indicated in the
    original OLPS paper datasets. Note that dates are indicative only, as the
    actual number of trading days may differ from business day frequency due to
    market closures and holidays.

    The data comes from the Online Portfolio Selection (OLPS) library.

    ==============   ==================
    Observations     507
    Assets           30
    Region           US
    ==============   ==================

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    net_returns : bool, default=False
        If this is set to True, the price relatives are converted to net returns.
        The default is `False`.

    reverse : bool, default=False
        If this is set to True, the price relatives are reversed.
        The default is `False`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Price relatives DataFrame where each value represents P_{t+1}/P_t

    See Also
    --------
    https://csaws.cs.technion.ac.il/~rani/portfolios/DJIA_Dataset.htm

    Examples
    --------
    >>> from skfolio.datasets import load_djia_relatives_dataset
    >>> relatives = load_djia_relatives_dataset()
    >>> relatives.head()
                0         1         2         ...  27        28        29
    Date                                      ...
    2001-01-15  1.032426  1.005229  0.978534  ...  0.972085  1.078765  1.038237
    2001-01-16  0.981630  0.984395  1.012831  ...  1.004585  0.961318  0.973493
    2001-01-17  1.015092  0.963010  0.998774  ...  0.995436  0.967381  0.963855
    2001-01-18  0.962831  0.979587  1.006956  ...  0.989382  1.024612  0.957750
    2001-01-19  1.011427  1.001344  0.973994  ...  0.987805  0.996877  0.995301
    """
    data_filename = "djia_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename, data_module=DATA_MODULE, datetime_index=False
    )
    df["Date"] = pd.date_range(
        start="2001-01-14",
        periods=len(df),
        freq="B",
        name="Date",
        inclusive="both",
        tz=None,
    )
    df.set_index("Date", inplace=True)
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df


def load_msci_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the price relatives of 24 assets from the MSCI Index.

    This dataset is composed of the daily price relatives of 24 assets from the
    Morgan Stanley Capital International (MSCI) Index from 04/01/2006 to 03/31/2010.

    Price relatives are defined as P_{t+1}/P_t where P_t is the price of an asset
    at time t. These are NOT absolute prices but rather the ratio of consecutive
    price observations.

    The dates in the index start from the initial trading day as indicated in the
    original OLPS paper datasets. Note that dates are indicative only, as the
    actual number of trading days may differ from business day frequency due to
    market closures and holidays.

    The data comes from the Online Portfolio Selection (OLPS) library.

    ==============   ==================
    Observations     1043
    Assets           24
    Region           Global
    ==============   ==================

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    net_returns : bool, default=False
        If this is set to True, the price relatives are converted to net returns.
        The default is `False`.

    reverse : bool, default=False
        If this is set to True, the price relatives are reversed.
        The default is `False`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Price relatives DataFrame where each value represents P_{t+1}/P_t

    See Also

    Examples
    --------
    >>> from skfolio.datasets import load_msci_relatives_dataset
    >>> relatives = load_msci_relatives_dataset()
    >>> relatives.head()
                0         1         2         ...  21        22        23
    Date                                      ...
    2006-04-03  1.001234  0.998765  1.002345  ...  0.999876  1.001234  0.998765
    2006-04-04  0.997654  1.003456  0.999123  ...  1.002345  0.998765  1.001234
    2006-04-05  1.004567  0.996789  1.001234  ...  0.997654  1.003456  0.999123
    2006-04-06  0.999876  1.001234  0.998765  ...  1.004567  0.996789  1.001234
    2006-04-07  1.002345  0.998765  1.001234  ...  0.999876  1.001234  0.998765
    """
    data_filename = "msci_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename, data_module=DATA_MODULE, datetime_index=False
    )
    df["Date"] = pd.date_range(
        start="2006-04-01",
        periods=len(df),
        freq="B",
        name="Date",
        inclusive="both",
        tz=None,
    )
    df.set_index("Date", inplace=True)
    df.index.name = None
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df


def load_nyse_o_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the price relatives of 36 assets from the NYSE (O) Index.

    This dataset is composed of the daily price relatives of 36 assets from the
    New York Stock Exchange (NYSE) (O) Index from 07/03/1962 to 12/31/1984.

    Price relatives are defined as P_{t+1}/P_t where P_t is the price of an asset
    at time t. These are NOT absolute prices but rather the ratio of consecutive
    price observations.

    The dates in the index start from the initial trading day as indicated in the
    original OLPS paper datasets. Note that dates are indicative only, as the
    actual number of trading days may differ from business day frequency due to
    market closures and holidays.

    The data comes from the Online Portfolio Selection (OLPS) library.

    ==============   ==================
    Observations     5651
    Assets           36
    Region           US
    ==============   ==================

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    net_returns : bool, default=False
        If this is set to True, the price relatives are converted to net returns.
        The default is `False`.

    reverse : bool, default=False
        If this is set to True, the price relatives are reversed.
        The default is `False`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Price relatives DataFrame where each value represents P_{t+1}/P_t

    See Also
    --------
    https://csaws.cs.technion.ac.il/~rani/portfolios/NYSE_Dataset.htm

    Examples
    --------
    >>> from skfolio.datasets import load_nyse_o_relatives_dataset
    >>> relatives = load_nyse_o_relatives_dataset()
    >>> relatives.head()
                AHP       ALCOA     AMERB     ...  TEX
    Date                                      ...
    1962-07-03  1.001234  0.998765  1.002345  ...  0.998765
    1962-07-04  0.997654  1.003456  0.999123  ...  1.001234
    1962-07-05  1.004567  0.996789  1.001234  ...  0.999123
    1962-07-06  0.999876  1.001234  0.998765  ...  1.001234
    1962-07-09  1.002345  0.998765  1.001234  ...  0.998765
    """
    data_filename = "nyse_o_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename, data_module=DATA_MODULE, datetime_index=False
    )
    df.index.name = None
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df


def load_sp500_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the price relatives of 25 assets from the S&P 500 Index.

    This dataset is composed of the daily price relatives of 25 assets from the
    Standard & Poor's 500 (S&P 500) Index from 01/02/1998 to 01/31/2003.

    Price relatives are defined as P_{t+1}/P_t where P_t is the price of an asset
    at time t. These are NOT absolute prices but rather the ratio of consecutive
    price observations.

    The dates in the index start from the initial trading day as indicated in the
    original OLPS paper datasets. Note that dates are indicative only, as the
    actual number of trading days may differ from business day frequency due to
    market closures and holidays.

    The data comes from the Online Portfolio Selection (OLPS) library.

    ==============   ==================
    Observations     1276
    Assets           25
    Region           US
    ==============   ==================

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    net_returns : bool, default=False
        If this is set to True, the price relatives are converted to net returns.
        The default is `False`.

    reverse : bool, default=False
        If this is set to True, the price relatives are reversed.
        The default is `False`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Price relatives DataFrame where each value represents P_{t+1}/P_t

    See Also
    --------
    https://csaws.cs.technion.ac.il/~rani/portfolios/SP500_Dataset.htm

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_relatives_dataset
    >>> relatives = load_sp500_relatives_dataset()
    >>> relatives.head()
                0         1         2  ...         22        23        24
    Date                                      ...
    1998-01-02  1.001234  0.998765  1.002345  ...  0.999876  1.001234  0.998765
    1998-01-05  0.997654  1.003456  0.999123  ...  1.002345  0.998765  1.001234
    1998-01-06  1.004567  0.996789  1.001234  ...  0.997654  1.003456  0.999123
    1998-01-07  0.999876  1.001234  0.998765  ...  1.004567  0.996789  1.001234
    1998-01-08  1.002345  0.998765  1.001234  ...  0.999876  1.001234  0.998765
    """
    data_filename = "sp500_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename, data_module=DATA_MODULE, datetime_index=False
    )
    df.index.name = None
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df


def load_tse_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the price relatives of 88 assets from the TSE Index.

    This dataset is composed of the daily price relatives of 88 assets from the
    Toronto Stock Exchange (TSE) Index from 01/04/1994 to 12/31/1998.

    Price relatives are defined as P_{t+1}/P_t where P_t is the price of an asset
    at time t. These are NOT absolute prices but rather the ratio of consecutive
    price observations.

    The dates in the index start from the initial trading day as indicated in the
    original OLPS paper datasets. Note that dates are indicative only, as the
    actual number of trading days may differ from business day frequency due to
    market closures and holidays.

    The data comes from the Online Portfolio Selection (OLPS) library.

    ==============   ==================
    Observations     1259
    Assets           88
    Region           CA
    ==============   ==================

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    net_returns : bool, default=False
        If this is set to True, the price relatives are converted to net returns.
        The default is `False`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Price relatives DataFrame where each value represents P_{t+1}/P_t

    See Also
    --------
    https://csaws.cs.technion.ac.il/~rani/portfolios/TSE_Dataset.htm

    Examples
    --------
    >>> from skfolio.datasets import load_tse_relatives_dataset
    >>> relatives = load_tse_relatives_dataset()
    >>> relatives.head()
                WESTCOST ENERGY INC   BARRICK GOLD CORP
    1994-01-04  1.001234       ...    0.998765
    1994-01-05  0.997654       ...    1.001234
    1994-01-06  1.004567       ...    0.999123
    1994-01-07  0.999876       ...    1.001234
    1994-01-10  1.002345       ...    0.998765
    ...
    """
    data_filename = "tse_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename,
        data_module=DATA_MODULE,
        datetime_index=False,
    )
    df.index.name = None
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df


def load_cmc20_relatives_dataset(
    data_home: str | Path | None = None,
    net_returns: bool = False,
    reverse: bool = False,
) -> pd.DataFrame:
    """Load the daily price relatives of 20 assets from the CMC20 Index."""
    data_filename = "cmc20_relatives.csv.gz"
    df = load_gzip_compressed_csv_data(
        data_filename, data_module=DATA_MODULE, datetime_index=False
    )
    df.index.name = None
    if reverse:
        df = df.iloc[::-1]
    return df - 1 if net_returns else df
