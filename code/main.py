"""
This file executes Utah tax revenue estimation from a data center of a given
size.

Functions:
   get_avg_eff_proptax_rates_ut_cnty()
   get_taxbase_ut_cnty()
   get_dc_fmv()
   get_annual_elec()
   get_annual_ngas()
   gen_series_realprop_exp()
   gen_series_tpp_exp()
   gen_series_elec_ann_exp()
   gen_series_ngas_ann_exp()
   gen_series_realprop_txbl_val()
   gen_series_tpp_txbl_val()
   gen_series_cnty_vars()
"""

# Import packages
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
import fig_MWtoFMV as figfmv

# Set directory and file paths
main_dir = Path(__file__).resolve().parent.parent
data_dir = os.path.join(main_dir, "data")
images_dir = os.path.join(main_dir, "images")

"""
Set parameters and ranges

sales_tax_rate_state: State rate portion of state sales taxes
sales_tax_rate_medicaid: Medicaid rate portion of state sales taxes
sales_tax_rate_muni: Municipal rate portion of state sales taxes
sales_tax_rate_cnty: County rate portion of state sales taxes
sales_tax_rate_othlcl: Other local rate portion of state sales taxes
energy_excise_tax rate: Energy excise tax rate
dc_peak_elec_capac: Data center peak eletrical capacity (megawatts, MW)
tpp_val_pct_of_fmv: Tangible personal property value as a percent of total fair
    market value (FMV)
real_val_pct_of_fmv: Real property value as a percent of total fair
    market value (FMV), equals 1 - tpp_val_pct_of_fmv
avg_eff_proptax_rate: Average effective property tax rate for a county or a
    particular area
num_yrs_tpp_replace: Number of years after tangible personal property
    investment in which the data center fully replaces its tangible personal
    property
avg_elec_util_rate: Average electricity utilization rate of a data center
elec_commercial_rate: Electricity cost in $/kWh
ngas_Dth_per_day_per_MW: Natural gas dekatherms per day per megawatt
    (Dth/(MW * day))
ngas_commodity_rate: Commodity cost of natural gas ($/Dth)
ngas_transport_rate: Tranportation cost of natural gas ($/Dth)
cnty_taxbase: Value of the assessed county tax base in dollars
tpp_deprec_sched: ?
realprop_deprec_sched: ?
tpp_top_depr_rate:
realprop_TIF_pct: ?
tpp_TIF_pct: ?
"""
sales_tax_rate_state = 0.0470
sales_tax_rate_medicaid = 0.0015
sales_tax_rate_muni = 0.0100
sales_tax_rate_cnty = 0.0025
sales_tax_rate_othlcl = 0.0055
sales_tax_rate_combined = (
    sales_tax_rate_state + sales_tax_rate_medicaid + sales_tax_rate_muni +
    sales_tax_rate_cnty + sales_tax_rate_othlcl
)

energy_excise_tax_rate = 0.060
energy_excise_tax_rate_min = 0.055
energy_excise_tax_rate_max = 0.070

dc_peak_elec_capac = 100
dc_peak_elec_capac_min = 0.9
dc_peak_elec_capac_max = 1_200

tpp_val_pct_of_fmv = 0.80
tpp_val_pct_of_fmv_min = 0.75
tpp_val_pct_of_fmv_max = 0.90
real_val_pct_of_fmv = 1.0 - tpp_val_pct_of_fmv

avg_eff_proptax_rate = 0.01
avg_eff_proptax_rate_min = 0.0043
avg_eff_proptax_rate_max = 0.0151

num_yrs_tpp_replace = 3
num_yrs_tpp_replace_min = 2
num_yrs_tpp_replace_max = 6

avg_elec_util_rate = 0.75
avg_elec_util_rate_min = 0.70
avg_elec_util_rate_max = 0.90

elec_commercial_rate = 0.065
elec_commercial_rate_min = 0.050
elec_commercial_rate_max = 0.070

ngas_Dth_per_day_per_MW = 250
ngas_Dth_per_day_per_MW_min = 150
ngas_Dth_per_day_per_MW_max = 300

ngas_commodity_rate = 2.89
ngas_commodity_rate_min = 2.50
ngas_commodity_rate_max = 2.95

ngas_transport_rate = 0.5328
ngas_transport_rate_min = 0.10
ngas_transport_rate_max = 0.60

cnty_taxbase = 20_000_000_000
cnty_taxbase_min = 200_000_000
cnty_taxbase_max = 250_000_000_000

tpp_deprec_sched = np.array([1.00, 0.62, 0.46, 0.21, 0.09, 0.07])
realprop_deprec_sched = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])

tpp_top_depr_rate = 1.00
tpp_top_depr_rate_min = 0.80
tpp_top_depr_rate_min = 1.00

realprop_TIF_pct = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
)
realprop_TIF_pct_min = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
)
realprop_TIF_pct_max = np.array(
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
)

tpp_TIF_pct = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
)
tpp_TIF_pct_min = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
)
tpp_TIF_pct_max = np.array(
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
)

"""
Define functions
"""


def get_avg_eff_proptax_rates_ut_cnty(data_dir=data_dir):
    """
    Get a Pandas DataFrame of the average effective property tax rate for each
    county in Utah.

    Args:
        data_dir (string): OS independent path to the data directory

    Returns:
        df_avg_eff_proptax_rates_ut_cnty (DataFrame, 29 x 2): Average effective
            total property tax rates by county in Utah
    """
    df_avg_eff_proptax_rates_ut_cnty = pd.read_csv(
        os.path.join(data_dir, "avg_proptax_rate_by_cnty_ut_2025.csv"),
        header=0
    )

    return df_avg_eff_proptax_rates_ut_cnty


def get_taxbase_ut_cnty(data_dir=data_dir):
    """
    Get a Pandas DataFrame of the measures of the tax base for each county in
    Utah.

    Args:
        data_dir (string): OS independent path to the data directory

    Returns:
        df_taxbbase_ut_cnty (DataFrame, 30 x 8): Measures of the Utah tax base
            by county broken down by locally assessed real property, locally
            assessed personal property, centrally assessed properties, and the
            total
    """
    df_taxbase_ut_cnty = pd.read_csv(
        os.path.join(data_dir, "utah_taxbase_2025_table_1.csv"),
        header=0,
        names=[
            "county", "tot_local_assess_real_prop",
            "pct_local_assess_real_prop", "total_local_assess_pers_prop",
            "pct_local_assess_pers_prop", "total_central_assess_prop",
            "pct_central_assess_prop", "total_local_and_central_assess_prop"
        ],
        nrows=30
    )

    return df_taxbase_ut_cnty


def get_dc_fmv(elec_capac, data_dir=data_dir):
    """
    Calculate data center fair market value ($, FMV). This estimated
    relationship between peak electrical capacity of a data center and its fair
    market value is described in the appendix of "Introducing
    DataCenterAtlas.org"
    (https://econosseur.rickecon.com/p/introducing-datacenteratlas). See Figure
    3 and Table 1 in that article and the equations that follow. The
    func_params objects are estimated in the fig_MWtoFMV.py module

    Args:
        elec_capac (float or array_like): Peak electrical capacity of data
            center (MW)

    Returns:
        fmv (float or np.ndarray): Fair market value of data center ($),
            a float if elec_capac is a scalar, otherwise an array with the
            same shape as elec_capac
    """
    elec_capac_arr = np.asarray(elec_capac, dtype=float)

    func_params_dict = pickle.load(
        open(os.path.join(data_dir, "func_params_dict.pkl"), "rb")
    )

    slope = func_params_dict["slope"]
    intercept = func_params_dict["intercept"]
    exp_slope = func_params_dict["exp_slope"]
    exp_intercept = func_params_dict["exp_intercept"]
    constant = func_params_dict["constant"]

    fmv = np.where(
        elec_capac_arr >= 95,
        slope * elec_capac_arr + intercept,
        np.exp(exp_slope * elec_capac_arr + exp_intercept) + constant,
    ) * 1e9

    if fmv.ndim == 0:
        fmv = float(fmv)

    return fmv


def get_annual_elec(elec_capac, avg_elec_util_rate=avg_elec_util_rate):
    """
    Calculate the annual electricity use (TWh/year) of a data center of a given
    peak electrical capacity and an assumed average electrical utilization
    rate.

    Args:
        elec_capac (float): Peak electrical capacity of data center (MW)
        avg_elec_util_rate (float in [0.7,0.9]): average electrical utilization
            rate

    Returns:
        elec_annual (float): Annual electricity use (TWh/year)
    """
    elec_annual = (elec_capac * avg_elec_util_rate * 365 * 24) / 1e6

    return elec_annual


def get_annual_ngas(
    elec_capac,
    ngas_Dth_per_day_per_MW=ngas_Dth_per_day_per_MW,
    avg_elec_util_rate=avg_elec_util_rate
):
    """
    Calculate the annual natural gas use (dekatherms/year or Dth/year) of a
    data center of a given peak electrical capacity, natural gas dekatherms per
    day per megawatt, and an assumed average electrical utilization rate.

    Args:
        elec_capac (float): Peak electrical capacity of data center (MW)
        ngas_Dth_per_day_per_MW (float): Natural gas decatherms per day per
            megawatt
        avg_elec_util_rate (float in [0.7,0.9]): average electrical utilization
            rate

    Returns:
        ngas_annual (float): Annual natural gas use (Dth/year)
    """
    ngas_annual = (
        elec_capac * ngas_Dth_per_day_per_MW * avg_elec_util_rate * 365
    )

    return ngas_annual


def gen_series_realprop_exp(
    num_yrs_to_frcst=25,
    elec_capac=dc_peak_elec_capac,
    real_val_pct_of_fmv=real_val_pct_of_fmv
):
    """
    Generate time series of real property expenditures
    """
    fmv = get_dc_fmv(elec_capac)
    real_prop_val = real_val_pct_of_fmv * fmv
    series_realprop_exp = np.zeros(num_yrs_to_frcst)
    series_realprop_exp[1] = real_prop_val

    return series_realprop_exp


def gen_series_tpp_exp(
    num_yrs_to_frcst=25,
    elec_capac=dc_peak_elec_capac,
    tpp_val_pct_of_fmv=tpp_val_pct_of_fmv,
    num_yrs_tpp_replace=num_yrs_tpp_replace
):
    """
    Generate time series of tangible personal property expenditures
    """
    fmv = get_dc_fmv(elec_capac)
    tpp_prop_val_init = tpp_val_pct_of_fmv * fmv
    series_tpp_exp = np.zeros(num_yrs_to_frcst)
    series_tpp_exp[1::num_yrs_tpp_replace] = tpp_prop_val_init

    return series_tpp_exp


def gen_series_elec_ann_exp(
    elec_capac,
    avg_elec_util_rate,
    elec_commercial_rate=elec_commercial_rate,
    num_yrs_to_frcst=25
):
    """
    Generate time series of annual electricity expenditure
    """
    elec_annual = get_annual_elec(elec_capac, avg_elec_util_rate)  # TWh/year
    elec_ann_exp = elec_annual * elec_commercial_rate * 1e9  # $
    series_elec_ann_exp = elec_ann_exp * np.ones(num_yrs_to_frcst)
    series_elec_ann_exp[0] = 0.0

    return series_elec_ann_exp


def gen_series_ngas_ann_exp(
    elec_capac,
    ngas_Dth_per_day_per_MW,
    avg_elec_util_rate,
    ngas_commodity_rate=ngas_commodity_rate,
    ngas_transport_rate=ngas_transport_rate,
    num_yrs_to_frcst=25
):
    """
    Generate time series of annual natural gas expenditure
    """
    ngas_annual = get_annual_ngas(
        elec_capac,
        ngas_Dth_per_day_per_MW=ngas_Dth_per_day_per_MW,
        avg_elec_util_rate=avg_elec_util_rate
    )
    ngas_ann_exp = ngas_annual * (ngas_commodity_rate + ngas_transport_rate)
    series_ngas_ann_exp = ngas_ann_exp * np.ones(num_yrs_to_frcst)
    series_ngas_ann_exp[0] = 0.0

    return series_ngas_ann_exp


def gen_series_realprop_txbl_val(
    elec_capac,
    real_val_pct_of_fmv,
    realprop_TIF_pct,
    realprop_deprec_sched=realprop_deprec_sched,
    num_yrs_to_frcst=25
):
    """
    Generate time series of real property taxable value
    """
    # Resize realprop_TIF_pct vector
    realprop_TIF_pct_resize = np.zeros(num_yrs_to_frcst)
    realprop_TIF_pct_resize[:len(realprop_TIF_pct)] = realprop_TIF_pct
    realprop_TIF_pct_resize[len(realprop_TIF_pct):] = realprop_TIF_pct[-1]
    fmv = get_dc_fmv(elec_capac)
    series_realprop_txbl_val = (
        (1 - realprop_TIF_pct_resize) * real_val_pct_of_fmv * fmv *
        np.resize(realprop_deprec_sched, num_yrs_to_frcst)
    )
    series_realprop_txbl_val[0] = 0.0

    return series_realprop_txbl_val


def gen_series_tpp_txbl_val(
    elec_capac,
    tpp_val_pct_of_fmv,
    tpp_TIF_pct,
    num_yrs_tpp_replace,
    tpp_top_depr_rate,
    tpp_deprec_sched=tpp_deprec_sched,
    num_yrs_to_frcst=25
):
    """
    Generate time series of real property taxable value
    """
    tpp_TIF_pct_resize = np.zeros(num_yrs_to_frcst - 1)
    tpp_TIF_pct_resize[:len(tpp_TIF_pct)] = tpp_TIF_pct
    tpp_TIF_pct_resize[len(tpp_TIF_pct):] = tpp_TIF_pct[-1]
    fmv = get_dc_fmv(elec_capac)
    series_tpp_txbl_val = np.zeros(num_yrs_to_frcst)
    tpp_deprec_sched_adj = tpp_deprec_sched[:num_yrs_tpp_replace]
    tpp_deprec_sched_adj[0] = tpp_top_depr_rate
    series_tpp_txbl_val[1:] = (
        (1 - tpp_TIF_pct_resize) * tpp_val_pct_of_fmv * fmv *
        np.resize(tpp_deprec_sched_adj, num_yrs_to_frcst - 1)
    )

    return series_tpp_txbl_val


def gen_series_cnty_vars(
    elec_capac,
    real_val_pct_of_fmv,
    realprop_TIF_pct,
    tpp_val_pct_of_fmv,
    tpp_TIF_pct,
    num_yrs_tpp_replace,
    tpp_top_depr_rate,
    cnty_taxbase,
    avg_eff_proptax_rate,
    start_year = 2027,
    num_yrs_to_frcst=25
):
    """
    Generate time series of county tax base
    """
    year_vec = np.arange(start_year, start_year + num_yrs_to_frcst)
    series_realprop_txbl_val = gen_series_realprop_txbl_val(
        elec_capac,
        real_val_pct_of_fmv,
        realprop_TIF_pct,
        num_yrs_to_frcst=num_yrs_to_frcst
    )
    series_tpp_txbl_val = gen_series_tpp_txbl_val(
        elec_capac,
        tpp_val_pct_of_fmv,
        tpp_TIF_pct,
        num_yrs_tpp_replace,
        tpp_top_depr_rate,
        num_yrs_to_frcst=num_yrs_to_frcst
    )
    series_realprop_exp = gen_series_realprop_exp(
        num_yrs_to_frcst=num_yrs_to_frcst,
        elec_capac=elec_capac,
        real_val_pct_of_fmv=real_val_pct_of_fmv
    )
    series_tpp_exp = gen_series_tpp_exp(
        num_yrs_to_frcst=num_yrs_to_frcst,
        elec_capac=elec_capac,
        tpp_val_pct_of_fmv=tpp_val_pct_of_fmv,
        num_yrs_tpp_replace=num_yrs_tpp_replace
    )
    series_realprop_TIF_pct = np.zeros(num_yrs_to_frcst)
    series_realprop_TIF_pct[:len(realprop_TIF_pct)] = realprop_TIF_pct
    series_realprop_TIF_pct[len(realprop_TIF_pct):] = realprop_TIF_pct[-1]
    series_tpp_TIF_pct = np.zeros(num_yrs_to_frcst)
    series_tpp_TIF_pct[1:len(tpp_TIF_pct) + 1] = tpp_TIF_pct
    series_tpp_TIF_pct[len(tpp_TIF_pct) + 1:] = tpp_TIF_pct[-1]
    series_cnty_new_growth = (
        series_realprop_exp * (1 - series_realprop_TIF_pct)
    )
    series_cnty_taxbase = np.zeros(num_yrs_to_frcst)
    series_cnty_avg_eff_proptax_rate = np.zeros(num_yrs_to_frcst)
    series_cnty_new_growth_rev = np.zeros(num_yrs_to_frcst)
    series_cnty_tot_proptax_rev = np.zeros(num_yrs_to_frcst)
    series_tot_realprop_taxrev = np.zeros(num_yrs_to_frcst)
    for period in range(num_yrs_to_frcst):
        if period == 0:
            # Assume that the prior year tax revenue in the initial year is the
            # current year tax revenue
            cnty_prior_yr_proptax_rev = cnty_taxbase * avg_eff_proptax_rate
            series_cnty_taxbase[period] = cnty_taxbase
            series_cnty_avg_eff_proptax_rate[period] = avg_eff_proptax_rate
        else:
            cnty_prior_yr_proptax_rev = (
                series_cnty_tot_proptax_rev[period - 1]
            )
            series_cnty_taxbase[period] = (
                series_cnty_taxbase[period - 1] +
                series_realprop_txbl_val[period] +
                series_tpp_txbl_val[period] - series_cnty_new_growth[period]
            )
            series_cnty_avg_eff_proptax_rate[period] = (
                cnty_prior_yr_proptax_rev / series_cnty_taxbase[period]
            )
        series_cnty_new_growth_rev[period] = (
            series_cnty_avg_eff_proptax_rate[period] *
            series_cnty_new_growth[period]
        )
        series_cnty_tot_proptax_rev[period] = (
            cnty_prior_yr_proptax_rev + series_cnty_new_growth_rev[period]
        )
        series_tot_realprop_taxrev[period] = (
            series_realprop_txbl_val[period] *
            series_cnty_avg_eff_proptax_rate[period]
        )
    series_dict = {
        "series_realprop_txbl_val": series_realprop_txbl_val,
        "series_tpp_txbl_val": series_tpp_txbl_val,
        "series_realprop_exp": series_realprop_exp,
        "series_tpp_exp": series_tpp_exp,
        "series_realprop_TIF_pct": series_realprop_TIF_pct,
        "series_tpp_TIF_pct": series_tpp_TIF_pct,
        "series_cnty_taxbase": series_cnty_taxbase,
        "series_cnty_new_growth": series_cnty_new_growth,
        "series_cnty_avg_eff_proptax_rate": series_cnty_avg_eff_proptax_rate,
        "series_cnty_new_growth_rev": series_cnty_new_growth_rev,
        "series_cnty_tot_proptax_rev": series_cnty_tot_proptax_rev,
        "series_tot_realprop_taxrev": series_tot_realprop_taxrev,
        "year_vec": year_vec
    }

    return series_dict
