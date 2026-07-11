from __future__ import annotations

"""
Economic Mesh V2 — Private Baseline
===================================
A spatial agent-based economy with:
- households, labor, consumption, energy and subsistence activity;
- firms producing differentiated goods on a regenerating 3D opportunity mesh;
- privately funded banks, endogenous market rates and risk-priced credit;
- annual/periodic interest conversion and several amortization conventions;
- private creditor workouts, debt-equity swaps, fire-sale auctions and liquidation waterfalls;
- equity investment in firms by patient-capital, growth, momentum and
  leveraged-extraction funds;
- credit/equity network analysis with NetworkX;
- an animated Matplotlib dashboard and CSV export.

This is an experimental laissez-faire baseline, not a calibrated forecast. It intentionally contains no central bank, public bailout, deposit insurance, automatic bank replacement or statutory household discharge.
"""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

try:  # Optional: the model works without QuantLib.
    import QuantLib as ql  # type: ignore

    HAS_QUANTLIB = True
except Exception:
    ql = None
    HAS_QUANTLIB = False


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class SimulationConfig:
    seed: int = 42
    turns: int = 180
    grid_size: int = 42
    firms: int = 55
    households: int = 180
    banks: int = 4
    funds: int = 10

    periods_per_year: int = 12  # one turn = one month
    initial_market_rate_annual: float = 0.045
    private_time_preference_annual: float = 0.018
    initial_inflation_annual: float = 0.025

    resource_regeneration: float = 0.09
    structural_break_turn: int = 96
    structural_break_severity: float = 0.43
    movement_cost: float = 0.025

    household_metabolism: float = 0.100
    work_energy_cost: float = 0.105
    subsistence_energy_cost: float = 0.075
    energy_per_good: float = 0.27
    minimum_energy: float = 0.02
    base_wage: float = 1.15

    bankruptcy_missed_payment_threshold: int = 4
    distress_missed_payment_threshold: int = 2
    liquidation_discount_capital: float = 0.38
    liquidation_discount_inventory: float = 0.48
    liquidation_admin_cost: float = 0.08

    equity_market_depth: float = 260.0
    equity_price_impact: float = 0.035
    max_network_edges_drawn: int = 34

    private_spin_off_interval: int = 8
    private_spin_off_cash: float = 5.0
    private_spin_off_capital: float = 2.5


# ============================================================
# FINANCIAL MATHEMATICS
# ============================================================


class RateMath:
    """Interest conversion inspired by QuantLib's explicit convention model."""

    @staticmethod
    def nominal_to_effective_annual(nominal_rate: float, compounds_per_year: int) -> float:
        m = max(1, int(compounds_per_year))
        return float((1.0 + nominal_rate / m) ** m - 1.0)

    @staticmethod
    def effective_annual_to_nominal(effective_rate: float, compounds_per_year: int) -> float:
        m = max(1, int(compounds_per_year))
        return float(m * ((1.0 + effective_rate) ** (1.0 / m) - 1.0))

    @staticmethod
    def nominal_to_periodic(
        nominal_rate: float,
        nominal_compounds_per_year: int,
        payments_per_year: int,
    ) -> float:
        effective = RateMath.nominal_to_effective_annual(
            nominal_rate, nominal_compounds_per_year
        )
        p = max(1, int(payments_per_year))
        return float((1.0 + effective) ** (1.0 / p) - 1.0)

    @staticmethod
    def equivalent_nominal(
        nominal_rate: float,
        from_compounds: int,
        to_compounds: int,
    ) -> float:
        effective = RateMath.nominal_to_effective_annual(nominal_rate, from_compounds)
        return RateMath.effective_annual_to_nominal(effective, to_compounds)

    @staticmethod
    def annuity_payment(principal: float, periodic_rate: float, periods: int) -> float:
        n = max(1, int(periods))
        if abs(periodic_rate) < 1e-12:
            return float(principal / n)
        denominator = 1.0 - (1.0 + periodic_rate) ** (-n)
        return float(principal * periodic_rate / max(denominator, 1e-12))

    @staticmethod
    def quantlib_equivalent_rate_if_available(
        nominal_rate: float,
        from_compounds: int,
        to_compounds: int,
    ) -> float:
        """Uses QuantLib when installed; otherwise returns the internal equivalent."""
        if not HAS_QUANTLIB:
            return RateMath.equivalent_nominal(
                nominal_rate, from_compounds, to_compounds
            )
        frequency_map = {
            1: ql.Annual,
            2: ql.Semiannual,
            4: ql.Quarterly,
            12: ql.Monthly,
        }
        from_frequency = frequency_map.get(from_compounds, ql.Monthly)
        to_frequency = frequency_map.get(to_compounds, ql.Monthly)
        source = ql.InterestRate(
            nominal_rate,
            ql.Actual365Fixed(),
            ql.Compounded,
            from_frequency,
        )
        return float(
            source.equivalentRate(
                ql.Compounded,
                to_frequency,
                1.0,
            ).rate()
        )


# ============================================================
# GENERAL HELPERS
# ============================================================


def gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 1e-12:
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * array) / (n * total)) - (n + 1.0) / n)


def top_share(values: Iterable[float], fraction: float = 0.10) -> float:
    array = np.maximum(np.asarray(list(values), dtype=float), 0.0)
    if array.size == 0 or array.sum() <= 1e-12:
        return 0.0
    count = max(1, int(math.ceil(array.size * fraction)))
    return float(np.sort(array)[-count:].sum() / array.sum())


def normalized_index(values: Sequence[float]) -> np.ndarray:
    if not values:
        return np.array([])
    array = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.abs(array) > 1e-9)
    base = array[valid[0]] if valid.size else 1.0
    return 100.0 * array / base


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ============================================================
# ECONOMIC LANDSCAPE
# ============================================================


class EconomicLandscape:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        size = config.grid_size
        large = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 7.5)
        medium = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 19.0)
        small = gaussian_filter(rng.normal(size=(size, size)), sigma=max(1.0, size / 50.0))

        raw = 0.60 * large / (np.std(large) + 1e-9)
        raw += 0.29 * medium / (np.std(medium) + 1e-9)
        raw += 0.11 * small / (np.std(small) + 1e-9)
        raw -= raw.min()
        raw /= raw.max() + 1e-9

        y_variance = np.linspace(0.78, 1.25, size)[:, None]
        local_cluster = gaussian_filter(rng.uniform(0.75, 1.25, size=(size, size)), sigma=6)
        local_cluster /= np.mean(local_cluster)

        self.capacity = (2.8 + 10.5 * raw) * y_variance * local_cluster
        self.resources = self.capacity.copy()
        self.regeneration_rate = config.resource_regeneration
        self.structural_break_severity = float(np.clip(config.structural_break_severity, 0.0, 0.95))
        self.break_applied = False

    def opportunity(self, row: int, col: int) -> float:
        return float(self.resources[row, col])

    def extract(self, row: int, col: int, requested: float) -> float:
        available = self.resources[row, col]
        amount = min(max(requested, 0.0), available)
        self.resources[row, col] -= amount
        return float(amount)

    def regenerate(self) -> None:
        carrying = np.maximum(self.capacity, 0.1)
        self.resources += (
            self.regeneration_rate
            * self.resources
            * (1.0 - self.resources / carrying)
        )
        self.resources = np.clip(self.resources, 0.02, carrying)

    def structural_break(self) -> None:
        if self.break_applied:
            return
        rows, cols = self.capacity.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        center_r, center_c = rows * 0.63, cols * 0.38
        radius_sq = (yy - center_r) ** 2 + (xx - center_c) ** 2
        regional_shock = 1.0 - self.structural_break_severity * np.exp(
            -radius_sq / (2.0 * (rows * 0.19) ** 2)
        )
        self.capacity *= regional_shock
        self.resources = np.minimum(self.resources * regional_shock, self.capacity)
        self.regeneration_rate *= 0.74
        self.break_applied = True


# ============================================================
