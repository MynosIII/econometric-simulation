from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class SimulationConfig:
    seed: int = 42
    grid_size: int = 46
    turns: int = 180
    firms: int = 90
    banks: int = 5
    funds: int = 8

    regeneration_rate: float = 0.075
    movement_cost: float = 0.05
    policy_rate: float = 0.025
    evolution_interval: int = 8
    structural_break_turn: int = 95
    asset_liquidity: float = 180.0
    asset_price_impact: float = 0.08
    exogenous_volatility: float = 0.012
    max_credit_edges_drawn: int = 35


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def gini(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * total)) - (n + 1.0) / n)


def top_share(values: Iterable[float], fraction: float = 0.10) -> float:
    x = np.maximum(np.asarray(list(values), dtype=float), 0.0)
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    count = max(1, int(np.ceil(x.size * fraction)))
    return float(np.sort(x)[-count:].sum() / x.sum())


def safe_normalized(series: List[float]) -> np.ndarray:
    if not series:
        return np.array([])
    arr = np.asarray(series, dtype=float)
    base = arr[0] if abs(arr[0]) > 1e-9 else 1.0
    return 100.0 * arr / base


# ============================================================
# ECONOMIC LANDSCAPE
# ============================================================


class EconomicLandscape:
    """Spatial productive capacity and temporarily available resources."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        size = config.grid_size
        large = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 7.0)
        medium = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 18.0)
        small = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 45.0)

        raw = 0.58 * large / (np.std(large) + 1e-9)
        raw += 0.30 * medium / (np.std(medium) + 1e-9)
        raw += 0.12 * small / (np.std(small) + 1e-9)

        raw -= raw.min()
        raw /= raw.max() + 1e-9

        y_gradient = np.linspace(0.82, 1.18, size)[:, None]
        capacity = (3.0 + 10.0 * raw) * y_gradient

        self.capacity = capacity
        self.resources = capacity.copy()
        self.regeneration_rate = config.regeneration_rate
        self.break_applied = False

    def value_at(self, row: int, col: int) -> float:
        return float(self.resources[row, col])

    def extract(self, row: int, col: int, requested: float) -> float:
        amount = min(max(requested, 0.0), self.resources[row, col])
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

    def apply_structural_break(self) -> None:
        if self.break_applied:
            return
        rows, cols = self.capacity.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        center_r, center_c = rows * 0.60, cols * 0.42
        radius = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
        regional_shock = 1.0 - 0.38 * np.exp(-(radius ** 2) / (2.0 * (rows * 0.20) ** 2))
        self.capacity *= regional_shock
        self.resources = np.minimum(self.resources * regional_shock, self.capacity)
        self.regeneration_rate *= 0.72
        self.break_applied = True


# ============================================================
# AGENT STRATEGIES
# ============================================================


@dataclass
class FirmTraits:
    exploration: float
    credit_appetite: float
    target_leverage: float
    investment_rate: float
    risk_tolerance: float
    mobility: float

    def mutated(self, rng: np.random.Generator, scale: float = 0.08) -> "FirmTraits":
        values = np.array([
            self.exploration,
            self.credit_appetite,
            self.target_leverage,
            self.investment_rate,
            self.risk_tolerance,
            self.mobility,
        ])
        values += rng.normal(0.0, scale, size=values.size)
        lower = np.array([0.02, 0.05, 0.05, 0.08, 0.02, 0.10])
        upper = np.array([1.60, 1.50, 2.20, 0.95, 1.00, 1.00])
        values = np.clip(values, lower, upper)
        return FirmTraits(*values.tolist())


STRATEGY_CENTROIDS: Dict[str, np.ndarray] = {
    "Conservative": np.array([0.16, 0.22, 0.25, 0.32, 0.18, 0.35]),
    "Balanced": np.array([0.42, 0.55, 0.65, 0.52, 0.45, 0.55]),
    "Growth": np.array([0.70, 0.90, 1.10, 0.73, 0.68, 0.72]),
    "Speculative": np.array([1.05, 1.30, 1.75, 0.88, 0.90, 0.86]),
}

STRATEGY_COLORS = {
    "Conservative": "tab:blue",
    "Balanced": "tab:green",
    "Growth": "tab:orange",
    "Speculative": "tab:red",
}


def classify_strategy(traits: FirmTraits) -> str:
    vector = np.array([
        traits.exploration,
        traits.credit_appetite,
        traits.target_leverage,
        traits.investment_rate,
        traits.risk_tolerance,
        traits.mobility,
    ])
    return min(
        STRATEGY_CENTROIDS,
        key=lambda name: float(np.linalg.norm(vector - STRATEGY_CENTROIDS[name])),
    )


def random_traits(rng: np.random.Generator) -> FirmTraits:
    name = rng.choice(list(STRATEGY_CENTROIDS.keys()))
    base = STRATEGY_CENTROIDS[str(name)]
    jitter = rng.normal(0.0, 0.07, size=base.size)
    values = base + jitter
    values = np.clip(
        values,
        [0.02, 0.05, 0.05, 0.08, 0.02, 0.10],
        [1.60, 1.50, 2.20, 0.95, 1.00, 1.00],
    )
    return FirmTraits(*values.tolist())


# ============================================================
# CREDIT AND ASSET MARKETS
# ============================================================


@dataclass
class Loan:
    loan_id: int
    lender: "BaseAgent"
    borrower: "BaseAgent"
    principal: float
    outstanding: float
    rate: float
    term: int
    remaining: int
    purpose: str
    status: str = "active"

    def due(self) -> Tuple[float, float, float]:
        # The quoted rate is the total simple rate for the whole loan term,
        # not a per-turn rate. This keeps the time scale economically coherent.
        principal_due = self.outstanding / max(self.remaining, 1)
        interest_due = self.principal * self.rate / max(self.term, 1)
        return principal_due + interest_due, principal_due, interest_due


@dataclass
class CreditOffer:
    lender: "BaseAgent"
    amount: float
    minimum_rate: float


@dataclass
class CreditApplication:
    borrower: "BaseAgent"
    amount: float
    maximum_rate: float
    term: int
    purpose: str
    risk_score: float


class AssetMarket:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.price = 100.0
        self.history = [self.price]
        self.net_order = 0.0
        self.last_return = 0.0

    def buy(self, agent: "BaseAgent", cash_amount: float) -> float:
        amount = min(max(cash_amount, 0.0), max(agent.cash, 0.0))
        if amount <= 0:
            return 0.0
        shares = amount / self.price
        agent.cash -= amount
        agent.securities += shares
        self.net_order += shares
        return float(shares)

    def sell(self, agent: "BaseAgent", shares: float) -> float:
        quantity = min(max(shares, 0.0), max(agent.securities, 0.0))
        if quantity <= 0:
            return 0.0
        proceeds = quantity * self.price
        agent.securities -= quantity
        agent.cash += proceeds
        self.net_order -= quantity
        return float(proceeds)

    def close_turn(self, rng: np.random.Generator, exogenous_shock: float = 0.0) -> None:
        order_component = (
            self.config.asset_price_impact
            * self.net_order
            / max(self.config.asset_liquidity, 1.0)
        )
        noise = rng.normal(0.0, self.config.exogenous_volatility)
        mean_reversion = 0.002 * np.log(100.0 / max(self.price, 1.0))
        log_return = order_component + noise + mean_reversion + exogenous_shock
        log_return = float(np.clip(log_return, -0.32, 0.25))
        old_price = self.price
        self.price = max(1.0, self.price * np.exp(log_return))
        self.last_return = self.price / old_price - 1.0
        self.history.append(self.price)
        self.net_order = 0.0


class CreditMarket:
    def __init__(self, policy_rate: float):
        self.policy_rate = policy_rate
        self.offers: List[CreditOffer] = []
        self.applications: List[CreditApplication] = []
        self.loans: List[Loan] = []
        self.next_loan_id = 1
        self.new_rates: List[float] = []
        self.default_loss_last_turn = 0.0

    @property
    def active_loans(self) -> List[Loan]:
        return [loan for loan in self.loans if loan.status == "active"]

    def register_offer(self, lender: "BaseAgent", amount: float, minimum_rate: float) -> None:
        if lender.alive and amount > 0:
            self.offers.append(CreditOffer(lender, amount, minimum_rate))

    def register_application(
        self,
        borrower: "BaseAgent",
        amount: float,
        maximum_rate: float,
        term: int,
        purpose: str,
        risk_score: float,
    ) -> None:
        if not borrower.alive or amount <= 0:
            return
        if any(app.borrower is borrower for app in self.applications):
            return
        self.applications.append(
            CreditApplication(
                borrower=borrower,
                amount=amount,
                maximum_rate=maximum_rate,
                term=term,
                purpose=purpose,
                risk_score=float(np.clip(risk_score, 0.0, 3.0)),
            )
        )

    def clear(self) -> List[Loan]:
        self.new_rates = []
        originated: List[Loan] = []
        self.offers.sort(key=lambda offer: offer.minimum_rate)
        self.applications.sort(
            key=lambda app: (app.maximum_rate, app.risk_score), reverse=True
        )

        for application in self.applications:
            for offer in self.offers:
                lender = offer.lender
                if not lender.alive or not application.borrower.alive:
                    continue
                if offer.amount < application.amount or lender.cash < application.amount:
                    continue

                quoted_rate = (
                    offer.minimum_rate
                    + 0.018 * application.risk_score
                    + 0.008 * application.borrower.debt_ratio()
                )
                if quoted_rate > application.maximum_rate:
                    continue

                lender.cash -= application.amount
                application.borrower.cash += application.amount
                offer.amount -= application.amount

                loan = Loan(
                    loan_id=self.next_loan_id,
                    lender=lender,
                    borrower=application.borrower,
                    principal=application.amount,
                    outstanding=application.amount,
                    rate=quoted_rate,
                    term=application.term,
                    remaining=application.term,
                    purpose=application.purpose,
                )
                self.next_loan_id += 1
                self.loans.append(loan)
                lender.loans_lent.append(loan)
                application.borrower.loans_borrowed.append(loan)
                originated.append(loan)
                self.new_rates.append(quoted_rate)
                break

        self.offers.clear()
        self.applications.clear()
        return originated

    def settle(self, asset_market: AssetMarket) -> List["BaseAgent"]:
        self.default_loss_last_turn = 0.0
        defaulted: List[BaseAgent] = []

        borrowers: Dict[int, BaseAgent] = {}
        borrower_loans: Dict[int, List[Loan]] = {}
        for loan in self.active_loans:
            if not loan.borrower.alive:
                loan.status = "defaulted"
                continue
            key = id(loan.borrower)
            borrowers[key] = loan.borrower
            borrower_loans.setdefault(key, []).append(loan)

        for key, loans in borrower_loans.items():
            borrower = borrowers[key]
            dues = [loan.due() for loan in loans]
            total_due = sum(item[0] for item in dues)

            if borrower.cash + 1e-9 >= total_due:
                for loan, (_, principal_due, interest_due) in zip(loans, dues):
                    payment = principal_due + interest_due
                    borrower.cash -= payment
                    if loan.lender.alive:
                        loan.lender.cash += payment
                    loan.outstanding = max(0.0, loan.outstanding - principal_due)
                    loan.remaining -= 1
                    if loan.remaining <= 0 or loan.outstanding <= 1e-7:
                        loan.status = "repaid"
            else:
                liquidation_pool = borrower.liquidate(asset_market)
                total_claim = sum(loan.outstanding for loan in loans)
                recovery_ratio = min(1.0, liquidation_pool / max(total_claim, 1e-9))

                for loan in loans:
                    recovery = loan.outstanding * recovery_ratio
                    if loan.lender.alive:
                        loan.lender.cash += recovery
                    loss = loan.outstanding - recovery
                    self.default_loss_last_turn += loss
                    loan.outstanding = 0.0
                    loan.remaining = 0
                    loan.status = "defaulted"

                borrower.fail()
                defaulted.append(borrower)

        return defaulted

    def total_outstanding(self) -> float:
        return float(sum(loan.outstanding for loan in self.active_loans))

