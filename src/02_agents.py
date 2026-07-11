# ============================================================
# AGENTS
# ============================================================


class BaseAgent:
    def __init__(self, agent_id: str, cash: float):
        self.id = agent_id
        self.cash = float(cash)
        self.securities = 0.0
        self.alive = True
        self.loans_borrowed: List[Loan] = []
        self.loans_lent: List[Loan] = []
        self.failure_turn: Optional[int] = None

    def debt_outstanding(self) -> float:
        return float(
            sum(
                loan.outstanding
                for loan in self.loans_borrowed
                if loan.status == "active"
            )
        )

    def loan_assets(self) -> float:
        return float(
            sum(
                loan.outstanding
                for loan in self.loans_lent
                if loan.status == "active"
            )
        )

    def debt_ratio(self) -> float:
        assets = self.cash + self.loan_assets() + 1.0
        return self.debt_outstanding() / assets

    def scheduled_due(self) -> float:
        return float(
            sum(
                loan.due()[0]
                for loan in self.loans_borrowed
                if loan.status == "active"
            )
        )

    def net_worth(self, asset_price: float) -> float:
        return (
            self.cash
            + self.securities * asset_price
            + self.loan_assets()
            - self.debt_outstanding()
        )

    def liquidation_value(self, asset_price: float) -> float:
        return self.cash + self.securities * asset_price

    def liquidate(self, asset_market: AssetMarket) -> float:
        if self.securities > 0:
            asset_market.sell(self, self.securities)
        pool = max(self.cash, 0.0)
        self.cash = 0.0
        return float(pool)

    def fail(self) -> None:
        self.alive = False
        self.cash = 0.0
        self.securities = 0.0


class Firm(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: Tuple[int, int],
        traits: FirmTraits,
        rng: np.random.Generator,
        cash: float = 9.0,
        productive_capital: float = 6.0,
        generation: int = 0,
    ):
        super().__init__(agent_id, cash=cash)
        self.position = np.array(position, dtype=int)
        self.traits = traits
        self.productive_capital = float(productive_capital)
        self.productivity = float(rng.lognormal(mean=0.0, sigma=0.12))
        self.generation = generation
        self.age = 0
        self.last_profit = 0.0
        self.last_output = 0.0
        self.profit_history: List[float] = []
        self.wealth_at_birth = self.net_worth(100.0)

    @property
    def strategy(self) -> str:
        return classify_strategy(self.traits)

    def net_worth(self, asset_price: float) -> float:
        return (
            super().net_worth(asset_price)
            + 0.72 * self.productive_capital
        )

    def liquidation_value(self, asset_price: float) -> float:
        return super().liquidation_value(asset_price) + 0.34 * self.productive_capital

    def liquidate(self, asset_market: AssetMarket) -> float:
        if self.securities > 0:
            asset_market.sell(self, self.securities)
        pool = max(self.cash, 0.0) + 0.34 * self.productive_capital
        self.cash = 0.0
        self.productive_capital = 0.0
        return float(pool)

    def risk_score(self, asset_price: float = 100.0) -> float:
        leverage = self.debt_outstanding() / max(self.net_worth(asset_price), 1.0)
        profit_penalty = max(0.0, -self.last_profit) / 5.0
        cash_penalty = max(0.0, 2.0 - self.cash) / 2.0
        return float(
            np.clip(
                0.25
                + 0.65 * leverage
                + 0.35 * profit_penalty
                + 0.25 * cash_penalty
                + 0.15 * self.traits.risk_tolerance,
                0.0,
                3.0,
            )
        )

    def move_and_produce(
        self,
        landscape: EconomicLandscape,
        occupancy: Dict[Tuple[int, int], int],
        rng: np.random.Generator,
        movement_cost: float,
    ) -> float:
        if not self.alive:
            return 0.0

        rows, cols = landscape.resources.shape
        moves = [
            np.array([dr, dc], dtype=int)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ]

        current = tuple(self.position.tolist())
        best_move = np.array([0, 0], dtype=int)
        best_score = -np.inf

        for move in moves:
            if rng.random() > self.traits.mobility and np.any(move != 0):
                continue
            nr, nc = self.position + move
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            opportunity = landscape.resources[nr, nc]
            congestion = occupancy.get((int(nr), int(nc)), 0)
            exploration_noise = rng.normal(
                0.0,
                self.traits.exploration * max(np.std(landscape.resources), 0.1),
            )
            switching_penalty = movement_cost if np.any(move != 0) else 0.0
            score = (
                opportunity * self.productivity
                - 0.32 * congestion
                - switching_penalty
                + exploration_noise
            )
            if score > best_score:
                best_score = score
                best_move = move

        self.position += best_move
        new_position = tuple(self.position.tolist())
        if new_position != current:
            self.cash -= movement_cost

        row, col = self.position
        scale = 0.70 + 0.50 * np.sqrt(max(self.productive_capital, 0.0))
        requested = 0.28 * scale
        extracted = landscape.extract(int(row), int(col), requested)

        revenue = extracted * self.productivity * (1.55 + 0.08 * np.log1p(self.productive_capital))
        operating_cost = 0.13 * scale + 0.018 * self.productive_capital
        profit = revenue - operating_cost

        self.cash += profit
        self.productive_capital *= 0.992
        self.last_profit = float(profit)
        self.last_output = float(revenue)
        self.profit_history.append(float(profit))
        self.profit_history = self.profit_history[-12:]
        self.age += 1
        return float(revenue)

    def invest(self) -> None:
        if not self.alive:
            return
        desired_buffer = 2.3 + 0.12 * self.productive_capital
        surplus = max(0.0, self.cash - desired_buffer)
        investment = surplus * self.traits.investment_rate
        self.cash -= investment
        self.productive_capital += investment

    def request_credit(self, market: CreditMarket, asset_price: float) -> None:
        if not self.alive:
            return
        wealth = max(self.net_worth(asset_price), 1.0)
        leverage = self.debt_outstanding() / wealth
        leverage_gap = self.traits.target_leverage - leverage
        if leverage_gap <= 0:
            return

        desired_investment = (
            2.0
            + self.traits.credit_appetite * (2.0 + 0.30 * self.productive_capital)
        )
        funding_gap = max(0.0, desired_investment - max(self.cash - 2.0, 0.0))
        amount = min(18.0, funding_gap + 4.0 * leverage_gap)
        if amount < 1.5:
            return

        max_rate = 0.045 + 0.16 * self.traits.risk_tolerance
        term = int(np.clip(round(7 - 3 * self.traits.risk_tolerance), 3, 8))
        market.register_application(
            borrower=self,
            amount=amount,
            maximum_rate=max_rate,
            term=term,
            purpose="productive investment",
            risk_score=self.risk_score(asset_price),
        )

    def evolutionary_fitness(self, asset_price: float) -> float:
        recent_profit = np.mean(self.profit_history[-5:]) if self.profit_history else 0.0
        wealth = self.net_worth(asset_price)
        survival_bonus = min(self.age, 25) * 0.03
        return float(max(0.01, recent_profit + 0.035 * max(wealth, 0.0) + survival_bonus))


class Bank(BaseAgent):
    def __init__(self, agent_id: str, index: int, grid_size: int, rng: np.random.Generator):
        cash = float(rng.uniform(150.0, 210.0))
        super().__init__(agent_id, cash=cash)
        self.deposits = float(rng.uniform(105.0, 155.0))
        self.liquidity_buffer = float(rng.uniform(0.15, 0.28))
        self.risk_appetite = float(rng.uniform(0.35, 0.85))
        angle = 2.0 * np.pi * index / max(1, 5)
        radius = grid_size * 0.43
        center = (grid_size - 1) / 2.0
        self.map_position = np.array(
            [
                np.clip(center + radius * np.sin(angle), 1, grid_size - 2),
                np.clip(center + radius * np.cos(angle), 1, grid_size - 2),
            ]
        )

    def net_worth(self, asset_price: float) -> float:
        return super().net_worth(asset_price) - self.deposits

    def offer_credit(self, market: CreditMarket, policy_rate: float) -> None:
        if not self.alive:
            return
        required_cash = self.liquidity_buffer * self.deposits
        lendable = max(0.0, self.cash - required_cash)
        lendable *= 0.50 + 0.40 * self.risk_appetite
        if lendable > 2.0:
            minimum_rate = policy_rate + 0.012 + 0.024 * (1.0 - self.risk_appetite)
            market.register_offer(self, lendable, minimum_rate)

    def funding_cost(self, policy_rate: float) -> None:
        if self.alive:
            self.cash -= self.deposits * (policy_rate / 20.0)

    def check_solvency(self, asset_price: float) -> bool:
        if self.alive and self.net_worth(asset_price) < 0.0:
            self.fail()
            return True
        return False

    def liquidation_value(self, asset_price: float) -> float:
        return max(0.0, super().liquidation_value(asset_price) - self.deposits)


class HedgeFund(BaseAgent):
    def __init__(self, agent_id: str, index: int, grid_size: int, rng: np.random.Generator):
        super().__init__(agent_id, cash=float(rng.uniform(32.0, 58.0)))
        self.target_leverage = float(rng.uniform(0.7, 1.8))
        self.risk_tolerance = float(rng.uniform(0.65, 1.0))
        side = index % 4
        offset = 3 + (index * 5) % max(grid_size - 6, 1)
        if side == 0:
            pos = [2, offset]
        elif side == 1:
            pos = [grid_size - 3, offset]
        elif side == 2:
            pos = [offset, 2]
        else:
            pos = [offset, grid_size - 3]
        self.map_position = np.array(pos, dtype=float)

    def decide(
        self,
        market: AssetMarket,
        credit_market: CreditMarket,
    ) -> None:
        if not self.alive:
            return

        asset_value = self.securities * market.price
        equity = max(self.cash + asset_value - self.debt_outstanding(), 0.1)
        leverage = self.debt_outstanding() / equity
        momentum = market.last_return

        scheduled_due = self.scheduled_due()
        liquidity_gap = max(0.0, 1.35 * scheduled_due + 4.0 - self.cash)
        if liquidity_gap > 0 and self.securities > 0:
            shares_needed = liquidity_gap / max(market.price, 1.0)
            market.sell(self, min(self.securities, shares_needed))

        if leverage > self.target_leverage * 1.35 or momentum < -0.055 or self.cash < 1.5:
            market.sell(self, self.securities * (0.25 + 0.25 * self.risk_tolerance))
            return

        if momentum > -0.015:
            investable_cash = max(0.0, self.cash - max(5.0, 1.25 * scheduled_due))
            investment = investable_cash * (0.12 + 0.24 * self.risk_tolerance)
            market.buy(self, investment)

        if leverage < self.target_leverage and momentum > -0.01 and self.cash > 4.0:
            amount = min(15.0, 3.0 + 7.0 * (self.target_leverage - leverage))
            credit_market.register_application(
                borrower=self,
                amount=amount,
                maximum_rate=0.09 + 0.07 * self.risk_tolerance,
                term=5,
                purpose="leveraged asset purchase",
                risk_score=float(np.clip(0.55 + leverage + 0.75 * self.risk_tolerance, 0.0, 3.0)),
            )

    def deploy_borrowed_cash(self, market: AssetMarket) -> None:
        if self.alive and market.last_return > -0.02:
            reserve = max(6.0, 1.35 * self.scheduled_due())
            market.buy(self, max(0.0, self.cash - reserve) * 0.28)

