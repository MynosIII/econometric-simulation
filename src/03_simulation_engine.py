# ============================================================
# SIMULATION ENGINE
# ============================================================


class EconomicSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.turn = 0
        self.landscape = EconomicLandscape(config, self.rng)
        self.asset_market = AssetMarket(config)
        self.credit_market = CreditMarket(config.policy_rate)

        self.firms: List[Firm] = []
        self.banks: List[Bank] = []
        self.funds: List[HedgeFund] = []
        self.next_firm_id = 0

        for _ in range(config.firms):
            self.firms.append(self._new_random_firm())
        for index in range(config.banks):
            self.banks.append(Bank(f"B{index}", index, config.grid_size, self.rng))
        for index in range(config.funds):
            self.funds.append(HedgeFund(f"H{index}", index, config.grid_size, self.rng))

        self.history: Dict[str, List[float]] = {
            "output": [],
            "predicted_output": [],
            "forecast_error": [],
            "asset_price": [],
            "credit": [],
            "gini": [],
            "top10": [],
            "defaults": [],
            "bank_failures": [],
            "mean_rate": [],
            "alive_firms": [],
            "total_wealth": [],
        }
        self.strategy_history: Dict[str, List[int]] = {
            name: [] for name in STRATEGY_CENTROIDS
        }
        self.last_defaulted_agents: List[BaseAgent] = []

    @property
    def agents(self) -> List[BaseAgent]:
        return [*self.firms, *self.banks, *self.funds]

    def _new_random_firm(self) -> Firm:
        size = self.config.grid_size
        firm = Firm(
            agent_id=f"E{self.next_firm_id}",
            position=(int(self.rng.integers(0, size)), int(self.rng.integers(0, size))),
            traits=random_traits(self.rng),
            rng=self.rng,
        )
        self.next_firm_id += 1
        return firm

    def _forecast_next_output(self) -> float:
        output = self.history["output"]
        if not output:
            return 0.0
        if len(output) < 3:
            return output[-1]
        window = np.asarray(output[-6:], dtype=float)
        trend = np.diff(window).mean() if window.size > 1 else 0.0
        return float(max(0.0, window[-1] + trend))

    def _occupancy(self) -> Dict[Tuple[int, int], int]:
        occupancy: Dict[Tuple[int, int], int] = {}
        for firm in self.firms:
            if firm.alive:
                key = tuple(firm.position.tolist())
                occupancy[key] = occupancy.get(key, 0) + 1
        return occupancy

    def _evolve_and_replace(self) -> None:
        if self.turn % self.config.evolution_interval != 0:
            return

        living = [firm for firm in self.firms if firm.alive]
        missing = self.config.firms - len(living)
        if missing <= 0:
            return

        if not living:
            self.firms = [self._new_random_firm() for _ in range(self.config.firms)]
            return

        fitness = np.array(
            [firm.evolutionary_fitness(self.asset_market.price) for firm in living],
            dtype=float,
        )
        fitness = np.maximum(fitness, 1e-6)
        probabilities = fitness / fitness.sum()

        children: List[Firm] = []
        for _ in range(missing):
            parent = living[int(self.rng.choice(len(living), p=probabilities))]
            # Most entrants inherit successful strategies, while a minority are
            # independent entrants. This prevents premature loss of diversity.
            child_traits = (
                random_traits(self.rng)
                if self.rng.random() < 0.24
                else parent.traits.mutated(self.rng)
            )
            row = int(
                np.clip(
                    parent.position[0] + self.rng.integers(-3, 4),
                    0,
                    self.config.grid_size - 1,
                )
            )
            col = int(
                np.clip(
                    parent.position[1] + self.rng.integers(-3, 4),
                    0,
                    self.config.grid_size - 1,
                )
            )

            seed_cash = min(max(parent.cash * 0.12, 0.8), 4.0)
            seed_capital = min(max(parent.productive_capital * 0.08, 1.2), 3.5)
            parent.cash = max(0.0, parent.cash - seed_cash)
            parent.productive_capital = max(0.2, parent.productive_capital - seed_capital)

            child = Firm(
                agent_id=f"E{self.next_firm_id}",
                position=(row, col),
                traits=child_traits,
                rng=self.rng,
                cash=seed_cash + 1.5,
                productive_capital=seed_capital + 1.0,
                generation=parent.generation + 1,
            )
            self.next_firm_id += 1
            children.append(child)

        self.firms = living + children

    def _record_metrics(
        self,
        aggregate_output: float,
        predicted_output: float,
        defaults: int,
        bank_failures: int,
    ) -> None:
        living_firms = [firm for firm in self.firms if firm.alive]
        firm_wealth = [
            firm.net_worth(self.asset_market.price)
            for firm in living_firms
        ]
        total_wealth = sum(
            max(agent.net_worth(self.asset_market.price), 0.0)
            for agent in self.agents
            if agent.alive
        )
        error = abs(aggregate_output - predicted_output) / max(abs(aggregate_output), 1.0)

        self.history["output"].append(float(aggregate_output))
        self.history["predicted_output"].append(float(predicted_output))
        self.history["forecast_error"].append(float(error))
        self.history["asset_price"].append(float(self.asset_market.price))
        self.history["credit"].append(self.credit_market.total_outstanding())
        self.history["gini"].append(gini(firm_wealth))
        self.history["top10"].append(top_share(firm_wealth, 0.10))
        self.history["defaults"].append(float(defaults))
        self.history["bank_failures"].append(float(bank_failures))
        self.history["mean_rate"].append(
            float(np.mean(self.credit_market.new_rates))
            if self.credit_market.new_rates
            else np.nan
        )
        self.history["alive_firms"].append(float(len(living_firms)))
        self.history["total_wealth"].append(float(total_wealth))

        counts = {name: 0 for name in STRATEGY_CENTROIDS}
        for firm in living_firms:
            counts[firm.strategy] += 1
        for name in STRATEGY_CENTROIDS:
            self.strategy_history[name].append(counts[name])

    def step(self) -> None:
        if self.turn >= self.config.turns:
            return

        predicted_output = self._forecast_next_output()
        if self.turn == self.config.structural_break_turn:
            self.landscape.apply_structural_break()

        occupancy = self._occupancy()
        aggregate_output = 0.0

        for firm in [firm for firm in self.firms if firm.alive]:
            aggregate_output += firm.move_and_produce(
                self.landscape,
                occupancy,
                self.rng,
                self.config.movement_cost,
            )
            firm.invest()
            firm.request_credit(self.credit_market, self.asset_market.price)

        for fund in [fund for fund in self.funds if fund.alive]:
            fund.decide(self.asset_market, self.credit_market)

        for bank in [bank for bank in self.banks if bank.alive]:
            bank.funding_cost(self.config.policy_rate)
            bank.offer_credit(self.credit_market, self.config.policy_rate)

        self.credit_market.clear()

        for firm in [firm for firm in self.firms if firm.alive]:
            firm.invest()
        for fund in [fund for fund in self.funds if fund.alive]:
            fund.deploy_borrowed_cash(self.asset_market)

        defaulted = self.credit_market.settle(self.asset_market)

        # Firms can also fail from negative net worth even without a payment event.
        for firm in [firm for firm in self.firms if firm.alive]:
            if firm.net_worth(self.asset_market.price) <= 0.0:
                firm.liquidate(self.asset_market)
                firm.fail()
                defaulted.append(firm)

        bank_failures = 0
        for bank in self.banks:
            if bank.check_solvency(self.asset_market.price):
                bank_failures += 1

        exogenous_shock = -0.16 if self.turn == self.config.structural_break_turn else 0.0
        self.asset_market.close_turn(self.rng, exogenous_shock=exogenous_shock)
        self.landscape.regenerate()

        self.last_defaulted_agents = defaulted
        self.turn += 1
        self._evolve_and_replace()
        self._record_metrics(
            aggregate_output=aggregate_output,
            predicted_output=predicted_output,
            defaults=len(defaulted),
            bank_failures=bank_failures,
        )

    def run(self) -> None:
        while self.turn < self.config.turns:
            self.step()

    def export_metrics_csv(self, path: str) -> None:
        fields = ["turn", *self.history.keys(), *[f"strategy_{name.lower()}" for name in STRATEGY_CENTROIDS]]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(len(self.history["output"])):
                row = {"turn": index + 1}
                for key, values in self.history.items():
                    row[key] = values[index]
                for name, values in self.strategy_history.items():
                    row[f"strategy_{name.lower()}"] = values[index]
                writer.writerow(row)

