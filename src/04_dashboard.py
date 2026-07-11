# ============================================================
# VISUAL DASHBOARD
# ============================================================


class EconomicDashboard:
    def __init__(self, simulation: EconomicSimulation):
        self.sim = simulation
        self.fig = plt.figure(figsize=(17, 10))
        grid = GridSpec(
            3,
            4,
            figure=self.fig,
            width_ratios=[1.35, 1.35, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0],
        )
        self.ax_mesh = self.fig.add_subplot(grid[:, :2], projection="3d")
        self.ax_macro = self.fig.add_subplot(grid[0, 2:])
        self.ax_lorenz = self.fig.add_subplot(grid[1, 2])
        self.ax_credit = self.fig.add_subplot(grid[1, 3])
        self.ax_strategy = self.fig.add_subplot(grid[2, 2])
        self.ax_model = self.fig.add_subplot(grid[2, 3])
        self.fig.suptitle("Spatial agent-based economy", fontsize=16)

    def _agent_position(self, agent: BaseAgent) -> Optional[Tuple[float, float, float]]:
        surface = self.sim.landscape.resources
        rows, cols = surface.shape
        if isinstance(agent, Firm):
            row, col = agent.position
            return float(col), float(row), float(surface[row, col] + 0.45)
        if isinstance(agent, (Bank, HedgeFund)):
            row, col = agent.map_position
            r = int(np.clip(round(row), 0, rows - 1))
            c = int(np.clip(round(col), 0, cols - 1))
            lift = 3.0 if isinstance(agent, Bank) else 1.8
            return float(col), float(row), float(surface[r, c] + lift)
        return None

    def draw(self) -> None:
        self._draw_mesh()
        self._draw_macro()
        self._draw_lorenz()
        self._draw_credit()
        self._draw_strategies()
        self._draw_model_error()
        self.fig.tight_layout(rect=[0, 0, 1, 0.965])

    def _draw_mesh(self) -> None:
        ax = self.ax_mesh
        ax.clear()
        z = self.sim.landscape.resources
        rows, cols = z.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))

        ax.plot_surface(
            x,
            y,
            z,
            cmap="terrain",
            rstride=2,
            cstride=2,
            linewidth=0.18,
            edgecolor=(0, 0, 0, 0.20),
            alpha=0.78,
            antialiased=True,
        )

        living_firms = [firm for firm in self.sim.firms if firm.alive]
        for strategy, color in STRATEGY_COLORS.items():
            group = [firm for firm in living_firms if firm.strategy == strategy]
            if not group:
                continue
            xs = [firm.position[1] for firm in group]
            ys = [firm.position[0] for firm in group]
            zs = [z[firm.position[0], firm.position[1]] + 0.5 for firm in group]
            sizes = [
                18.0 + 13.0 * np.log1p(max(firm.net_worth(self.sim.asset_market.price), 0.0))
                for firm in group
            ]
            ax.scatter(xs, ys, zs, s=sizes, c=color, label=strategy, alpha=0.88)

        for bank in [bank for bank in self.sim.banks if bank.alive]:
            pos = self._agent_position(bank)
            if pos:
                ax.scatter(*pos, marker="^", s=150, c="black", depthshade=False)

        for fund in [fund for fund in self.sim.funds if fund.alive]:
            pos = self._agent_position(fund)
            if pos:
                ax.scatter(*pos, marker="s", s=85, c="purple", depthshade=False)

        active_loans = sorted(
            self.sim.credit_market.active_loans,
            key=lambda loan: loan.outstanding,
            reverse=True,
        )[: self.sim.config.max_credit_edges_drawn]
        for loan in active_loans:
            if not loan.borrower.alive:
                continue
            lender_pos = self._agent_position(loan.lender)
            borrower_pos = self._agent_position(loan.borrower)
            if lender_pos and borrower_pos:
                ax.plot(
                    [lender_pos[0], borrower_pos[0]],
                    [lender_pos[1], borrower_pos[1]],
                    [lender_pos[2], borrower_pos[2]],
                    color="black",
                    linewidth=0.35 + 0.035 * np.sqrt(loan.outstanding),
                    alpha=0.18,
                )

        ax.set_xlim(0, cols - 1)
        ax.set_ylim(0, rows - 1)
        ax.set_zlim(0, max(16.0, float(z.max() + 5.0)))
        ax.set_xlabel("Economic space X")
        ax.set_ylabel("Economic space Y")
        ax.set_zlabel("Available opportunity")
        ax.view_init(elev=38, azim=-56)
        ax.set_title(
            f"Turn {self.sim.turn} | firms={sum(f.alive for f in self.sim.firms)} | "
            f"banks={sum(b.alive for b in self.sim.banks)} | funds={sum(h.alive for h in self.sim.funds)}"
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    def _draw_macro(self) -> None:
        ax = self.ax_macro
        ax.clear()
        turns = np.arange(1, len(self.sim.history["output"]) + 1)
        if turns.size:
            ax.plot(turns, safe_normalized(self.sim.history["output"]), label="Real output index")
            ax.plot(turns, safe_normalized(self.sim.history["asset_price"]), label="Asset price index")
            ax.plot(turns, safe_normalized(self.sim.history["credit"]), label="Outstanding credit index")
        ax.axvline(
            self.sim.config.structural_break_turn + 1,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.55,
            label="Structural break",
        )
        ax.set_title("Macro-financial dynamics (first observation = 100)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Index")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8, ncol=2)

    def _draw_lorenz(self) -> None:
        ax = self.ax_lorenz
        ax.clear()
        wealth = np.maximum(
            np.array(
                [
                    firm.net_worth(self.sim.asset_market.price)
                    for firm in self.sim.firms
                    if firm.alive
                ],
                dtype=float,
            ),
            0.0,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        if wealth.size and wealth.sum() > 0:
            ordered = np.sort(wealth)
            cumulative = np.insert(np.cumsum(ordered) / ordered.sum(), 0, 0.0)
            population = np.linspace(0.0, 1.0, cumulative.size)
            ax.plot(population, cumulative, linewidth=2)
        latest_gini = self.sim.history["gini"][-1] if self.sim.history["gini"] else 0.0
        latest_top = self.sim.history["top10"][-1] if self.sim.history["top10"] else 0.0
        ax.set_title(f"Firm wealth | Gini={latest_gini:.2f}, top 10%={latest_top:.0%}")
        ax.set_xlabel("Cumulative firms")
        ax.set_ylabel("Cumulative wealth")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.20)

    def _draw_credit(self) -> None:
        ax = self.ax_credit
        ax.clear()
        outstanding = {name: 0.0 for name in STRATEGY_CENTROIDS}
        outstanding["Hedge funds"] = 0.0
        for loan in self.sim.credit_market.active_loans:
            if isinstance(loan.borrower, Firm):
                outstanding[loan.borrower.strategy] += loan.outstanding
            elif isinstance(loan.borrower, HedgeFund):
                outstanding["Hedge funds"] += loan.outstanding
        names = list(outstanding.keys())
        values = [outstanding[name] for name in names]
        colors = [STRATEGY_COLORS.get(name, "purple") for name in names]
        ax.barh(names, values, color=colors, alpha=0.82)
        mean_rate = self.sim.history["mean_rate"][-1] if self.sim.history["mean_rate"] else np.nan
        rate_text = "n/a" if np.isnan(mean_rate) else f"{mean_rate:.1%}"
        ax.set_title(f"Outstanding credit by strategy | new rate={rate_text}")
        ax.set_xlabel("Principal outstanding")
        ax.grid(axis="x", alpha=0.20)

    def _draw_strategies(self) -> None:
        ax = self.ax_strategy
        ax.clear()
        counts = {
            name: sum(
                1
                for firm in self.sim.firms
                if firm.alive and firm.strategy == name
            )
            for name in STRATEGY_CENTROIDS
        }
        names = list(counts.keys())
        values = [counts[name] for name in names]
        colors = [STRATEGY_COLORS[name] for name in names]
        ax.bar(names, values, color=colors, alpha=0.84)
        ax.set_title("Current evolutionary strategy mix")
        ax.set_ylabel("Living firms")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.20)

    def _draw_model_error(self) -> None:
        ax = self.ax_model
        ax.clear()
        actual = self.sim.history["output"]
        predicted = self.sim.history["predicted_output"]
        if actual:
            start = max(0, len(actual) - 45)
            turns = np.arange(start + 1, len(actual) + 1)
            ax.plot(turns, actual[start:], label="Actual output")
            ax.plot(turns, predicted[start:], linestyle="--", label="Naive forecast")
            latest_error = self.sim.history["forecast_error"][-1]
        else:
            latest_error = 0.0
        ax.axvline(
            self.sim.config.structural_break_turn + 1,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.45,
        )
        ax.set_title(f"Model limitation | latest relative error={latest_error:.1%}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Output")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)

    def animate(self, interval_ms: int = 160) -> FuncAnimation:
        def update(_frame: int):
            self.sim.step()
            self.draw()
            return []

        animation = FuncAnimation(
            self.fig,
            update,
            frames=self.sim.config.turns,
            interval=interval_ms,
            repeat=False,
            blit=False,
        )
        return animation

