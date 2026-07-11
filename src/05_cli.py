# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial agent-based economic simulation with a 3D mesh dashboard."
    )
    parser.add_argument("--turns", type=int, default=180)
    parser.add_argument("--firms", type=int, default=90)
    parser.add_argument("--banks", type=int, default=5)
    parser.add_argument("--funds", type=int, default=8)
    parser.add_argument("--grid", type=int, default=46)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=int, default=160, help="Animation delay in milliseconds.")
    parser.add_argument("--no-animation", action="store_true", help="Run all turns and show only the final dashboard.")
    parser.add_argument("--snapshot", type=str, default="", help="Save the final dashboard as PNG.")
    parser.add_argument("--save-gif", type=str, default="", help="Save the animation as a GIF (requires Pillow).")
    parser.add_argument("--export-csv", type=str, default="", help="Export turn-by-turn metrics as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        seed=args.seed,
        grid_size=args.grid,
        turns=args.turns,
        firms=args.firms,
        banks=args.banks,
        funds=args.funds,
        structural_break_turn=max(5, int(args.turns * 0.53)),
    )
    simulation = EconomicSimulation(config)
    dashboard = EconomicDashboard(simulation)

    if args.no_animation:
        simulation.run()
        dashboard.draw()
        if args.snapshot:
            dashboard.fig.savefig(args.snapshot, dpi=170, bbox_inches="tight")
        if args.export_csv:
            simulation.export_metrics_csv(args.export_csv)
        if not args.snapshot:
            plt.show()
        return

    animation = dashboard.animate(interval_ms=args.interval)
    if args.save_gif:
        animation.save(args.save_gif, writer="pillow", fps=max(1, int(1000 / args.interval)))
        if args.export_csv:
            simulation.export_metrics_csv(args.export_csv)
    else:
        plt.show()


if __name__ == "__main__":
    main()
