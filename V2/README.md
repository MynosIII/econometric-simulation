# Economic Mesh V2 — Private Baseline

Economic Mesh V2 is a spatial agent-based economic simulation written in Python. It extends the first version with households, labor, consumption, energy, private banking, endogenous credit conditions, bankruptcy, equity investment, fire sales, and financial contagion.

The model is intentionally a **private-sector baseline**. It contains no central bank, lender of last resort, public bailout, deposit insurance, automatic bank replacement, unemployment insurance, or other public stabilization mechanism. This allows future versions to compare the same economy with and without monetary or fiscal intervention.

## Main components

- A regenerating 3D economic-opportunity mesh.
- Households that work, earn wages, consume goods, lose energy each turn, and may become indebted.
- Firms that hire labor, produce inventories, invest, borrow, repay, reorganize, or liquidate.
- Banks that create deposits when lending, price credit risk, face withdrawals, and can fail.
- Several loan structures and explicit annual-to-periodic interest-rate conversion.
- Private creditor workouts, debt-equity swaps, collateral recovery, and liquidation waterfalls.
- Funds using patient-capital, growth, momentum, and leveraged-extraction strategies.
- Direct equity investment between agents and a credit/equity network analyzed with NetworkX.
- An animated Matplotlib dashboard, GIF export, snapshots, and CSV metrics.

## Install

From the `V2` folder:

```bash
python -m pip install -r requirements.txt
```

QuantLib is optional. The simulation runs without it, but it can be installed to compare interest-rate conventions:

```bash
python -m pip install QuantLib
```

## Run

Interactive animation:

```bash
python economic_mesh_v2_private.py
```

Create a GIF:

```bash
python economic_mesh_v2_private.py --turns 200 --interval 180 --save-gif economic_mesh.gif
```

Create a final image and export the metrics:

```bash
python economic_mesh_v2_private.py --turns 200 --no-animation --snapshot final_state.png --export-csv metrics.csv
```

A more severe private crisis can be tested with:

```bash
python economic_mesh_v2_private.py --turns 200 --market-rate 0.12 --shock-severity 0.82 --save-gif crisis.gif
```

## Interpretation

The mesh is a metaphorical economic state space rather than literal geography. Simulation outcomes depend on the model's assumptions, parameters, random seed, contracts, and behavioral rules. A single run should not be interpreted as an empirical forecast or as proof that one strategy is universally superior.
