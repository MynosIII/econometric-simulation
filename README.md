# Economic Mesh Simulation

This is a spatial agent-based economic model. It is intended as an experimental laboratory, not as a calibrated forecast of a real economy.

## What is represented

- The 3D mesh is an economic opportunity landscape. Height represents currently available productive opportunity, demand, resources, or market capacity.
- Firms move across the landscape, produce, invest, borrow, repay, default, and reproduce through an evolutionary selection mechanism.
- Firm strategies are continuous parameter combinations classified for visualization as Conservative, Balanced, Growth, or Speculative.
- Banks supply credit subject to liquidity buffers, price borrower risk, receive repayments, and can become insolvent.
- Hedge funds borrow, buy financial assets, deleverage, and can trigger fire sales.
- The credit network is drawn as lines between lenders and borrowers.
- Inequality is shown through the Lorenz curve, Gini coefficient, and top-10-percent wealth share.
- A hidden structural break changes regional productive capacity and asset prices.
- A naive output forecast is compared with realized output to visualize model error under regime change.

## Install

```bash
pip install -r requirements.txt
```

## Run the interactive animation

```bash
python economic_mesh_simulation.py
```

A smaller and faster run:

```bash
python economic_mesh_simulation.py --turns 100 --firms 60 --grid 36
```

## Save a final dashboard

Linux/macOS:

```bash
python economic_mesh_simulation.py \
  --turns 180 \
  --no-animation \
  --snapshot final_economy.png \
  --export-csv metrics.csv
```

Windows PowerShell:

```powershell
py economic_mesh_simulation.py `
  --turns 180 `
  --no-animation `
  --snapshot final_economy.png `
  --export-csv metrics.csv
```

## Save an animated GIF

Linux/macOS:

```bash
python economic_mesh_simulation.py \
  --turns 100 \
  --interval 180 \
  --save-gif economic_evolution.gif \
  --export-csv metrics.csv
```

Windows PowerShell:

```powershell
py economic_mesh_simulation.py `
  --turns 100 `
  --interval 180 `
  --save-gif economic_evolution.gif `
  --export-csv metrics.csv
```

## Project structure

The executable loads the source sections in `src/` into one shared simulation namespace. They are separated only to keep the model readable and easy to extend:

- `01_core_and_markets.py`: configuration, landscape, strategies, credit and asset markets.
- `02_agents.py`: firms, banks and hedge funds.
- `03_simulation_engine.py`: turn sequencing, evolution, shocks and metrics.
- `04_dashboard.py`: animated 3D mesh and analytical panels.
- `05_cli.py`: command-line interface.

## Main modeling choices to edit

At the top of `src/01_core_and_markets.py`, `SimulationConfig` controls population size, the policy rate, resource regeneration, the structural-break turn, market liquidity, volatility, and price impact.

`FirmTraits` contains the evolvable strategy variables:

- exploration;
- appetite for credit;
- target leverage;
- investment rate;
- risk tolerance;
- spatial mobility.

`Loan.due()` treats the quoted interest rate as the total simple rate over the complete loan term. Changing that convention has a very large effect on defaults and should be done explicitly.

## Important limitations

The mesh is a metaphorical state space, not geographic land. The model currently omits households, labor markets, wages, consumption, government, taxes, inflation, money creation, interbank funding, bankruptcy courts, expectations learned from data, and empirical calibration. Strategy survival therefore reflects the rules of this artificial economy, not a universal economic law.

A useful next step would be to separate the model into scenario files and estimate some parameters from historical or experimental data. Multiple random seeds should always be compared; a single run is only one possible trajectory.
