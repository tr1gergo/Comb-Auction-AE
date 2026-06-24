# Comb-Auction-AE

Research code for combinatorial-auction experiments and manipulation analyses.

This repository contains simulation code and notebooks for two instance families:
- AIRPORT: airport time-slot auctions
- GRID: spatial grid point auctions

Quick start
- Requirements: Python 3.8+ and common packages: numpy, pandas, matplotlib, seaborn, jupyter.
- Open and run the notebooks `AIRPORT_simulations.ipynb` and `GRID_simulations.ipynb` with Jupyter or JupyterLab.

Files & folders
- `AIRPORT_generation.py`, `AIRPORT_heuristic.py` — airport instance generator and heuristics.
- `GRID_generation.py`, `GRID_heuristic.py` — grid instance generator and heuristics.
- `AIRPORT_simulations.ipynb`, `GRID_simulations.ipynb` — primary experiment notebooks.
- `airport_data/`, `grid_data/` — saved CSV/NPY experiment outputs (load/save paths in notebooks).
- `airport_figs/`, `GRID_figs/` — generated figures.

Usage
- Run experiments interactively via the notebooks. 
- Re-run cells or adapt parameters at the top of each notebook to reproduce or extend experiments.

Notes
- The repo is research-oriented (not packaged). Use a virtual environment and install required packages before running.
