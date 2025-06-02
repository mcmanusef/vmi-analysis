# vmi-analysis

vmi-analysis provides tools and example pipelines for analyzing data from Velocity Map Imaging (VMI) experiments. It includes utilities for converting detector output to usable `h5` files, running analysis pipelines and a small GUI application for data acquisition.

## Features

- Modular pipeline framework for reading, processing and saving data
- Conversion utilities for TPX3 detector files
- Example analysis scripts and pipelines
- Simple acquisition GUI built on `tkinter`

## Requirements

The project targets Python 3.13 and relies on `numpy`, `pandas`, `numba`, `plotly` and other common scientific packages. Install the dependencies via pip using the provided `pyproject.toml`.

```bash
pip install -e .
```

## Usage

The `Examples/` directory contains sample pipelines demonstrating how to process and analyze VMI data. To run the UConn example UI:

```bash
python Examples/uconn_vmi/uconn_ui.py
```

This starts a simple interface with options to run data conversion or acquisition pipelines. The pipelines themselves can also be used programmatically by importing from `vmi_analysis.processing`.

## Repository Layout

- `vmi_analysis/` – core modules for building pipelines and processes
- `Examples/` – example scripts and notebook-based analysis
- `Analysis/` – assorted notebooks for data exploration
- `Testing/` – benchmark utilities (currently minimal)

## Development

After cloning the repository install the dependencies as above. Code formatting is enforced with `ruff` and static typing is checked by `pyright`.

```bash
pip install -e .[dev]
ruff .
pyright
```

## License

This project is provided as-is under an open license. See individual files for any specific terms.

