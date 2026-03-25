# SWAP-tools: Solar Wind Advection & Prediction

[![Language](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Field](https://img.shields.io/badge/Field-Space_Physics-orange.svg)](https://wikipedia.org/wiki/Space_physics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**SWAP** (Solar Wind Advection & Prediction) is a Python package designed to streamline the comparison of propagation methods for solar wind plasma from L1 (Lagrangian point 1) to the Earth's **Bow Shock Nose**.

## Background

Accurately predicting when a solar wind disturbance will hit the Earth's magnetosphere is a cornerstone of Space Weather research. However, the Interplanetary Magnetic Field (IMF) may change during the transit from the L1 Lagrange point to Earth.

**SWAP_tools** allows researchers to:
1. Apply different advection algorithms to upstream solar wind (Simple, Ballistic, or MHD).
2. Compare these methods against "in-situ" observations from ARTEMIS.

---

## Features

* **Propagator Comparison:** Compare standard ballistic propagation ($t = d/v$) against more nuanced MHD methods.
* **Coordinate Transformations:** Easy handling of GSE (Geocentric Solar Ecliptic) and GSM (Geocentric Solar Magnetospheric) coordinate systems.
* **Time-Series Analysis:** Tools to resample and shift L1 data (ACE/DSCOVR/Wind) to the subsolar point.

---

## Installation

Clone the repository and install via `pip`:

```bash
git clone [https://github.com/yourusername/SWAP_tools.git](https://github.com/yourusername/SWAP_tools.git)
cd SWAP_tools
pip install -e .
