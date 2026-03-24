# SWAP_tools: Solar Wind Advection & Prediction

[![Language](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Field](https://img.shields.io/badge/Field-Space_Physics-orange.svg)](https://wikipedia.org/wiki/Space_physics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**SWAP** (Solar Wind Advection & Prediction) is a Python package designed to streamline the comparison of propagation methods for solar wind plasma from L1 (Lagrangian point 1) to the Earth's **Bow Shock Nose**.

## Background

Accurately predicting when a solar wind disturbance will hit the Earth's magnetosphere is a cornerstone of Space Weather research. However, the "nose" of the bow shock is a moving target that depends on upstream dynamic pressure ($D_p$) and the Interplanetary Magnetic Field (IMF). 

**SWAP_tools** allows researchers to:
1. Calculate the dynamic position of the Bow Shock.
2. Apply different advection algorithms (Simple, Ballistic, or Phase-Front).
3. Compare these methods against "ground truth" observations.

---

## Features

* **Propagator Comparison:** Compare standard ballistic propagation ($t = d/v$) against more nuanced phase-front alignment methods.
* **Bow Shock Models:** Built-in support for classic models including:
    * *Farris & Russell (1994)*
    * *Shue et al. (1997/1998)*
* **Coordinate Transformations:** Easy handling of GSE (Geocentric Solar Ecliptic) and GSM (Geocentric Solar Magnetospheric) coordinate systems.
* **Time-Series Analysis:** Tools to resample and shift L1 data (ACE/DSCOVR/Wind) to the subsolar point.

---

## Installation

Clone the repository and install via `pip`:

```bash
git clone [https://github.com/yourusername/SWAP_tools.git](https://github.com/yourusername/SWAP_tools.git)
cd SWAP_tools
pip install -e .
