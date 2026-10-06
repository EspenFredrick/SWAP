# Data Folder

Folder containing all data files for simulations/scripts
---

* **CDAWeb** contains downloaded data for ACE/Wind/OMNI/ARTEMIS/etc.
* **Interpolated** contains all the CDAWeb data interpolated to cadence. OMNI resolution is 1-minute, others are 1-second used in the upstream reconstruction of the solar wind.
* **Upstream** contains the originating solar wind reconstructed from ACE/Wind used in the creation of OMNI data. More information on this process is available at (insert link here)
* **Downstream** contains all the computed and propagated solar wind data at Earth using various methods
* **Correlations** contains the correlation coefficients for hour intervals between ARTEMIS and the propagated solar wind in /Downstream/ from various methods as discussed.
* **Raw** is the data from the function that creates the unified time series from ACE/Wind (please investigate later)
* **Downstream Merged** How is this different from downstream? Investigate in file later.
