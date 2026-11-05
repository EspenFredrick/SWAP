# Changelog (rev. 20/10/2026)

## Python Scripts
- Updated `GetEvents.py` to point environment variable for pySPEDAS to the `/Data/Raw/` folder. Got rid of unused import statements. Refactored functions to be less redundant and simpler. 
- In `GetEvents.py`, some functions such as interpolation/backfill/frontfill have been lumped into a new function called `regrid()` in `GetSatellitesGSM.py`.
- A test comparing the new GetEvents file returns the same values.
- In `CalculateUpstream.py`, the `mfi` loop and the `swe` loop do the same thing for different columns, so they became one shared function.


## Folder structures
- Updated `/Data/Raw/` to include `raw_readme.md` to explain how to point python environment variables there. 
- Moved `GetSatellitesGSM.py` to the `/Functions/` folder

## Issues
 - In `CalculateUpstream.py`, the line `bound_dist = np.average(omni['BSN'])` is in R_E, but `l1_dist` and `raw['pos']` are in km. This would subtract about 13 km instead of 13 R_E, meaning we are effectively propagating to Earth’s center, not the bow shock nose. It probably needs * RE. I’m assuming OMNI’s `BSN_x` is in R_E, which is how the old `GetSatellitesGSM.py` treated it.