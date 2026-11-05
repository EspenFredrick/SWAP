# Description of Scripts

## GetEvents.py
The `GetEvents.py` script downloads all the relevant data that will be used in the study. It takes a command line argument for the input file containing the list of events to analyze. 
Usage:
`python GetEvents.py eventlist.csv`

## CalculateUpstream.py
The `CalculateUpstream.py` script takes the downloaded ACE/Wind data and reconstructs the upstream solar wind profile used by OMNI. OMNI provides the time (in seconds) each point of data was shifted and the satellite from which it came. To reconstruct the upstream profile, this script backtracks from this information to place the corresponding point in a dataframe.
