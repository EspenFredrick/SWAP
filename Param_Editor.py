import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta
import subprocess
import numpy as np
import pandas as pd
from dateutil.parser import isoparse
import shutil

# Command line arguments
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
args = parser.parse_args()


# Setting up the event list using arguments
event_list = pd.read_csv(args.input, delimiter=',', header=0)
for e in range(len(event_list)):
    date_start = pd.to_datetime(event_list['Date_Start'][e])
    date_string = date_start.strftime('%Y-%m-%d')

        #The following section reads every line in PARAM.in and saves each line as a list of strings in the variable lines.
    #It then modifies certain lines to the parameter we need
    #And finally write them in
    with open("PARAM_TEMP.in", "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines[12] = f"{date_start.year}                    iYear.\n"
    lines[13] = f"{date_start.month}                       iMonth.\n"
    lines[14] = f"{date_start.day}                       iDay.\n"
    lines[15] = f"{date_start.hour}                       iHour.\n"
    lines[16] = f"{date_start.minute}                       iMinute.\n"
    lines[17] = f"{date_start.second}                       iSecond.\n"
    lines[56] = f"{date_string}.dat                 NameSolarWindFile"
    lines[73] = f"{date_string}_sat.dat          NameTrajectoryFile   (GSM, RE; set y=z=0 in 1D)"

    with open(f"PARAM.in", "w", encoding="utf-8") as file:
        file.writelines(lines)
    os.chdir(r"../BATSRUS")
    pwd = os.getcwd()
    rundir_path = f"{pwd}/run_{date_string}"
    command = f"make rundir RUNDIR={rundir_path}"
    subprocess.run(command, shell=True)

    shutil.move("../SWAP/PARAM.in", f"run_{date_string}/PARAM.in")
    shutil.copy(f"../SWAP/Data/Upstream/dat/{date_string}.dat", f"run_{date_string}/{date_string}.dat")
    #shutil.copy(f"../SWAP/Data")
    os.chdir(r"..")