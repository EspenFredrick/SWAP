import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta
import subprocess
import numpy as np
import pandas as pd
from dateutil.parser import isoparse
import shutil
from pathlib import Path

from multiprocessing import Pool

# Function that prints what is going on in the terminal, so you know that the program isn't hanging
# It just prints the message you put in (msg) followed by a new line and 52 tildes (~)
def say(msg):
    print(f'{msg}\n{"~" * 52}')

# Set absolute working directories
SWAP_dir = Path(__file__).resolve().parent
BATS_dir = SWAP_dir.parent / "BATSRUS"

# Command line arguments
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
args = parser.parse_args()

# Function that we will multiprocess
def create_rundir(date_start):
    date_string = date_start.strftime('%Y-%m-%d')
    #say(f"Creating PARAM.in for {date_string}")

    with open(f"{SWAP_dir}/PARAM_TEMP.in", "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines[12] = f"{date_start.year}                    iYear.\n"
    lines[13] = f"{date_start.month}                       iMonth.\n"
    lines[14] = f"{date_start.day}                       iDay.\n"
    lines[15] = f"{date_start.hour}                       iHour.\n"
    lines[16] = f"{date_start.minute}                       iMinute.\n"
    lines[17] = f"{date_start.second}                       iSecond.\n"
    lines[56] = f"{date_string}.dat                 NameSolarWindFile"
    lines[73] = f"{date_string}_sat.dat          NameTrajectoryFile   (GSM, RE; set y=z=0 in 1D)"

    with open(f"{SWAP_dir}/PARAM.in", "w", encoding="utf-8") as file:
        file.writelines(lines)

    rundir_path = f"{BATS_dir}/run_{date_string}"
    #print(f"Moving PARAM.in to {rundir_path}")
    if os.path.exists(rundir_path):
        shutil.rmtree(rundir_path)
    command = f"make rundir RUNDIR={rundir_path}"
    subprocess.run(command, shell=True, cwd=BATS_dir, check=True)

    shutil.move(f"{SWAP_dir}/PARAM.in", f"{BATS_dir}/run_{date_string}/PARAM.in")
    shutil.copy(f"{SWAP_dir}/Data/Upstream/dat/{date_string}.dat", f"{BATS_dir}/run_{date_string}/{date_string}.dat")
    # shutil.copy(f"../SWAP/Data")
    #say("Done.")

if __name__ == "__main__":
    # Get list of events as a dataframe and then export all the datetimes from the first column to date_objs
    event_list = pd.read_csv(args.input, delimiter=',', header=0, parse_dates=['Date_Start'])
    date_objs = event_list['Date_Start']

    with Pool() as pool:
        pool.map(create_rundir, date_objs)





'''
# Setting up the event list using arguments
event_list = pd.read_csv(args.input, delimiter=',', header=0)
for e in range(len(event_list)):

    date_start = pd.to_datetime(event_list['Date_Start'][e])
    date_string = date_start.strftime('%Y-%m-%d')
    say(f"Creating PARAM.in for {date_string}")

    #The following section reads every line in PARAM.in and saves each line as a list of strings in the variable lines.
    #It then modifies certain lines to the parameter we need
    #And finally write them in
    with open(f"{SWAP_dir}/PARAM_TEMP.in", "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines[12] = f"{date_start.year}                    iYear.\n"
    lines[13] = f"{date_start.month}                       iMonth.\n"
    lines[14] = f"{date_start.day}                       iDay.\n"
    lines[15] = f"{date_start.hour}                       iHour.\n"
    lines[16] = f"{date_start.minute}                       iMinute.\n"
    lines[17] = f"{date_start.second}                       iSecond.\n"
    lines[56] = f"{date_string}.dat                 NameSolarWindFile"
    lines[73] = f"{date_string}_sat.dat          NameTrajectoryFile   (GSM, RE; set y=z=0 in 1D)"

    with open(f"{SWAP_dir}/PARAM.in", "w", encoding="utf-8") as file:
        file.writelines(lines)

    #os.chdir(r"../BATSRUS")
    #pwd = os.getcwd()

    rundir_path = f"{BATS_dir}/run_{date_string}"
    print(f"Moving PARAM.in to {rundir_path}")
    if os.path.exists(rundir_path):
        shutil.rmtree(rundir_path)
    command = f"make rundir RUNDIR={rundir_path}"
    subprocess.run(command, shell=True, cwd=BATS_dir, check=True)

    shutil.move(f"{SWAP_dir}/PARAM.in", f"{BATS_dir}/run_{date_string}/PARAM.in")
    shutil.copy(f"{SWAP_dir}/Data/Upstream/dat/{date_string}.dat", f"{BATS_dir}/run_{date_string}/{date_string}.dat")
    #shutil.copy(f"../SWAP/Data")
    say("Done.")
'''