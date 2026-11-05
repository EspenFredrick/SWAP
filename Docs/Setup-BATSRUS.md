# Setup

BATS-R-US Setup for Ideal 1D MHD
---
To specify 1D ideal MHD, please run these configuration settings from the root BATSRUS directory:
```bash
./Config.pl -e=Mhd -u=Default -g=8,1,1 -ng=2   # ideal MHD, no user module, 8×1×1-cell blocks, 2 ghost cells
./Config.pl -s                                 # confirm the configuration
make -j                                        # builds src/BATSRUS.exe
make PIDL                                      # post-processor for the IDL plot files
```
Make a run directory for each event. Give it an absolute path:
```bash
make rundir RUNDIR=$PWD/run_yyyymmdd
```
This creates BATSRUS.exe (a link), the `IO2/ output` folder, `restartIN/`, `restartOUT/`, and the scripts `PostProc.pl` and `pIDL`.

Next, put four input files in that directory:
- `PARAM.in`: replace the default one with mine.
- `IMF.dat`: your reconstructed L1 series. The order on each line is yr mn dy hr min sec msec bx by bz vx vy vz n T, in nT, km/s, cm⁻³ and K:
```
  Reconstructed L1 (ACE/Wind), Bx set to event mean
  #COOR
  GSM

  #START
   2015  1  7  0  0  0  0   2.10  -3.1  -3.7  -400.0  0.0  0.0  5.3  1.0E+05
   2015  1  7  0  1  0  0   2.10  -3.2  -3.6  -405.0  0.0  0.0  5.4  1.0E+05
```
Write v_x as negative (sunward is +x) and keep B_x constant. The manual caps the file at 50,000 lines.
`artemis_p1.dat` and `artemis_p2.dat`: each probe's trajectory, with x in R_E and y = z = 0 for the 1D run:
```
  #COOR
  GSM

  #START
   2015  1  7  0  0  0  0   55.2  0.0  0.0
   2015  1  7  0  1  0  0   55.2  0.0  0.0
```
Check the parameter file from the `BATSRUS/` directory:
```bash
share/Scripts/CheckParam.pl run_20150107/PARAM.in
```
`CheckParam.pl` catches typos, missing files and wrong parameter types. Alternatively, `share/Scripts/ParamEditor.pl run_20150107/PARAM.in` opens the file in a browser editor with the manual built in.

Now run it:
```bash
cd run_yyyymmdd
mpiexec -np 4 ./BATSRUS.exe |& tee runlog    # or ./BATSRUS.exe |& tee runlog for a serial build
```
The 1D grid has 128 blocks, so 1–4 processes is plenty. The run is finished when runlog ends with the timing table.

Post-process:
```bash
./PostProc.pl -cat
```
The satellite output lands in `IO2/sat_artemis_p1_*.sat`. It is plain text: one header line, a line of variable names, then the data. You can read it with `spacepy.pybats.bats.VirtSat` or with `pandas read_csv(..., sep=r'\s+', skiprows=1)`. The 1D x-profiles become `.out` files, which `spacepy.pybats.IdlFile` reads.
