# SuPerPoV: Score and evolution of polar vortices via persistent homology

The goal of this project is to provide a constant-free method to study the classification and evolution of polar vortices. The details can be found in the paper "SuPerPoV: Score and evolution of polar vortices via
persistent homology", accessible at https://arxiv.org/pdf/2601.11386.

This repo contains a python script and two folders containing scirpts written by Jake Cordes. These scripts can be used to replicate the experiments from the above paper, but, more importantly, can be used on new data for future research. The data used in the experiments are available on Zenodo, https://doi.org/10.5281/zenodo.17903131

__SuPerPoV3.1.0.py__: This is the main file of the project. Enter a given date, number of days before and a number of days after to be computed. The user will also be asked the following:
* What pressures to compute.
* What pressure to plot.
* The resolution for the data (how sparse the data is).
* If the user wants to have reversal of wind speed plotted.
* If the user wants all numeric values printed to the terminal.
* Folder path to the data.
The output will be 5 plots: 2 showing the shape of the vortex for the entered date and pressure to plot. The remaining 3 plots will describe the height, split scores, and displacement scores over the days entered.<br>
<!--SuPerPoV_hPa_1.0.1.py: Enter a given date, number of days before and a number of days after. The output will be 5 plots: 3 showing the heights for the vortices (one for 10 hPa, one for 50 hPa, and one for 100 hPa), and the remaining 2 plots will describe the split scores, and displacement scores over the days entered, for 10 hPa, 50 hPa, and 100 hPa.<br>-->

OldVersions:<br>
This folder contains two scripts similar to the most updated version __SuPerPoV3.1.0.py__.
There is a YouTube video describing the files in more detail, which can be found here: (https://www.youtube.com/watch?v=Vg8sULzG_Fw)

Pre-Computed:<br>
This folder contains two python scripts and a .txt file, and it is used for the comparison with previous method and for additional statistics, but it is not needed to compute the SuPerPov scores.<br>
proper_plot.py: This script outputs a scatter plot, comparing our method to previous methods. To run, Sp_Disp_Def_Sign_date.txt must be in the same folder.<br>
yearly_ratios_plot,py: Enter a given year, and this will output the displacement scores and split scores for an entire year. To run, Sp_Disp_Def_Sign_date.txt must be in the same folder.<br>
Sp_Disp_Def_Sign_date.txt: A giant array that has information for every winter day from 1960-2022, whose goal is to perform statistal analysis (such as average, median, and comparison) quickly.

# Usage:
**Requirements**
* Python 3.10 with NumPy, SciPy, pandas, and GUDHI. The tested versions are in [benchmarks/requirements.txt](benchmarks/requirements.txt).
* The graphical application also needs Matplotlib, geopy, and Tkinter.

**Installation**
* Download or clone this repo.
* Ensure Python and NumPy are installed on your system.
* To use the data set from the paper:
  + Download the dataset from Zenodo
  + Unzip the downloaded data files. The script will ask for the path to the data files you unzip, so copy and paste it once you unzip it.
* To use your own data:
  + The script defaults to the geopotential height data having 46 latititudes (the data in the paper is all even latitudes from 0 to 90N) and 180 longitudes (the data in the paper is all even longitudes from 0 to 358). It can handle resolutions perfectly fine however, just enter in whatever the resolution is when prompted.
  + If you want to use wind speed data, the file must be named "u1060.txt" and the there must be two columns in the wind speed data: one titled DATES and one titled SPEEDS.

**Running the Code**
From a command line, navigate to the directory containing the scripts (and data) and run:
```bash
python SuPerPoV3.1.0.py
```

## Faster daily calculations on the CPU

The application shares a `DailyAnalyzer` across plotting, dates, and pressure
levels. It reads and validates each winter once, indexes dates directly, and
reuses the original triangulations. Daily persistence and scores are recomputed
from the requested field each time.

An optional C++ routine speeds up persistence on these planar triangulations.
Build it once using a C++11 compiler on Linux or macOS; no GPU is needed:

```bash
python -m pip install -r benchmarks/requirements.txt
python superpov_planar.py --build
python superpov_compute.py --file gph_winter_1965.txt \
  --dates 1965-12-23 1966-04-30 --pressure 10
```

The default `--method auto` selects the compiled routine when available and
otherwise uses GUDHI. Importing or running the analyzer never launches a
compiler. `--method planar` explicitly requires the build; `--method direct`
uses GUDHI ordinary persistence; `--method extended` retains the original
extended-persistence approach for numerical comparisons. Equal leading feature
lifetimes use GUDHI to preserve the displacement score's selection rule.

Both the CLI and GUI store validated winter arrays and triangulations in
`.superpov-cache` under the working directory. Later launches reuse those files.
The cache contains numeric arrays and JSON, uses atomic writes, and invalidates
changed inputs or relevant dependency versions. Unavailable or corrupt cache
files are rebuilt when possible. It holds no completed scores. Each selected
winter uses about 12 MB at the standard resolution; disk entries remain until
you remove the cache directory. Eight winters are retained in memory per analyzer.

Use `--cache-dir PATH` to choose a CLI cache location, or `--no-disk-cache` to
keep preparation in memory only. The Python API defaults to in-memory caching:
`DailyAnalyzer(cache_dir=PATH)` explicitly enables persistence across launches.
The first query still pays for parsing, validation, and geometry preparation.

For December 23, the warm daily calculation now takes **9.0 ms**, versus 78.9 ms in the GUDHI direct engine and 4.287 s in the original function. A fresh analyzer using the disk cache takes 112 ms; a complete new CLI process takes 0.309 s. The first query without a disk cache still takes 1.042 s. These are medians of three CPU runs with warm filesystem caches; only the CLI figure includes process startup and imports. Both full diagrams and scores matched the original extended calculation on all 181 supplied winter days.

The headless command prints one JSON result per date, including the selected
method and query time. Its query timer excludes process startup, initial module
imports, analyzer construction, and JSON output. Dependencies first imported
during a query are included, so an uncached CLI query can take longer than the
in-process benchmark with dependencies already loaded. The separate CLI
benchmark measures the entire process.

Input files must contain `# date`, `lev`, `lat`, `lon`, and `data` columns,
one pressure level per winter file, and complete regular latitude/longitude
grids ordered north-to-south and eastward. Use
`--resolution LATITUDES LONGITUDES` for another resolution.

See [the benchmark documentation](benchmarks/README.md) for measured speedups,
full-winter validation, mathematical references, reproducible commands, and
limitations. Keep `superpov_compute.py`, `superpov_cache.py`,
`superpov_planar.py`, and `superpov_planar.cpp` next to the main application.
