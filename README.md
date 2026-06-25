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
* Python >= 3.8
* Numpy >= 1.20

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
