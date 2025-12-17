# SuPerPoV: Score and evolution of polar vortices via persistent homology

The goal of this project was to define a new way of viewing Geopotential Height when analyzing the shape of Polar Vortices. The details can be found in the paper "SuPerPoV: Score and evolution of polar vortices via
persistent homology".

This repo contains two python scripts and a folder containing two more written by Jake Cordes and Barbara Giunti. Experiments using these scripts can be found in the paper referenced above. The data used in the experiments are available on Zenodo, https://doi.org/10.5281/zenodo.17903131

__SuPerPoV1.1.1.py__: This is the main file of the project. Enter a given date, number of days before and a number of days after. The output will be 5 plots: 2 showing the shape of the vortex for the entered date, and the remaining 3 plots will describe the height, split scores, and displacement scores over the days entered.<br>
SuPerPoV_hPa_1.0.1.py: Enter a given date, number of days before and a number of days after. The output will be 5 plots: 3 showing the heights for the vortices (one for 10 hPa, one for 50 hPa, and one for 100 hPa), and the remaining 2 plots will describe the split scores, and displacement scores over the days entered, for 10 hPa, 50 hPa, and 100 hPa.<br>
There is a YouTube video describing the files in more detail, which can be found here: XXXXXXXXXXXXXX

Pre-Computed:<br>
This folder contains two python scripts and a .txt file.<br>
proper_plot.py: This script outputs a scatter plot, comparing our method to previous methods. To run, Sp_Disp_Def_Sign_date.txt must be in the same folder.<br>
yearly_ratios_plot,py: Enter a given year, and this will output the displacement scores and split scores for an entire year. To run, Sp_Disp_Def_Sign_date.txt must be in the same folder.<br>
Sp_Disp_Def_Sign_date.txt: A giant array that has information for every winter day from 1960-2022.

Usage:
* Download the scripts and run them in python. Must have python and numpy downloaded to run.
  + Download the data from Zenodo, unzip the files, and place them in the same folder as the scripts.
  + In a command line, run "python SuPerPoV1.1.1.py" to run the file!
