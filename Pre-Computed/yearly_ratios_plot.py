# For a given year, plot the ratios for every day

#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------

import matplotlib                            #for plotting
import matplotlib.pyplot as plt              #for plotting
import ast
import numpy as np
import pandas as pd                          #for file import
matplotlib.use("TkAgg")                      #for GUI
import tkinter as tk                         #for GUI
from tkinter import messagebox               #for GUI

#---------------------------------------------------------------------------
# Function
#---------------------------------------------------------------------------
# Function for the GUI
def ask_year_GUI():
   root = tk.Tk()
   root.title("Enter Year")
   root.lift()
   root.attributes('-topmost', True)
   year_var = tk.StringVar()
   tk.Label(
      root,
      text="Enter the year you want to plot for (It will plot the winter starting with the year you enter, ending with the next year):"
   ).pack(padx=80, pady=40)
   entry = tk.Entry(root, textvariable=year_var)
   entry.pack(padx=40, pady=20)
   entry.focus_force()
   def on_submit(event = None):
      year = year_var.get().strip()
      if len(year) != 4:
         messagebox.showerror("Invalid Date", "Please enter a valid year of the form YYYY:")
         entry.focus_force()
         return
      if int(year) < 1959 or int(year) > 2021:
         messagebox.showerror("Invalid Date", "Please enter a year in between 1959 and 2021:")
         entry.focus_force()
         return
      root.quit()
      return year
   #year = on_submit()
   root.bind('<Return>', on_submit)
   tk.Button(root, text="Enter", command=on_submit).pack(pady=10)

   root.mainloop()
   year = year_var.get().strip()
   root.destroy()

   return year


#-----Import-----#
with open("Sp_Disp_Def_Sign_date.txt", "r") as f:
    data = f.read()

array = ast.literal_eval(data)

year = ask_year_GUI()
#year = '1976'
later_year = str(int(year) + 1)

dates = []
ratio_data = []
neg_wind_dates = []
NWD_D_data = []
NWD_S_data = []
for item in array:
   if item[3][0:7] in [year + '-11', year + '-12'] or item[3][0:7] in [later_year + '-01', later_year + '-02', later_year + '-03', later_year + '-04']:
      dates.append(item[3])
      ratio_data.append(item[0])
      if item[2] < 0:
         neg_wind_dates.append(item[3])
         NWD_D_data.append(item[0][0])
         NWD_S_data.append(item[0][1])


disp_ratios = [item[0] for item in ratio_data]
split_ratios = [item[1] for item in ratio_data]

fig = plt.figure(figsize = (15,7), layout = 'constrained')
#fig.suptitle(year)
subfigs = fig.subfigures(2, 1, wspace = 0.07)

split_ax = subfigs[0].subplots(1,1)
disp_ax = subfigs[1].subplots(1,1)

split_ax.plot(dates, split_ratios, marker = "o", linestyle = "dashed")
split_ax.scatter(neg_wind_dates, NWD_S_data, marker = "s", c = "red", zorder  = 3)
split_ax.set_ylim(0,105)
split_ax.set_ylabel("Split Score")
split_ax.tick_params(axis = 'x', labelrotation = 60)
split_labels = split_ax.get_xticklabels()
for i, label in enumerate(split_labels):
    if i % 5 != 0:
        label.set_visible(False)

disp_ax.plot(dates, disp_ratios, marker = "o", linestyle = "dashed")
disp_ax.scatter(neg_wind_dates, NWD_D_data, marker = "s", c = "red", zorder  = 3)
disp_ax.set_ylim(0,105)
disp_ax.set_ylabel("Displacement Score")
disp_ax.tick_params(axis = 'x', labelrotation = 60)
disp_labels = disp_ax.get_xticklabels()
for i, label in enumerate(disp_labels):
    if i % 5 != 0:
        label.set_visible(False)

#-----------
# Show plots
#-----------
plt.show()
