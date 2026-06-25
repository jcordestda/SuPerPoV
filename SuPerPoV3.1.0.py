#Outputs GUI that will ask for a date. Then will ask for how many days before that date, and how many days after that date you want to collect data for
#Will also ask if you what pressure to plot, if you want an Output to terminal and if you want to display wind speed
#Version 3.1.0: Can accept different resolutions and user has to input folder path to data

#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------

import glob                                  #for file import
import pandas as pd                          #for file import
import matplotlib                            #for plotting
matplotlib.use("TkAgg")                      #for GUI
import matplotlib.pyplot as plt              #for plotting
import numpy as np                           #for data tracking
from datetime import datetime, timedelta     #for dates
from matplotlib import cm                    #for colors
from scipy.spatial import Delaunay           #for triangulation
import gudhi as gd                           #for TDA
import geopy.distance as geod                #for distance to NP
import tkinter as tk                         #for GUI
from tkinter import messagebox               #for GUI
import time

#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
# Function for the GUI
def ask_date_GUI():
   root = tk.Tk()
   root.title("SuPerPoV")
   root.lift()
   root.attributes('-topmost', True)
   root.grid_columnconfigure(0, weight=1, uniform="equal")
   root.grid_columnconfigure(1, weight=1, uniform="equal")
   input_dict = {}

   # Date Section
   date_frame = tk.LabelFrame(root, text="DATE", padx = 10, pady = 10)
   date_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
   tk.Label(date_frame, text="Enter date (YYYY-MM-DD):\nOnly Valid Months are 11, 12, 01, 02, 03, 04.").pack(padx=80, pady=10)
   entry = tk.Entry(date_frame)
   entry.pack(padx=40, pady=20)
   entry.focus_force()
   # Entries for days before
   tk.Label(date_frame, text="Number of days BEFORE entered date:").pack()
   entry_before = tk.Entry(date_frame)
   entry_before.pack(pady=5)
   # Entries for days after
   tk.Label(date_frame, text="Number of days AFTER entered date:").pack()
   entry_after = tk.Entry(date_frame)
   entry_after.pack(pady=5)

   # Pressure Section
   press_frame = tk.LabelFrame(root, text="Pressures", padx=10, pady=10)
   press_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
   press_options = [2, 5, 10, 50, 100]
   press_vars = {}
   for press in press_options:
      if press == 10:
         var = tk.BooleanVar(value=True)
      else:
         var = tk.BooleanVar(value=False)
      tk.Checkbutton(press_frame, text=press, variable=var).pack(side="left", padx=10)
      press_vars[press] = var

   # Choose what to plot
   plot_var = tk.IntVar(value=10)
   plot_frame = tk.LabelFrame(root, text="WHAT PRESSURE TO PLOT", padx=10, pady=10)
   plot_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
   tk.Radiobutton(plot_frame, text="2", variable=plot_var, value=2).pack(side="left", padx=10)
   tk.Radiobutton(plot_frame, text="5", variable=plot_var, value=5).pack(side="left", padx=10)
   tk.Radiobutton(plot_frame, text="10", variable=plot_var, value=10).pack(side="left", padx=10)
   tk.Radiobutton(plot_frame, text="50", variable=plot_var, value=50).pack(side="left", padx=10)
   tk.Radiobutton(plot_frame, text="100", variable=plot_var, value=100).pack(side="left", padx=10)

   # Output Data to Terminal
   output_var = tk.StringVar(value="C")
   output_frame = tk.LabelFrame(root, text="OUTPUT DATA TO TERMINAL", padx=10, pady=10)
   output_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
   tk.Radiobutton(output_frame, text="Output", variable=output_var, value="C").pack(side="left", padx=10)
   tk.Radiobutton(output_frame, text="Don't output", variable=output_var, value="D").pack(side="left", padx=10)

   # Wind Speed
   wind_var = tk.StringVar(value="E")
   wind_frame = tk.LabelFrame(root, text="SHOW REVERSAL WIND SPEED", padx=10, pady=10)
   wind_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
   tk.Radiobutton(wind_frame, text="Display wind speed reversal", variable=wind_var, value="E").pack(side="left", padx=10)
   tk.Radiobutton(wind_frame, text="Don't display wind speed reversal", variable=wind_var, value="F").pack(side="left", padx=10)

   # Resolution
   lat_var = tk.IntVar(value=46)
   long_var = tk.IntVar(value=180)
   guess_var = tk.StringVar(value="default")
   def toggle_entries():
      if guess_var.get() == "default":
         lat_entry.config(state="normal")
         long_entry.config(state="normal")
      else:
          lat_entry.config(state="disabled")
          long_entry.config(state="disabled")
   res_frame = tk.LabelFrame(root, text="CHOOSE RESOLUTION", padx=10, pady=10)
   res_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
   tk.Radiobutton(res_frame, text="Enter resolution.", variable=guess_var, value="default", command=toggle_entries).grid(row=0, column=0, sticky="w", padx=10)
   #tk.Radiobutton(res_frame, text="Let code guess resolution.", variable=guess_var, value="custom", command=toggle_entries).grid(row=0, column=1, sticky="w", padx=10)
   lat_frame = tk.Frame(res_frame)
   lat_frame.grid(row=1, column=0, sticky="w", padx=10, pady=10)
   tk.Label(lat_frame, text="Latitude:").pack(side="left")
   lat_entry = tk.Entry(lat_frame, textvariable=lat_var, width=5)
   lat_entry.pack(side="left", padx=5)
   long_frame = tk.Frame(res_frame)
   long_frame.grid(row=1, column=1, sticky="w", padx=10, pady=10)
   tk.Label(long_frame, text="Longitude:").pack(side="left")
   long_entry = tk.Entry(long_frame, textvariable=long_var, width=5)
   long_entry.pack(side="left")

   # Path to data
   path_var = tk.StringVar(value="")
   path_frame = tk.LabelFrame(root, text="PASTE FOLDER LOCATION", padx=10, pady=10)
   path_frame.grid(row=3, column=0, columnspan = 2, padx=10, pady=10, sticky="nsew")
   tk.Label(path_frame, text="Enter folder location of the data:").pack(side="left")
   tk.Entry(path_frame, textvariable=path_var, width=70).pack(side="left", padx=10)

   def on_submit(event = None):
       date = entry.get()
       days_B = entry_before.get()
       days_A = entry_after.get()
       #PLOT TYPE
       input_dict["PlotType"] = plot_var.get()
       #PRESSURES
       input_dict["Pressures"] = []
       for press_name, press_obj in press_vars.items():
          if press_obj.get() == True: input_dict["Pressures"].append(press_name)
       if len(input_dict["Pressures"]) == 0:
          messagebox.showerror("Pressure required", "You must select a pressure to continue.", parent=root)
          return
       if input_dict["PlotType"] not in input_dict["Pressures"]:
          messagebox.showerror("Pressure required", "You can only plot a pressure that you select.", parent=root)
          return
       #RESOLUTION
       if guess_var.get() == "default":
          try:
             input_dict["Resolution"] = (lat_var.get(), long_var.get())
          except tk.TclError:
             messagebox.showerror("Resolution Required", "Please enter valid numbers for latitude and longitude.", parent=root)
             return
       else:
          input_dict["Resolution"] = "guess"
       #OUTPUT
       input_dict["Output"] = output_var.get()
       #WIND
       input_dict["Wind"] = wind_var.get()
       #PATH
       if path_var.get() == "":
          messagebox.showerror("Path Required", "Path to data required.", parent=root)
          return
       else:
          input_dict["Path"] = path_var.get().strip()
       try:
          #DATE
          input_dict["date"] = datetime.strptime(date, "%Y-%m-%d")
          input_dict["daysBefore"] = int(days_B)
          input_dict["daysAfter"] = int(days_A)
          root.destroy()
          root.quit()
       except ValueError:
          messagebox.showerror("Invalid Date", "Please enter a valid date in YYYY-MM-DD format. \nDon't leave days before or after blank. 0 is a valid entry.", parent=root)
          entry.focus_force()
   root.bind('<Return>', on_submit)
   tk.Button(root, text="Enter", command=on_submit).grid(row=4, column=0, columnspan=2, pady=10)
   root.mainloop()
   if "date" in input_dict and "daysBefore" in input_dict and "daysAfter" in input_dict:
      return input_dict
##------------------------------------------##
##-------------END ASK_DATE_GUI-------------##
##------------------------------------------##

# Returns the correct data frame for the given date
def correct_df(txt_files, date_year, agree):
   if agree:
      for file in txt_files:
         if pd.read_csv(file, sep='\t')["# date"][0][0:4] == date_year:
            return pd.read_csv(file, sep='\t')
   else:
      for file in txt_files:
         if pd.read_csv(file, sep='\t')["# date"][0][0:4] == str(int(date_year) - 1):
            return pd.read_csv(file, sep='\t')

# Returns [index1, index2] where index1 is the index in the df when that
# date starts and index2 is when that date ends
def date_index_returner(df, resolution, date = ""):
   return_tuple = [0,0]
   if df["# date"][0] == date:
      return (0, resolution[0] * resolution[1])                                                                              		#--FLAG
   i = 0
   while df["# date"][i] != date: i+=1
   return_tuple[0] = i
   if date[5:] == "04-30":
      return_tuple[1] = len(df)
      return return_tuple
   while df["# date"][i] == date: i+=1
   return_tuple[1] = i
   return return_tuple

# Returns the height, split_ratio and disp_ratio for a given date
def compute_data(date, txt_files, resolution):
   # Grab the year and month
   date_year = date[0:4]
   date_month = date[5:7]

   # Is 11-01 in the same year as the date given? Or is it in the year before?
   if date_month in ["11","12"]:
      date_year_agree = True
   else:
      date_year_agree = False

   # Grab the right Data Frame for the given year
   df = correct_df(txt_files, date_year, date_year_agree)

   # Reduce to the given day
   date_l_index, date_u_index = date_index_returner(df, resolution, date)

   z = np.array([])
   for i in range(date_l_index, date_u_index): z = np.append(z, df["data"][i])
   Z = z.reshape(resolution)                                                                      	#--FLAG

   #--SIMPLEX TREE CYL--
   r = np.linspace(0.001, 4, resolution[0])                                                                	#--FLAG
   theta = np.linspace(0, 2*np.pi, resolution[1], endpoint=False)                                         	#--FLAG
   R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')
   # Convert polar to cartesian
   X = R_grid * np.cos(Theta_grid)
   Y = R_grid * np.sin(Theta_grid)
   # Triangulate
   X_flat = X.flatten()
   Y_flat = Y.flatten()
   stack = np.column_stack([X_flat, Y_flat])
   tri = Delaunay(stack)

   st = gd.SimplexTree()
   coords = []
   for i in range(len(X_flat)):
      temp = [X_flat[i], Y_flat[i], z[i]]
      coords.append(temp)
   coords = np.array(coords)

   for i in range(len(tri.simplices)):
      st.insert([v for v in tri.simplices[i,:]], -10)
   for i in range(len(coords)):
      if st.find([i]):
         st.assign_filtration([i], coords[i,2])
   st.make_filtration_non_decreasing()
   st.extend_filtration()
   fvals_cyl = st.extended_persistence(min_persistence=1e-5)

   # Massage persistence results
   ordfvals_cyl = fvals_cyl[0]
   ordfvals_cyl.sort(reverse = True, key = lambda x : x[1][1] - x[1][0])
   ord1fvals_cyl = [bar for bar in ordfvals_cyl if bar[0] == 1]
   relfvals_cyl = fvals_cyl[1]
   relfvals_cyl.sort(reverse = True, key = lambda x : x[1][0] - x[1][1])
   rel2fvals_cyl = [bar for bar in relfvals_cyl if bar[0] == 2]

   #--SIMPLEX TREE GRID--
   Z2 = np.hstack((Z,Z)) #doubled matrix
   Z2_flat = Z2.flatten()
   # Triangulate
   X = np.linspace(0, 90, resolution[0])                                   	                                #--FLAG
   Y = np.linspace(0, 720, 2*resolution[1], endpoint=False)          						#--FLAG
   X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')
   X_flat = X_grid.flatten()
   Y_flat = Y_grid.flatten()
   stack2 = np.column_stack([X_flat, Y_flat])
   tri2 = Delaunay(stack2)

   st2 = gd.SimplexTree()
   coords2 = []
   for i in range(len(X_flat)):
      temp = [X_flat[i], Y_flat[i], Z2_flat[i]]
      coords2.append(temp)
   coords2 = np.array(coords2)

   for i in range(len(tri2.simplices)):
      st2.insert([v for v in tri2.simplices[i,:]], -10)
   for i in range(len(coords2)):
      if st2.find([i]):
         st2.assign_filtration([i], coords2[i,2])
   st2.make_filtration_non_decreasing()
   st2.extend_filtration()
   fvals_grid = st2.extended_persistence(min_persistence=1e-5)

   # Massage persistence results
   ordfvals_grid = fvals_grid[0]
   ordfvals_grid.sort(reverse = True, key = lambda x : x[1][1] - x[1][0])
   ord1fvals_grid = [bar for bar in ordfvals_grid if bar[0] == 1]
   relfvals_grid = fvals_grid[1]
   relfvals_grid.sort(reverse = True, key = lambda x : x[1][0] - x[1][1])
   rel2fvals_grid = [bar for bar in relfvals_grid if bar[0] == 2]

   depths = [(bar[1][0] - bar[1][1]) for bar in rel2fvals_cyl]
   if len(rel2fvals_cyl) > 1:
      split_ratio = ((rel2fvals_cyl[1][1][0] - rel2fvals_cyl[1][1][1]) / depths[0])
   else:
      split_ratio = 0

   matched = []
   for bar in rel2fvals_grid:
      if rel2fvals_cyl[0][1][1] == bar[1][1] and matched == []:
         matched = [bar, rel2fvals_cyl[0]]
   #print("matched:", matched)

   if matched == []:
      disp_ratio = 0
   else:
      #print("top:", matched[0][1][0] - matched[0][1][1])
      #print("bottom:", matched[1][1][0] - matched[1][1][1])
      disp_ratio = ((matched[0][1][0] - matched[0][1][1]) / (matched[1][1][0] - matched[1][1][1])) 

   return [depths, split_ratio, disp_ratio]
##------------------------------------------##
##-------------END COMPUTE_DATA-------------##
##------------------------------------------##

def n_days_prior(date, days_B):
   test_array = []
   nums = np.arange(days_B)
   nums += 1
   for num in nums:
      if num < 10:
         test_array.append("11-0" + str(num))
      else:
         test_array.append("11-" + str(num))
   return_arr = []
   if date[5:10] in test_array:
      year = date[0:4]
      for test in test_array:
         return_arr.append(year + "-" + test)
      return return_arr
   else:
      DTdate = datetime.strptime(date, "%Y-%m-%d")
      nums = nums[::-1]
      for num in nums:
         return_arr.append(str((DTdate-timedelta(days=int(num))).date()))
      return return_arr

def n_days_after(date, days_A):
   test_array = []
   nums = np.arange(1, days_A + 1)
   large_nums = 30 - nums
   for num in large_nums:
      test_array.append("04-" + str(num))
   return_arr = []
   if date[5:10] in test_array:
      year = date[0:4]
      for test in test_array:
         return_arr.append(year + "-" + test)
      return return_arr
   else:
      DTdate = datetime.strptime(date, "%Y-%m-%d")
      for num in nums:
         return_arr.append(str((DTdate+timedelta(days=int(num))).date()))
      return return_arr

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
start_time = time.perf_counter()

input_dict = ask_date_GUI()
date = input_dict["date"].date()
date = str(date)
OG_date = date
print()
print("-" * 90)
days_B = input_dict["daysBefore"]
days_A = input_dict["daysAfter"]
pressures = input_dict["Pressures"]
if len(pressures) == 1:
   pressures_string = str(pressures[0])
   print(f"\nComputing pressure {pressures_string}hpa for {days_B} day(s) before {date} and {days_A} day(s) after {date}.\n")
else:
   pressures_string = ", ".join(map(str, pressures[:-1])) + f", and {pressures[-1]}"
   print(f"\nComputing pressures {pressures_string} in hpa for {days_B} day(s) before {date} and {days_A} day(s) after {date}.\n")
print("-" * 90)
print()
output_yn = input_dict["Output"]     #C = Output, D = No output
wind_yn = input_dict["Wind"]         #E = Show wind speed, F = Don't show wind speed
plot_type = input_dict["PlotType"]
resolution = input_dict["Resolution"]

# Define figure with plots
fig = plt.figure(figsize = (15,8), layout = 'constrained')
subfigs = fig.subfigures(2, 2, wspace = 0.07)

# Folder path containing the datasets
parent_path = input_dict["Path"]
folder_path2 = parent_path + "/gph_2hpa/"
folder_path5 = parent_path + "/gph_5hPa/"
folder_path10 = parent_path + "/gph_10hpa/"
folder_path50 = parent_path + "/gph_50hPa/"
folder_path100 = parent_path + "/gph_100hPa/"

# Use glob to find all .txt files
txt_files2 = glob.glob(f"{folder_path2}/*.txt")
txt_files5 = glob.glob(f"{folder_path5}/*.txt")
txt_files10 = glob.glob(f"{folder_path10}/*.txt")
txt_files50 = glob.glob(f"{folder_path50}/*.txt")
txt_files100 = glob.glob(f"{folder_path100}/*.txt")

# Grab the year and month
date_year = date[0:4]
date_month = date[5:7]

# Is 11-01 in the same year as the date given? Or is it in the year before?
if date_month in ["11","12"]:
   date_year_agree = True
else:
   date_year_agree = False

# Grab the right Data Frame for the given year
if plot_type == 2: df = correct_df(txt_files2, date_year, date_year_agree)
elif plot_type == 5: df = correct_df(txt_files5, date_year, date_year_agree)
elif plot_type == 10: df = correct_df(txt_files10, date_year, date_year_agree)
elif plot_type == 50: df = correct_df(txt_files50, date_year, date_year_agree)
elif plot_type == 100: df = correct_df(txt_files100, date_year, date_year_agree)

# Reduce to the given day
date_l_index, date_u_index = date_index_returner(df, resolution, date)

z = np.array([])
for i in range(date_l_index, date_u_index): z = np.append(z, df["data"][i])
Z = z.reshape(resolution)												#--FLAG

#------------------------
# Cylindrical Plot
#------------------------
graph_ax = subfigs[0][0].subplots(1, 2, subplot_kw = {"projection" : "3d"})

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0.001, 4, resolution[0])											#--FLAG
theta = np.linspace(0, 2*np.pi, resolution[1])										#--FLAG
R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')

# Convert polar to cartesian for 3D plotting
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Plot the surface.
graph_ax[0].plot_surface(X, Y, Z, cmap=cm.managua)

graph_ax[0].set_title(f"Plot for {plot_type}hPa.")

# Customize the z axis.
if plot_type == 10: graph_ax[0].set_zlim(27811, 31668) #min = 27811.846 , max = 31667.023
graph_ax[0].view_init(elev = 15, azim = -10)

#------------------------
# Grid Plot
#------------------------

graph_ax[1].set_xlabel("Latitude")
graph_ax[1].set_ylabel("Longitude")
graph_ax[1].view_init(elev = 20, azim = -10)

# Create meshgrid for lats and longs
lats = np.array([90 - 2*i for i in range(0, resolution[0])])								#--FLAG
longs = np.array([2*i for i in range(0, resolution[1])])									#--FLAG
longs, lats = np.meshgrid(longs, lats)

surf = graph_ax[1].plot_surface(lats, longs, Z, cmap = cm.managua, linewidth=0)

# Red line on surface that represents one point
x_line = lats[0, :]
y_line = longs[0, :]
z_line_on_surface = Z[0, :]
graph_ax[1].plot(x_line, y_line, z_line_on_surface, color='red', linewidth = 3)

# Customize the z axis.
if plot_type == 10: graph_ax[1].set_zlim(27811, 31668) #min = 27811.846 , max = 31667.023

#------------------------
# Outputs:
#------------------------
# Gather the correct dates and organize them
prior_dates = n_days_prior(date, days_B)
after_dates = n_days_after(date, days_A)
all_dates = list((set(prior_dates).union([date])).union(after_dates))
sorted_dates = sorted(all_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d"))
num_dates = len(sorted_dates)
skips = int(num_dates / 20)

# Start compiling data
all_data2 = []
all_data5 = []
all_data10 = []
all_data50 = []
all_data100 = []
for date in sorted_dates:
   if 2 in pressures:
      all_data2.append(compute_data(date, txt_files2, resolution))
   if 5 in pressures:
      all_data5.append(compute_data(date, txt_files5, resolution))
   if 10 in pressures:
      all_data10.append(compute_data(date, txt_files10, resolution))
   if 50 in pressures:
      all_data50.append(compute_data(date, txt_files50, resolution))
   if 100 in pressures:
      all_data100.append(compute_data(date, txt_files100, resolution))
all_depths2 = [item[0] for item in all_data2]
first_depths2 = [depth[0] for depth in all_depths2]
second_depths2 = [depth[1] if len(depth) > 1 else 0 for depth in all_depths2]
split_ratios2 = [item[1] for item in all_data2]
disp_ratios2 = [item[2] for item in all_data2]
all_depths5 = [item[0] for item in all_data5]
first_depths5 = [depth[0] for depth in all_depths5]
second_depths5 = [depth[1] if len(depth) > 1 else 0 for depth in all_depths5]
split_ratios5 = [item[1] for item in all_data5]
disp_ratios5 = [item[2] for item in all_data5]
all_depths10 = [item[0] for item in all_data10]
first_depths10 = [depth[0] for depth in all_depths10]
second_depths10 = [depth[1] if len(depth) > 1 else 0 for depth in all_depths10]
split_ratios10 = [item[1] for item in all_data10]
disp_ratios10 = [item[2] for item in all_data10]
all_depths50 = [item[0] for item in all_data50]
first_depths50 = [depth[0] for depth in all_depths50]
second_depths50 = [depth[1] if len(depth) > 1 else 0 for depth in all_depths50]
split_ratios50 = [item[1] for item in all_data50]
disp_ratios50 = [item[2] for item in all_data50]
all_depths100 = [item[0] for item in all_data100]
first_depths100 = [depth[0] for depth in all_depths100]
second_depths100 = [depth[1] if len(depth) > 1 else 0 for depth in all_depths100]
split_ratios100 = [item[1] for item in all_data100]
disp_ratios100 = [item[2] for item in all_data100]

if output_yn == "C":
   # Print exact values to terminal
   print("-" * 90)
   print("   DATE    : [HEIGHT,   SPLIT SCORE,   DISPLACEMENT SCORE]\n")
   output2 = {k: [v1, v2, v3] for k, v1, v2, v3 in zip(sorted_dates, first_depths2, split_ratios2, disp_ratios2)}
   output5 = {k: [v1, v2, v3] for k, v1, v2, v3 in zip(sorted_dates, first_depths5, split_ratios5, disp_ratios5)}
   output10 = {k: [v1, v2, v3] for k, v1, v2, v3 in zip(sorted_dates, first_depths10, split_ratios10, disp_ratios10)}
   output50 = {k: [v1, v2, v3] for k, v1, v2, v3 in zip(sorted_dates, first_depths50, split_ratios50, disp_ratios50)}
   output100 = {k: [v1, v2, v3] for k, v1, v2, v3 in zip(sorted_dates, first_depths100, split_ratios100, disp_ratios100)}
   if disp_ratios2 != []: print("2 HPA:")
   for key, value in output2.items():
      print(f"{key} : {value}")
   if disp_ratios5 != []: print("5 HPA:")
   for key, value in output5.items():
      print(f"{key} : {value}")
   if disp_ratios10 != []: print("10 HPA:")
   for key, value in output10.items():
      print(f"{key} : {value}")
   if disp_ratios50 != []: print("50 HPA:")
   for key, value in output50.items():
      print(f"{key} : {value}")
   if disp_ratios100 != []: print("100 HPA:")
   for key, value in output100.items():
      print(f"{key} : {value}")
   print("-" * 90)
   print()

neg_wind_dates = []
NWD_D_data2 = []        #Negative Wind Dates Displacement data at 2 hPa
NWD_D_data5 = []        #Negative Wind Dates Displacement data at 5 hPa
NWD_D_data10 = []       #Negative Wind Dates Displacement data at 10 hPa
NWD_D_data50 = []       #Negative Wind Dates Displacement data at 50 hPa
NWD_D_data100 = []      #Negative Wind Dates Displacement data at 100 hPa
NWD_S_data2 = []        #Negative Wind Dates Split data at 2 hPa
NWD_S_data5 = []        #Negative Wind Dates Split data at 5 hPa
NWD_S_data10 = []       #Negative Wind Dates Split data at 10 hPa
NWD_S_data50 = []       #Negative Wind Dates Split data at 50 hPa
NWD_S_data100 = []      #Negative Wind Dates Split data at 100 hPa
if wind_yn == "E":
   # Wind speeds
   wind_file = "/Users/jakecordes/Desktop/Coding/TDAProject1/data_from_code/u1060.txt"
   read_wind_file = pd.read_csv(wind_file, sep='\t')
   for date in sorted_dates:
      for i in range(len(read_wind_file['DATES'])):
         if date == read_wind_file['DATES'][i]:
            if read_wind_file['SPEEDS'][i] < 0:
               neg_wind_dates.append(date)
               index = sorted_dates.index(date)
               if disp_ratios2 != []: NWD_D_data2.append(disp_ratios2[index])
               if disp_ratios5 != []: NWD_D_data5.append(disp_ratios5[index])
               if disp_ratios10 != []: NWD_D_data10.append(disp_ratios10[index])
               if disp_ratios50 != []: NWD_D_data50.append(disp_ratios50[index])
               if disp_ratios100 != []: NWD_D_data100.append(disp_ratios100[index])
               if split_ratios2 != []: NWD_S_data2.append(split_ratios2[index])
               if split_ratios5 != []: NWD_S_data5.append(split_ratios5[index])
               if split_ratios10 != []: NWD_S_data10.append(split_ratios10[index])
               if split_ratios50 != []: NWD_S_data50.append(split_ratios50[index])
               if split_ratios100 != []: NWD_S_data100.append(split_ratios100[index])
               """
               neg_wind_dates.append(date)
               index = sorted_dates.index(date)
               NWD_D_data2.append(disp_ratios2[index])
               NWD_D_data5.append(disp_ratios5[index])
               NWD_D_data10.append(disp_ratios10[index])
               NWD_D_data50.append(disp_ratios50[index])
               NWD_D_data100.append(disp_ratios100[index])
               NWD_S_data2.append(split_ratios2[index])
               NWD_S_data5.append(split_ratios5[index])
               NWD_S_data10.append(split_ratios10[index])
               NWD_S_data50.append(split_ratios50[index])
               NWD_S_data100.append(split_ratios100[index])
               """
## -- Split Ratio Plot -- ##
split_ax = subfigs[0][1].subplots(1, 1)
if split_ratios2 != []:
   split_ax.plot(sorted_dates, split_ratios2, marker = "o", color = 'cyan', linestyle = "dashed", label = "2 hPa")
   split_ax.scatter(neg_wind_dates, NWD_S_data2, s = 60, marker = "s", c = "red", zorder  = 3)
if split_ratios5 != []:
   split_ax.plot(sorted_dates, split_ratios5, marker = "o", color = 'olive', linestyle = "dashed", label = "5 hPa")
   split_ax.scatter(neg_wind_dates, NWD_S_data5, s = 60, marker = "s", c = "red", zorder  = 3)
if split_ratios10 != []:
   split_ax.plot(sorted_dates, split_ratios10, marker = "o", color = 'blue', linestyle = "dashed", label = "10 hPa")
   split_ax.scatter(neg_wind_dates, NWD_S_data10, s = 60, marker = "s", c = "red", zorder  = 3)
if split_ratios50 != []:
   split_ax.plot(sorted_dates, split_ratios50, marker = "o", color = 'green', linestyle = "dashed", label = "50 hPa")
   split_ax.scatter(neg_wind_dates, NWD_S_data50, s = 60, marker = "s", c = "red", zorder  = 3)
if split_ratios100 != []:
   split_ax.plot(sorted_dates, split_ratios100, marker = "o", color = 'purple', linestyle = "dashed", label = "100 hPa")
   split_ax.scatter(neg_wind_dates, NWD_S_data100, s = 60, marker = "s", c = "red", zorder  = 3)
split_ax.set_ylim(0,1.05)
split_ax.set_ylabel("Split Score")
split_ax.tick_params(axis = 'x', labelrotation = 60)
split_ax.legend()
for label in split_ax.get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")
        break
# If there are too many dates, clean up x-axis
if skips > 0:
   split_labels = split_ax.get_xticklabels()
   for i, label in enumerate(split_labels):
      if i % skips != 0:
         label.set_visible(False)

## -- Displacement Ratio Plot -- ##
disp_ax = subfigs[1][1].subplots(1, 1)
if disp_ratios2 != []:
   disp_ax.plot(sorted_dates, disp_ratios2, marker = "o", color = 'cyan', linestyle = "dashed", label = "2 hPa")
   disp_ax.scatter(neg_wind_dates, NWD_D_data2, s = 60, marker = "s", c = "red", zorder  = 3)
if disp_ratios5 != []:
   disp_ax.plot(sorted_dates, disp_ratios5, marker = "o", color = 'olive', linestyle = "dashed", label = "5 hPa")
   disp_ax.scatter(neg_wind_dates, NWD_D_data5, s = 60, marker = "s", c = "red", zorder  = 3)
if disp_ratios10 != []:
   disp_ax.plot(sorted_dates, disp_ratios10, marker = "o", color = 'blue', linestyle = "dashed", label = "10 hPa")
   disp_ax.scatter(neg_wind_dates, NWD_D_data10, s = 60, marker = "s", c = "red", zorder  = 3)
if disp_ratios50 != []:
   disp_ax.plot(sorted_dates, disp_ratios50, marker = "o", color = 'green', linestyle = "dashed", label = "50 hPa")
   disp_ax.scatter(neg_wind_dates, NWD_D_data50, s = 60, marker = "s", c = "red", zorder  = 3)
if disp_ratios100 != []:
   disp_ax.plot(sorted_dates, disp_ratios100, marker = "o", color = 'purple', linestyle = "dashed", label = "100 hPa")
   disp_ax.scatter(neg_wind_dates, NWD_D_data100, s = 60, marker = "s", c = "red", zorder  = 3)
disp_ax.set_ylim(0,1.05)
disp_ax.set_ylabel("Displacement Score")
disp_ax.tick_params(axis = 'x', labelrotation = 60)
disp_ax.legend()
for label in disp_ax.get_xticklabels():
    if label.get_text() == OG_date: 
        label.set_fontweight("bold")  
        break
# If there are too many dates, clean up x-axis
if skips > 0:
   disp_labels = disp_ax.get_xticklabels()
   for i, label in enumerate(disp_labels):
      if i % skips != 0:
         label.set_visible(False)

## -- Life Span Plot -- ##
bar_ax = subfigs[1][0].subplots(1, 1)
if plot_type == 2:
   first_depths_var = first_depths2
   second_depths_var = second_depths2
if plot_type == 5:
   first_depths_var = first_depths5
   second_depths_var = second_depths5
if plot_type == 10:
   first_depths_var = first_depths10
   second_depths_var = second_depths10
if plot_type == 50:
   first_depths_var = first_depths50
   second_depths_var = second_depths50
if plot_type == 100:
   first_depths_var = first_depths100
   second_depths_var = second_depths100
bar_ax.bar(sorted_dates, first_depths_var, color = "orange", label = "Largest")
bar_ax.bar(sorted_dates, second_depths_var, color = "steelblue", label = "2nd Largest")
top_depth = max(3500, max(first_depths_var))
bar_ax.set_ylim(0, top_depth)
bar_ax.set_ylabel(f"Life Span for {plot_type}hPa")
bar_ax.tick_params(axis = 'x', labelrotation = 60)
bar_ax.axhline(y = 2387.7611 + 518.5322, linestyle=':', color='black')
bar_ax.axhline(y = 2387.7611, linestyle='--', color = 'black', label = 'Average')
bar_ax.axhline(y = 2387.7611 - 518.5322, linestyle=':', color = 'black')
bar_ax.legend()
for label in bar_ax.get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")  
        break
# If there are too many dates, clean up x-axis
if skips > 0:
   bar_labels = bar_ax.get_xticklabels()
   for i, label in enumerate(bar_labels):
      if i % skips != 0:
         label.set_visible(False)

#---------
# Finished
#---------
end_time = time.perf_counter()
print(f"Code ran in {end_time - start_time:.4f} seconds.")

#-----------
# Show plots
#-----------
plt.show()
