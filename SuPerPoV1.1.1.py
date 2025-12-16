#Outputs GUI that will ask for a date. Then will ask for how many days before that date, and how many days after that date you want to collect data for
#Both will compare the Cubical Complex and Extended Persistence
#Version 1.1.1: Fixed file location

#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------

import glob                                  #for file import
import os                                    #for file import
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

#---------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------
# Function for the GUI
def ask_date_GUI():
   root = tk.Tk()
   root.title("Enter Date")
   root.lift()
   root.attributes('-topmost', True)
   tk.Label(root, text="Enter date (YYYY-MM-DD):\nOnly Valid Months are 11, 12, 01, 02, 03, 04.").pack(padx=80, pady=40)
   entry = tk.Entry(root)
   entry.pack(padx=40, pady=20)
   entry.focus_force()
   # Entries for before/after days
   tk.Label(root, text="Number of days BEFORE entered date:").pack()
   entry_before = tk.Entry(root)
   entry_before.pack(pady=5)

   tk.Label(root, text="Number of days AFTER entered date:").pack()
   entry_after = tk.Entry(root)
   entry_after.pack(pady=5)
   input_dict = {}
   def on_submit(event = None):
       date = entry.get()
       days_B = entry_before.get()
       days_A = entry_after.get()
       try:
           input_dict["date"] = datetime.strptime(date, "%Y-%m-%d")
           input_dict["daysBefore"] = days_B
           input_dict["daysAfter"] = days_A
           root.destroy()
           root.quit()
       except ValueError:
           messagebox.showerror("Invalid Date", "Please enter a valid date in YYYY-MM-DD format.")
           entry.focus_force()
   root.bind('<Return>', on_submit)
   tk.Button(root, text="Enter", command=on_submit).pack(pady=10)
   root.mainloop()
   if "date" in input_dict and "daysBefore" in input_dict and "daysAfter" in input_dict:
      return input_dict

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
def date_index_returner(df, date):
   if date[5:] == "11-01" and df["# date"][0][5:] != "11-01": return False
   return_tuple = [0,0]
   if df["# date"][0] == date:
      j = 0
      while df["# date"][j] == date: j+=1
      return [0, j]
   i = 0
   try:
      while df["# date"][i] != date: i+=1
      return_tuple[0] = i
   except:
      return False
   if date[5:] == "04-30":
      return_tuple[1] = len(df)
      return return_tuple
   while df["# date"][i] == date:
      if i >= len(df) - 1:
         return_tuple[1] = len(df)
         return return_tuple
      else:
         i+=1
   return_tuple[1] = i
   return return_tuple

# Returns the data for a given day
def compute_data(date):
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
   if not date_index_returner(df, date):
      return [] # No data from this date was found
   else:
      date_l_index, date_u_index = date_index_returner(df, date)

   z = np.array([])
   for i in range(date_l_index, date_u_index): z = np.append(z, df["data"][i])
   Z = z.reshape((46,180))

   #--EXTENDED PERSISTENCE--
   r = np.linspace(0.001, 4, 46)
   theta = np.linspace(0, 2*np.pi, 180, endpoint=False)
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
   fvals = st.extended_persistence(min_persistence=1e-5)

   # Massage persistence results
   ordfvals = fvals[0]
   ordfvals.sort(reverse = True, key = lambda x : x[1][0] - x[1][1])
   ord1fvals = [bar for bar in ordfvals if bar[0] == 1]
   relfvals = fvals[1]
   relfvals.sort(reverse = True, key = lambda x : x[1][0] - x[1][1])
   rel2fvals = [bar for bar in relfvals if bar[0] == 2]
   depths = [(bar[1][0] - bar[1][1]) for bar in rel2fvals]
   split_ratio = ((rel2fvals[1][1][0] - rel2fvals[1][1][1]) / depths[0]) * 100
   # For the shallow ratio
   reverse_depths = [(bar[1][1] - bar[1][0]) for bar in ord1fvals]
   shallow_ratio = max(reverse_depths) / max(depths) 

   #--CUBICAL COMPLEX--
   Z2 = np.hstack((Z,Z)) #doubled matrix
   Z2 = Z2 * (-1)
   tdc = Z2.flatten('F') #tdc = top dimensional cells
   dimensions = Z2.shape
   cubical_complex = gd.CubicalComplex(top_dimensional_cells = tdc, dimensions = dimensions)
   super_fvals = cubical_complex.persistence()
   # Make the negated values positive again
   for i in range(len(super_fvals)):
      temp = list(super_fvals[i])
      temp[1] = list(temp[1])
      temp[1][0] = -temp[1][0]
      temp[1][1] = -temp[1][1]
      temp[1] = tuple(temp[1])
      super_fvals[i] = tuple(temp)
   super_fvals.sort(reverse = True, key = lambda x : x[1][0] - x[1][1])
   superH1 = [bar for bar in super_fvals if bar[0] == 1]

   matched = []
   for bar in superH1:
      if rel2fvals[0][1][1] == bar[1][1] and matched == []:
         matched = [bar, rel2fvals[0]]

   if matched == []:
      disp_ratio = 0
   else:
      disp_ratio = ((matched[0][1][0] - matched[0][1][1]) / (matched[1][1][0] - matched[1][1][1])) * 100

   return [depths, split_ratio, disp_ratio, shallow_ratio]
##END CYL_COMPUTE_DATA##

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
input_dict = ask_date_GUI()
date = input_dict["date"].date()
date = str(date)
OG_date = date
print(date)
days_B = int(float(input_dict["daysBefore"]))
days_A = int(float(input_dict["daysAfter"]))

# Define figure with plots
fig = plt.figure(figsize = (15,8), layout = 'constrained')
#fig.suptitle(date)
subfigs = fig.subfigures(2, 2, wspace = 0.07)

# Get the directory where the script itself lives
script_dir = os.path.dirname(os.path.abspath(__file__))
script_data_dir = script_dir + '/data_10hPa'

# Glob all .txt files in the same folder
txt_files = glob.glob(os.path.join(script_data_dir, "*.txt"))

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
if not date_index_returner(df,date):
   found = False
else:
   date_l_index, date_u_index = date_index_returner(df, date)
   z = np.array([])
   for i in range(date_l_index, date_u_index): z = np.append(z, df["data"][i])
   found = True

try:
   if found:
      Z = z.reshape((46,180))
   else:
      Z = np.array([])
except ValueError:
   messagebox.showerror("Invalid data entry", "Expected data to have 46 rows (Latitude) and 180 columns (Longitude).")

#------------------------
# Cylindrical Plot
#------------------------
graph_ax = subfigs[0][0].subplots(1, 2, subplot_kw = {"projection" : "3d"})

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0.001, 4, 46)
theta = np.linspace(0, 2*np.pi, 180)
R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')

# Convert polar to cartesian for 3D plotting
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Plot the surface.
if found:
   graph_ax[0].plot_surface(X, Y, Z, cmap=cm.managua)

# Customize the z axis.
graph_ax[0].set_zlim(27811, 31668) #min = 27811.846 , max = 31667.023
graph_ax[0].view_init(elev = 15, azim = -10)

#------------------------
# Grid Plot
#------------------------

graph_ax[1].set_xlabel("Latitude")
graph_ax[1].set_ylabel("Longitude")
graph_ax[1].view_init(elev = 20, azim = -10)

# Create meshgrid for lats and longs
lats = np.array([90 - 2*i for i in range(0, 46)])
longs = np.array([2*i for i in range(0, 180)])
longs, lats = np.meshgrid(longs, lats)

if found:
   surf = graph_ax[1].plot_surface(lats, longs, Z, cmap = cm.managua, linewidth=0)

# Red line on surface that represents one point
if found:
   x_line = lats[0, :]
   y_line = longs[0, :]
   z_line_on_surface = Z[0, :]
   graph_ax[1].plot(x_line, y_line, z_line_on_surface, color='red', linewidth = 3)

# Customize the z axis.
graph_ax[1].set_zlim(27811, 31668) #min = 27811.846 , max = 31667.023

#------------------------
# Outputs:
#------------------------
# Gather the correct dates and organize them
prior_dates = n_days_prior(date, days_B)
after_dates = n_days_after(date, days_A)
all_dates = list((set(prior_dates).union([date])).union(after_dates))
sorted_dates = sorted(all_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d"))
num_dates = len(sorted_dates)
skips = int(num_dates / 18)

# Start compiling data
missing_dates = []
all_data = []
for date in sorted_dates:
   computed_data = compute_data(date)
   if computed_data != []:
      all_data.append(computed_data)
   else:
      all_data.append([[np.nan, np.nan], np.nan, np.nan])
      missing_dates.append(date)

if missing_dates != []:
   messagebox.showinfo("Invalid data entry", f"The following dates are missing from the data: {missing_dates}.")

all_depths = [item[0] for item in all_data]
first_depths = [depth[0] for depth in all_depths]
second_depths = [depth[1] for depth in all_depths]
split_ratios = [item[1] for item in all_data]
disp_ratios = [item[2] for item in all_data]
   
# Wind speeds
try:
   wind_file = script_dir + "/u1060.txt"
   read_wind_file = pd.read_csv(wind_file, sep='\t')
   neg_wind_dates = []
   NWD_D_data = []      #Negative Wind Dates Displacement data
   NWD_S_data = []      #Negative Wind Dates Split data
   for date in sorted_dates:
      for i in range(len(read_wind_file['DATES'])):
         if date == read_wind_file['DATES'][i]:
            if read_wind_file['SPEEDS'][i] < 0:
               neg_wind_dates.append(date)
               index = sorted_dates.index(date)
               NWD_D_data.append(disp_ratios[index])
               NWD_S_data.append(split_ratios[index])
except:
   neg_wind_dates = []
   NWD_D_data = []      #Negative Wind Dates Displacement data
   NWD_S_data = []      #Negative Wind Dates Split data

## -- Split Ratio Plot -- ##
split_ax = subfigs[0][1].subplots(1, 1)
split_ax.plot(sorted_dates, split_ratios, marker = "o", linestyle = "dashed")
split_ax.scatter(neg_wind_dates, NWD_S_data, s = 60, marker = "s", c = "red", zorder  = 3)
split_ax.set_ylim(0,105)
split_ax.set_ylabel("Split Score")
split_ax.tick_params(axis = 'x', labelrotation = 60)
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
disp_ax.plot(sorted_dates, disp_ratios, marker = "o", linestyle = "dashed")
disp_ax.scatter(neg_wind_dates, NWD_D_data, s = 60, marker = "s", c = "red", zorder  = 3)
disp_ax.set_ylim(0,105)
disp_ax.set_ylabel("Displacement Score")
disp_ax.tick_params(axis = 'x', labelrotation = 60)
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

## -- Height Plot -- ##
bar_ax = subfigs[1][0].subplots(1, 1)
bar_ax.bar(sorted_dates, first_depths, color = "orange", label = "Largest")
bar_ax.bar(sorted_dates, second_depths, color = "steelblue", label = "2nd Largest")
top_depth = max(3500, max(first_depths))
bar_ax.set_ylim(0, top_depth)
bar_ax.set_ylabel("Height")
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

#-----------
# Show plots
#-----------
plt.show()
