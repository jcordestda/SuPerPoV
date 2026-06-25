#Outputs GUI that will ask for a date. Then will ask for how many days before that date, and how many days after that date you want to collect data for
#Both will compare the Cubical Complex and Extended Persistence
#Will plot for 10hPa, 50hPa, and 100hPa
#Version 1.0.1: Changed the average heights on the bars back to the average below 10%
#Version 1.0.2: Changed the scoring from 0-100 to 0-1

#---------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------

import os                                    #for file import
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

def compute_data(date, txt_files):
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
   if len(rel2fvals) > 1:
      split_ratio = ((rel2fvals[1][1][0] - rel2fvals[1][1][1]) / depths[0])
   else:
      split_ratio = 0
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
      disp_ratio = ((matched[0][1][0] - matched[0][1][1]) / (matched[1][1][0] - matched[1][1][1]))

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
# Gather information from GUI
input_dict = ask_date_GUI()
date = input_dict["date"].date()
date = str(date)
OG_date = date
print(date)
days_B = int(float(input_dict["daysBefore"]))
days_A = int(float(input_dict["daysAfter"]))

# Define figure with plots
fig = plt.figure(figsize = (15,8), layout = 'constrained')
fig.suptitle(date)
subfigs = fig.subfigures(1, 2, wspace = 0.07)

# Get the directory where the script itself lives
script_dir = os.path.dirname(os.path.abspath(__file__))
script_data_dir10 = script_dir + '/data_10hPa'
script_data_dir50 = script_dir + '/data_50hPa'
script_data_dir100 = script_dir + '/data_100hPa'

# Glob all .txt files in the same folder
txt_files10 = glob.glob(os.path.join(script_data_dir10, "*.txt"))
txt_files50 = glob.glob(os.path.join(script_data_dir50, "*.txt"))
txt_files100 = glob.glob(os.path.join(script_data_dir100, "*.txt"))

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
missing_dates10 = []
missing_dates50 = []
missing_dates100 = []
all_data10 = []
all_data50 = []
all_data100 = []
for date in sorted_dates:
   # 10
   computed_data10 = compute_data(date, txt_files10)
   if computed_data10 != []:
      all_data10.append(computed_data10)
   else:
      all_data10.append([[np.nan, np.nan], np.nan, np.nan])
      missing_dates10.append(date)
   # 50
   computed_data50 = compute_data(date, txt_files50)
   if computed_data50 != []:
      all_data50.append(computed_data50)
   else:
      all_data50.append([[np.nan, np.nan], np.nan, np.nan])
      missing_dates50.append(date)
   # 100
   computed_data100 = compute_data(date, txt_files100)
   if computed_data100 != []:
      all_data100.append(computed_data100)
   else:
      all_data100.append([[np.nan, np.nan], np.nan, np.nan])
      missing_dates100.append(date)

if missing_dates10 != [] or missing_dates50 != [] or missing_dates100 != []:
   messagebox.showinfo("Invalid data entry", f"The following dates are missing from the data:\n From 10 hPa, {missing_dates10}\n From 50 hPa, {missing_dates50}\n From 100 hPa, {missing_dates100}")

all_depths10 = [item[0] for item in all_data10]
all_depths50 = [item[0] for item in all_data50]
all_depths100 = [item[0] for item in all_data100]
first_depths10 = [depth[0] for depth in all_depths10]
second_depths10 = [depth[1] for depth in all_depths10]
first_depths50 = [depth[0] for depth in all_depths50]
second_depths50 = [depth[1] for depth in all_depths50]
first_depths100 = [depth[0] for depth in all_depths100]
second_depths100 = [depth[1] for depth in all_depths100]
split_ratios10 = [item[1] for item in all_data10]
split_ratios50 = [item[1] for item in all_data50]
split_ratios100 = [item[1] for item in all_data100]
disp_ratios10 = [item[2] for item in all_data10]
disp_ratios50 = [item[2] for item in all_data50]
disp_ratios100 = [item[2] for item in all_data100]

# Wind speeds
try:
   wind_file = script_dir + "/u1060.txt"
   read_wind_file = pd.read_csv(wind_file, sep='\t')
   neg_wind_dates = []
   NWD_D_data10 = []       #Negative Wind Dates Displacement data at 10 hPa
   NWD_D_data50 = []       #Negative Wind Dates Displacement data at 50 hPa
   NWD_D_data100 = []      #Negative Wind Dates Displacement data at 100 hPa
   NWD_S_data10 = []       #Negative Wind Dates Split data at 10 hPa
   NWD_S_data50 = []       #Negative Wind Dates Split data at 50 hPa
   NWD_S_data100 = []      #Negative Wind Dates Split data at 100 hPa
   for date in sorted_dates:
      for i in range(len(read_wind_file['DATES'])):
         if date == read_wind_file['DATES'][i]:
            if read_wind_file['SPEEDS'][i] < 0:
               neg_wind_dates.append(date)
               index = sorted_dates.index(date)
               NWD_D_data10.append(disp_ratios10[index])
               NWD_D_data50.append(disp_ratios50[index])
               NWD_D_data100.append(disp_ratios100[index])
               NWD_S_data10.append(split_ratios10[index])
               NWD_S_data50.append(split_ratios50[index])
               NWD_S_data100.append(split_ratios100[index])
except:
   neg_wind_dates = []
   NWD_D_data10 = []       #Negative Wind Dates Displacement data at 10 hPa
   NWD_D_data50 = []       #Negative Wind Dates Displacement data at 50 hPa
   NWD_D_data100 = []      #Negative Wind Dates Displacement data at 100 hPa
   NWD_S_data10 = []       #Negative Wind Dates Split data at 10 hPa
   NWD_S_data50 = []       #Negative Wind Dates Split data at 50 hPa
   NWD_S_data100 = []      #Negative Wind Dates Split data at 100 hPa

scatter_ax = subfigs[1].subplots(2, 1)
scatter_ax[0].plot(sorted_dates, split_ratios10, marker = "o", color = 'blue', linestyle = "dashed", label = "10 hPa")
scatter_ax[0].scatter(neg_wind_dates, NWD_S_data10, marker = "s", c = "red", zorder  = 3)
scatter_ax[0].plot(sorted_dates, split_ratios50, marker = "o", color = 'green', linestyle = "dashed", label = "50 hPa")
scatter_ax[0].scatter(neg_wind_dates, NWD_S_data50, marker = "s", c = "red", zorder  = 3)
scatter_ax[0].plot(sorted_dates, split_ratios100, marker = "o", color = 'purple', linestyle = "dashed", label = "100 hPa")
scatter_ax[0].scatter(neg_wind_dates, NWD_S_data100, marker = "s", c = "red", zorder  = 3)
scatter_ax[0].set_ylim(0,1.05)
scatter_ax[0].set_ylabel("Split Score")
scatter_ax[0].tick_params(axis = 'x', labelrotation = 60)
for label in scatter_ax[0].get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")
        break
scatter_ax[0].legend()
if skips > 0:
   split_labels = scatter_ax[0].get_xticklabels()
   for i, label in enumerate(split_labels):
      if i % skips != 0:
         label.set_visible(False)

scatter_ax[1].plot(sorted_dates, disp_ratios10, marker = "o", color = 'blue', linestyle = "dashed", label = "10 hPa")
scatter_ax[1].scatter(neg_wind_dates, NWD_D_data10, marker = "s", c = "red", zorder  = 3)
scatter_ax[1].plot(sorted_dates, disp_ratios50, marker = "o", color = 'green', linestyle = "dashed", label = "50 hPa")
scatter_ax[1].scatter(neg_wind_dates, NWD_D_data50, marker = "s", c = "red", zorder  = 3)
scatter_ax[1].plot(sorted_dates, disp_ratios100, marker = "o", color = 'purple', linestyle = "dashed", label = "100 hPa")
scatter_ax[1].scatter(neg_wind_dates, NWD_D_data100, marker = "s", c = "red", zorder  = 3)
scatter_ax[1].set_ylim(0,1.05)
scatter_ax[1].set_ylabel("Displacement Score")
scatter_ax[1].tick_params(axis = 'x', labelrotation = 60)
for label in scatter_ax[1].get_xticklabels():
    if label.get_text() == OG_date: 
        label.set_fontweight("bold")  
        break
if skips > 0:
   disp_labels = scatter_ax[1].get_xticklabels()
   for i, label in enumerate(disp_labels):
      if i % skips != 0:
         label.set_visible(False)

scatter_ax[1].legend()
bar_ax = subfigs[0].subplots(3, 1)
bar_ax[0].bar(sorted_dates, first_depths10, color = "orange", label = "Largest")
bar_ax[0].bar(sorted_dates, second_depths10, color = "steelblue", label = "2nd Largest")
top_depth = max(3500, max(first_depths10))
bar_ax[0].set_ylim(0, top_depth)
bar_ax[0].set_ylabel("Life Span @ 10 hPa")
bar_ax[0].tick_params(axis = 'x', labelrotation = 30)
bar_ax[0].axhline(y = 2387.7611 + 518.5322, linestyle=':', color='black')
bar_ax[0].axhline(y = 2387.7611, linestyle='--', color = 'black', label = 'Average')
bar_ax[0].axhline(y = 2387.7611 - 518.5322, linestyle=':', color = 'black')
bar_ax[0].legend()
for label in bar_ax[0].get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")  
        break
if skips > 0:
   labels = bar_ax[0].get_xticklabels()
   for i, label in enumerate(labels):
      if i % skips != 0:
         label.set_visible(False)
bar_ax[1].bar(sorted_dates, first_depths50, color = "orange", label = "Largest")
bar_ax[1].bar(sorted_dates, second_depths50, color = "steelblue", label = "2nd Largest")
top_depth = max(3500, max(first_depths50))
bar_ax[1].set_ylim(0, top_depth)
bar_ax[1].set_ylabel("Life Span @ 50 hPa")
bar_ax[1].tick_params(axis = 'x', labelrotation = 30)
bar_ax[1].legend()
for label in bar_ax[1].get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")  
        break
if skips > 0:
   labels = bar_ax[1].get_xticklabels()
   for i, label in enumerate(labels):
      if i % skips != 0:
         label.set_visible(False)
bar_ax[2].bar(sorted_dates, first_depths100, color = "orange", label = "Largest")
bar_ax[2].bar(sorted_dates, second_depths100, color = "steelblue", label = "2nd Largest")
top_depth = max(3500, max(first_depths100))
bar_ax[2].set_ylim(0, top_depth)
bar_ax[2].set_ylabel("Life Span @ 100 hPa")
bar_ax[2].tick_params(axis = 'x', labelrotation = 30)
bar_ax[2].legend()
for label in bar_ax[2].get_xticklabels():
    if label.get_text() == OG_date:
        label.set_fontweight("bold")
        break
if skips > 0:
   labels = bar_ax[2].get_xticklabels()
   for i, label in enumerate(labels):
      if i % skips != 0:
         label.set_visible(False)
#-----------
# Show plots
#-----------
#plt.tight_layout()
plt.show()
	 
