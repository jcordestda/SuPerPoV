import matplotlib.pyplot as plt              #for plotting
import ast
import numpy as np

#-----Import-----#
with open("Sp_Disp_Def_Sign_date.txt", "r") as f:
    data = f.read()

array = ast.literal_eval(data)


Twenty = []
Fourty = []
Sixty = []
Eighty = []
Hundred = []

#-----Plot-----#
majors = [item for item in array if item[1] != []]
# X,Y coordinates respectively
dispPoints = []
splitPoints = []
for item in majors:
   item_index = array.index(item)           # This is assuming that no two elements in array will ever be the same
   defs = item[1]
   if 'U&T' in defs or 'CP07' in defs or 'U6090' in defs or 'U65' in defs: length = 20
   if 'MOM' in defs: length = 30
   if 'ZPOL' in defs or 'EOFU' in defs: length = 30
   temp = array[item_index: item_index + length + 1]
   temp_sharp_disp = [item[0][0] for item in temp]
   max_disp = max(temp_sharp_disp)
   dispPoints.append(max_disp)
   # Find matching split ratio
   for temp_item in temp:
      if temp_item[0][0] == max_disp:
         max_split = temp_item[0][1]
         splitPoints.append(max_split)
         break

   if max_disp >= 20 or max_split >= 20: Twenty.append( (max_disp , max_split) )
   if max_disp >= 40 or max_split >= 40: Fourty.append( (max_disp , max_split) )
   if max_disp >= 60 or max_split >= 60: Sixty.append( (max_disp , max_split) )
   if max_disp >= 80 or max_split >= 80: Eighty.append( (max_disp , max_split) )
   if max_disp >= 100 or max_split >= 100: Hundred.append( (max_disp , max_split) )

# Labels
labels_raw = [item[1] for item in majors]
# Label categories
processed_labels = []
for names in labels_raw:
   if len(names) == 1:
      processed_labels.append(names[0])
   else:
      processed_labels.append("MULTI")

unique_labels = sorted(set(processed_labels))
label_to_num = {label: i for i, label in enumerate(unique_labels)}
colors = [label_to_num[label] for label in processed_labels]

fig = plt.figure(figsize = (14,7), layout = 'constrained')
ax = fig.add_subplot(111)
scatter = ax.scatter(dispPoints, splitPoints, c = colors, cmap='tab10')
ax.set_xlim(0,105)
ax.set_ylim(0,105)
ax.set_xlabel("Displacement Score")
ax.set_ylabel("Split Score")

# Stuff for Legend
label_colors = [
    [0.58039216, 0.40392157, 0.74117647, 1.0],
    [0.09019608, 0.74509804, 0.81176471, 1.0],
    [0.89019608, 0.46666667, 0.76078431, 1.0],
    [0.17254902, 0.62745098, 0.17254902, 1.0],
    [0.12156863, 0.46666667, 0.70588235, 1.0],
    [0.7372549, 0.74117647, 0.13333333, 1.0],
]
labels = ["MULTI", "ZPOL", "U6090", "MOM", "EOFU", "U65"]
handles = [
    plt.Line2D([], [], marker='o', linestyle='', markersize=10, color=c)
    for c in label_colors
]
plt.legend(handles, labels, title="Definitions")

ax.grid(True)
plt.show()
