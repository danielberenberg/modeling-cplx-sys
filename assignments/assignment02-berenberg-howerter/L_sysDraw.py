'''
    Plotting our L-system generated string:
                : shows plot and saves figure to directory:
                '../assignments/assignment02-berenberg-howerter/figs/L-system_plots/'
'''
import sys
import numpy as np
import math
import glob

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import colors
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Load in L-system string and rules:
with open('L_sysString.txt','r') as txt_file:
    L_sys_string = txt_file.readlines()
print('The final L_system string is {} characters long...'.format(len(list(L_sys_string))))
rules = np.load('L_sysRules.npy'),item()


# ask for drawing rules/parameters:
draw_rules = {}
for key in rules.keys():
    draw_rules[key] = float(input('How long should the lines be when drawing: "{}"?'.format(key)))
draw_rules['+'] = float(input('How many degrees should I turn to the left?'))
draw_rules['-'] = float(input('How many degrees should I turn to the right?'))


# determine starting point and initial angle:
init_cond = (0,0,math.pi/2))

# Find number of plot for filename:
plotfiles = glob.glob('../assignments/assignment02-berenberg-howerter/figs/L-system_plots/*')
plot_counter = np.max([int(re.search('(?<=-)\d+',p).group(0)) for p in plotfiles])
plot_counter += 1


# initiate starting point and initial angle
x = init_cond[0]
y = init_cond[1]
ang = init_cond[2]

stack = {'x':[x],'y':[y],'ang':[ang]}
cntr = 0

steps = list(L_sys_string)

F_lines = []
G_lines = []

for i,step in enumerate(steps):
    if step == 'F':
        next_x = x + lenF * math.cos(ang)
        next_y = y + lenF * math.sin(ang)
        F_lines.append([[x,y],[next_x,next_y]])
        x = next_x
        y = next_y

    elif step == 'G':
        next_x = x + lenG * math.cos(ang)
        next_y = y + lenG * math.sin(ang)
        G_lines.append([[x,y],[next_x,next_y]])
        x = next_x
        y = next_y

    elif step == '+':
        ang = ang + angP

    elif step == '-':
        ang = ang - angM

    elif step == '[':
        stack['x'].append(x)
        stack['y'].append(y)
        stack['ang'].append(ang)
        cntr += 1

    elif step == ']':
        x = stack['x'][cntr]
        y = stack['y'][cntr]
        ang = stack['ang'][cntr]
        cntr = cntr - 1

    else:
        print('error, unrecognized character!')
    #if gif_it:
        #plt.savefig('../assignments/assignment02-berenberg-howerter/figs/L-system_plots/gif-L-sys-{:02d}/step-{:02d}.pdf'.format(plot_counter,i),bbox_inches='tight')

Flc = mc.LineCollection(F_lines,colors='seagreen',linewidths=2)
Glc = mc.LineCollection(G_lines,colors='lightseagreen',linewidths=1)
fig, ax = plt.subplots(figsize=(8,13))
ax.add_collection(Flc)
ax.add_collection(Glc)
ax.autoscale()
ax.margins(0.1)
plt.title('Plot Number {:03d} - [{}]:[{}] - iterations = {}'.format(plot_counter,rules.keys(),rules.values(),reps),fontsize=15)
plt.savefig('../assignments/assignment02-berenberg-howerter/figs/L-system_plots/L-sys-{:03d}.pdf'.format(plot_counter),bbox_inches='tight')
plt.show()
