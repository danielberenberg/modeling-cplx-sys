'''
    Plotting our L-system generated string:
                : shows plot and saves figure to directory:
                '../assignments/assignment02-berenberg-howerter/figs/L-system_plots/'
'''
import sys
import os
import numpy as np
import math
import glob
import re

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
    L_sys_string = txt_file.read()
print('The final L_system string is {} characters long...'.format(len(list(L_sys_string))))
rules = np.load('L_sysRules.npy').item()


# ask for drawing rules/parameters:
draw_dim = {}
for key in rules.keys():
    draw_dim[key] = float(input('How long should the lines be when drawing: "{}"? '.format(key)))
draw_dim['+'] = float(input('How many degrees should I turn to the left? '))* math.pi /180
draw_dim['-'] = -float(input('How many degrees should I turn to the right? '))* math.pi /180


# determine starting point and initial angle:
init_cond = (0,0,math.pi/2)

# Find number of plot for filename:
plotfiles = glob.glob('figs/L-system_plots/*')
plot_counter = np.max([int(re.search('(?<=-)\d+',p).group(0)) for p in plotfiles])
plot_counter += 1

reps = int(sys.argv[1])

# initiate starting point and initial angle
x = init_cond[0]
y = init_cond[1]
ang = init_cond[2]

stack = {'x':[x],'y':[y],'ang':[ang]}
cntr = 0

steps = list(L_sys_string)

lines = {dim:[] for dim in rules.keys()}
trunk = float(input('What thickness do you want to start with? (between 1 & {}) '.format(reps)))/reps
thickness = {dim:[] for dim in rules.keys()}
thkness_cntr = {dim:0 for dim in rules.keys()}
last_step = ''
rand_len = input('Would you like to randomize the length? (Y/N) ')
rand_ang = input('Would you like to randomize the angles? (Y/N) ')
turncount = 1

for i,step in enumerate(steps):
    if (last_step != step) and last_step in thickness:
        thickness[last_step].extend(thkness_cntr[last_step]*[thkness_cntr[last_step]*trunk])
        thkness_cntr[last_step] = 0

    if step in list(rules.keys()):
        if rand_len == 'Y':
            mult = np.absolute(np.random.normal(loc=0.0,scale=trunk/2))
        else:
            mult = 1
        next_x = x + draw_dim[step] * mult * math.cos(ang)
        next_y = y + draw_dim[step] * mult * math.sin(ang)
        lines[step].append([[x,y],[next_x,next_y]])
        x = next_x
        y = next_y
        thkness_cntr[step] += 1

    elif step.isnumeric():
        turncount = int(step)

    elif step in ['+','-']:
        if rand_ang == 'Y':
            rang = np.random.normal(loc=0.0,scale=math.pi/12)
        else:
            rang = 0
        for i in range(turncount):
            ang = ang + draw_dim[step] + rang
        turncount = 1

    elif step == '[':
        stack['x'].append(x)
        stack['y'].append(y)
        stack['ang'].append(ang)
        cntr += 1

    elif step == ']':
        x = stack['x'][cntr]
        y = stack['y'][cntr]
        ang = stack['ang'][cntr]
        stack['x'].pop(-1)
        stack['y'].pop(-1)
        stack['ang'].pop(-1)
        cntr = cntr - 1

    else:
        print('error, unrecognized character!')

    last_step = step
    #if gif_it:
        #plt.savefig('../assignments/assignment02-berenberg-howerter/figs/L-system_plots/gif-L-sys-{:02d}/step-{:02d}.pdf'.format(plot_counter,i),bbox_inches='tight')

c = ['darkseagreen','seagreen','lightseagreen','lightgreen']

fig, ax = plt.subplots(figsize=(10,14))

for i,dim in enumerate(lines.keys()):
    lc = mc.LineCollection(lines[dim],colors=c[i],linewidths=thickness[dim])
    ax.add_collection(lc)

ax.autoscale()
ax.margins(0.1)
plt.axis('off')
plt.title(r'Plot Number {:03d} - {}:{} - iterations = {}'.format(plot_counter,list(rules.keys()),list(rules.values()),reps),fontsize=15)
plt.savefig('figs/L-system_plots/L-sys-{:03d}.pdf'.format(plot_counter),bbox_inches='tight')
#plt.show()

os.system('open figs/L-system_plots/L-sys-{:03d}.pdf'.format(plot_counter))
