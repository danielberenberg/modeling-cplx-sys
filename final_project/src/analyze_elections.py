############################ IMPORTS #############################
import numpy as np
import pickle
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from bltools import letter_subplots
# make figures better:
font = {'weight':'normal','size':11}
rc('font',**{'family':'serif','serif':['Palatino']})
rc('figure', figsize=(5.5,3.5))
rc('text', usetex=True)
#rc('xtick.major', pad=10) # xticks too close to border!
rc('xtick', labelsize=9)
rc('ytick', labelsize=9)
rc('legend',**{'fontsize':9})

import warnings
warnings.filterwarnings('ignore')
####################################################################

#    get average satisfation for every population
#        for each transparency level, for each voting system
V_systems = ['general','ranked','approval']
V_systems2label = {'general':"Plurality",'ranked':"Ranked-Choice",'approval':"Approval"}
happ_dist_files = {}
#uncomment once all voting systems are run
for V in V_systems:
    happ_dist_files[V] = glob.glob('archive/{}/*'.format(V))

transparencies = [1,2,3,4,5,6,7]
happ_avgs = {V:{T:[] for T in transparencies} for V in happ_dist_files}
full_dists = {V:{T:np.zeros(10000) for T in transparencies} for V in happ_dist_files}
for Vsys in happ_dist_files:
    for t in transparencies:
        print('{}_T{:02}_'.format(Vsys,t))
        for fname in [f for f in happ_dist_files[Vsys] if '_T{:02}_'.format(t) in f]:
            print(fname)
            with open(fname,'rb') as file:
                diss_dist = np.array(pickle.load(file))
                # calculate happines from dissatisfaction:
                happ_dist = [(2-diss)/2 for diss in diss_dist]
                #happ_dist = [(2-diss) for diss in diss_dist]
                # store average happiness & std for every population:
                happ_mean = np.mean(happ_dist)
                happ_avgs[Vsys][t].append(happ_mean)

                # store every happiness score for every population:
                full_dists[Vsys][t] += happ_dist

for v in full_dists:
    for t in full_dists[v]:
        full_dists[v][t] /= 100

cpairs = [('darkorange','gold'),('crimson','plum'),('darkslategrey','paleturquoise')]
# Plot candidate transparency vs avg happiness for each voting system:
if input('Would you like to plot transparency vs avg happiness? (y/Y) ').upper() == 'Y':
    plt.figure()
    for i,V in enumerate(happ_avgs):
        t_avgs = np.array([np.mean(happ_avgs[V][t]) for t in transparencies])
        # computing the standard error instead of std because we're concerned with the
        # range of values the mean of the means can take on
        t_std_err_ci = 1.96*(np.array([np.std(happ_avgs[V][t])for t in transparencies])/10)
        plt.scatter(transparencies,t_avgs,
                    marker='x',color=cpairs[i][0],alpha=0.7,zorder=4)
        plt.plot(transparencies,t_avgs,
                 color=cpairs[i][0],alpha=0.25,
                 label='{}'.format(V_systems2label[V]))

        plt.fill_between(transparencies, t_avgs + t_std_err_ci, t_avgs - t_std_err_ci,
                         alpha=0.2,
                         color=cpairs[i][1])
        #                 [u+1.96*s for u,s in zip(t_avgs,t_std)],
        #                 alpha=0.4,color=cpairs[i][1])
        #plt.fill_between(transparencies,t_avgs,[u-1.96*s for u,s in zip(t_avgs,t_std)],
        #                 alpha=0.4,color=cpairs[i][1],
        #                 label='95\% CI for {}'.format(V_systems2label[V]))
    plt.ylim(0.66,0.7)
    plt.ylabel(r"Avg. Satisfaction")
    plt.xlabel('Opinion Transparency')
    plt.legend(loc='best',frameon=True)
    #plt.title('End-of-Election Happiness vs. Candidate Transparency \nfor each Voting System')
    plt.savefig('figs/avghapp_transparency.pdf',bbox_inches='tight')
    plt.clf()


# Plot full distributions of happ scores of all populations for each V sys
# One plot for every transparency :O
if input('Would you like to plot the full distributions for all pop happiness? (y/Y) ').upper() == 'Y':
    fig, axarr = plt.subplots(3,1, sharex=True)
    letter_subplots(axarr, xoffset=3*[-0.05])
    axarr[0].set_xlim(0.62, 0.72)
    axarr[2].set_xlabel("Satisfaction with elected candidate")
    for (ax,t) in zip(axarr, [1,4,7]):
        for i,V in enumerate(full_dists):
            ax.hist(full_dists[V][t],bins='auto',
                    color=cpairs[i][1], histtype='step',
                    label='{}'.format(V_systems2label[V])) #round(np.mean(full_dists[V][t]),2)))
            ax.set_ylabel('Count')
        #ax.set_title('$T = {}$'.format(t))
        ax.text(0.99,0.05, "$T={}$".format(t), transform=ax.transAxes, ha='right')
    axarr[0].legend(frameon=True, loc='upper left')
    plt.savefig('figs/disagreeability-hist_T{:02}.pdf'.format(t),bbox_inches='tight')
    plt.clf()
