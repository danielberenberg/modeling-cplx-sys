############################# IMPORTS #############################
import numpy as np
import pickle
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc

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
#for V in V_systems:
#    happ_dist_files[V] = glob.glob('ElectoralProcesses/{}/*'.format(V))
happ_dist_files['general'] = glob.glob('ElectoralProcesses/general/*')

transparencies = [1,2,3,4,5,6,7]
happ_avgs = {V:{T:[] for T in transparencies} for V in happ_dist_files}
full_dists = {V:{T:[] for T in transparencies} for V in happ_dist_files}
for Vsys in happ_dist_files:
    for t in transparencies:
        print('_T{:02}_'.format(t))
        for fname in [f for f in happ_dist_files[Vsys] if '_T{:02}_'.format(t) in f]:
            with open(fname,'rb') as file:
                diss_dist = pickle.load(file)
                # calculate happines from dissatisfaction:
                happ_dist = [(2-diss)/2 for diss in diss_dist]
                #happ_dist = [(2-diss) for diss in diss_dist]
                # store average happiness & std for every population:
                happ_mean = np.mean(happ_dist)
                happ_avgs[Vsys][t].append(happ_mean)

                # store every happiness score for every population:
                full_dists[Vsys][t] += happ_dist


# Plot candidate transparency vs avg happiness for each voting system:
cpairs = [('darkorange','gold'),('crimson','plum'),('darkslategrey','paleturquoise')]
plt.figure()
for i,V in enumerate(happ_avgs):
    t_avgs = [np.mean(happ_avgs[V][t]) for t in transparencies]
    t_std = [np.std(happ_avgs[V][t]) for t in transparencies]
    plt.scatter(transparencies,t_avgs,
                marker = 'x',color= cpairs[i][0],alpha=0.7,zorder=4)
    plt.plot(transparencies,t_avgs,
             color= cpairs[i][0],alpha=0.7,
             label='Avg. happiness for {} Voting'.format(V_systems2label[V]))
    plt.fill_between(transparencies,t_avgs,[u+1.96*s for u,s in zip(t_avgs,t_std)],
                     alpha=0.5,color= cpairs[i][1])
    plt.fill_between(transparencies,t_avgs,[u-1.96*s for u,s in zip(t_avgs,t_std)],
                     alpha=0.5,color= cpairs[i][1],
                     label='95\% CI for {}'.format(V_systems2label[V]))
plt.ylim(0.5,1)
plt.ylabel('Avg. Happiness over 100 Populations')
plt.xlabel('Transparency \n(number of dimensions visible out of 7)')
plt.legend(loc='best')
plt.title('End-of-Election Happiness vs. Candidate Transparency \nfor each Voting System')
plt.savefig('figs/avghapp_transparency.pdf',bbox_inches='tight')
plt.clf()


# Plot full distributions of happ scores of all populations for each V sys
# One plot for every transparency :O
for t in transparencies:
    plt.figure()
    for i,V in enumerate(full_dists):
        plt.hist(full_dists[V][t],bins = 'auto',
                        color = cpairs[i][1], alpha = 0.5,
                        label = '{}, mean={}'.format(V_systems2label[V],round(np.mean(full_dists[V][t]),2)))
    plt.xlabel('Disagreeability with elected candidate')
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.title('Histograms of Disagreeability for each \nVoting System for Candidate Transparency = {}'.format(t))
    plt.savefig('figs/disagreeability-hist_T{:02}.pdf'.format(t),bbox_inches='tight')
    plt.clf()
