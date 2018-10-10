'''
    Set up and run L-system scripts
'''
import os
import sys
import numpy as np

# option to use auto set-up:
YN = input('Would you like to build your own system (or use the auto-set-up)? (Y/N) ')

if YN == 'Y':
    # ask for axiom:
    axiom = input('What is the axiom of your L-system? ')

    # ask for the rules:
    rules = {}
    key1 = str(input('What is the first character you want to make a rule for? '))
    rules[key1] = str(input('What would you like "{}" to map to? '.format(key1)))
    key2 = str(input('What is the second character you want to make a rule for? '))
    rules[key2] = str(input('What would you like "{}" to map to? '.format(key2)))
    more_rules = int(input('How many more rules do you want to add?'))
    for i in range(more_rules):
        k = str(input('What is the next character you want to make a rule for? '))
        rules[k] = str(input('What would you like "{}" to map to? '.format(k)))
else:
    axiom = 'G'
    rules = {'G':'F[+G][-G]F[+G][-G]FG','F':'FF'}

# save the rules:
np.save('L_sysRules.npy',rules)

# ask for number of iterations:
reps = int(input('How many iterations of the L-system would you like to perform? '))

# call L-system build script:
print('Building L-system...')
os.system('python L_sysExpand.py {} {}'.format(axiom, reps))

# call L-system plotting script:
print('Drawing L-system...')
os.system('python L_sysDraw.py {}'.format(reps))
