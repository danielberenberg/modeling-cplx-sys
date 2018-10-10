'''
     Build the L-system
'''
import numpy as np
import sys
import re

axiom = sys.argv[1]
reps = sys.argv[2]
rules = np.load('L_sysRules.npy'),item()

def L_system(axiom,rules,reps):
    '''
        L-system implementation function:
            looks through every element in a string, starting with the axiom
            and if a rule exists for that element, it is replaced with whatever the
            rule maps it to.

        args:
            : axiom : type = str, starting point to build off of
            : rules : type = dict, of the rules of the L-sys where key string elements when found
                in an iteration, will be replaced with their value elements in the dict.
            : reps : type = int, number of times to find and replace in the string
        returns:
                a dictionary of:
                'str': the final axiom string
                'rules': the original rules dictionary
                'reps': the number of iterations done
    '''
    for i in range(reps):
        for seed in sorted(rules.keys()):
            axiom = re.sub(seed,rules[seed],axiom) #axiom.replace(seed,rules[seed])
        print(list(axiom[:20]))
        print('...')

    return axiom

final_str = L_system(axiom,rules,reps)

with open('L_sysString.txt','w') as txt_file:
    txt_file.write(final_str)
