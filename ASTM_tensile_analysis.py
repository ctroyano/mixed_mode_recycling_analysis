#from asyncio.windows_utils import pip
import math as m
import pandas as pd
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import argparse
from glob import glob
from scipy import interpolate
import scipy.ndimage.filters as scf
import os
import IPython


##################### CHANGE HERE ########################

# folder with all the .TRA files
data_folder_directory = './'
sample_names_TRA = ''
sample_details_file_name = 'data_details_copy.csv'

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

### DETAILS ON THE DATA
separator = '/'
samples_details = pd.read_csv(f'{data_folder_directory}{sample_details_file_name}', header=0)

data_folder = data_folder_directory + 'CSVfiles/'
save_folder = data_folder_directory + 'data_files/'
analyzed_folder = data_folder_directory + 'analyzed_data/'
averages_folder = data_folder_directory + 'averages/'

try:
    os.mkdir(data_folder[:-1])
    os.mkdir(save_folder[:-1])
    os.mkdir(analyzed_folder[:-1])
    os.mkdir(averages_folder[:-1])
except FileExistsError:
    pass

sample_name = samples_details['sample'].values
thickness = samples_details['thickness'].values
width = samples_details['width'].values
file_name = samples_details['sample'].values


if True:
    names = sorted(glob(f'{data_folder_directory}TRAfiles/*.TRA'))
    errors = {}
    for i in names:
        name = i.split(separator)[-1]
        name = name.split('.')[0]
        try:
            with open(f'{data_folder}{name}.csv','w')as f:
                with open(i, 'r') as d:
                    lines = d.readlines()
                    f.writelines(lines[8:])
                d.close()
            f.close()
        except Exception as e:
            errors[i] = e 

### GAUGE LENGTH
GL=22


for i,n in enumerate(file_name):
    filen = data_folder+n+'.csv'
    data =  pd.read_csv(filen, header=None).values
    time=data[:,0]
    distance=data[:,1]
    force=data[:,2]

    ### CONVERSION TO STRESS AND STRAIN
    strain = distance/GL
    stress = force/(thickness[i]*width[i]*10**-6)

    ### DROPPING VALUES WHERE STRAIN IS CONSTANT
    # => begining acceleration and end slow down to full stop
    nonzero = np.where(strain[1:]-strain[:-1]>0)[0]
    strain = strain[nonzero]
    stress = stress[nonzero]

    f = interpolate.interp1d(strain,stress)

    N=8000
    x = np.linspace(strain.min(),strain.max(),N)
    y = f(x)

    ### MODIFY START OF CURVE

    # Computing dy, dx and derivative
    dy = np.diff(y)
    ddy = np.diff(dy)

    # Smoothed data
    sigma = 30
    dys = scf.gaussian_filter(dy, sigma)
    ddys = np.diff(dys)

    # Find second derivative zero

    zero = np.where(np.bitwise_and(ddys[:-1]*ddys[1:]<0,ddys[:-1]>ddys[1:]))[0][0]
    if x[zero]>0.2 or y[zero]>400000:
        zero=0
    x2 = x[zero:].copy() - x[zero]
    y2 = y[zero:].copy()
    dy2 = dy[zero:].copy()
    dys2 = dys[zero:].copy()


    ### ANALYSIS OF DATA
    # Young's modulus
    convol = np.exp(-np.arange(len(x2))**2/sigma**2)
    convol = convol / np.sum(convol)
    E = (-np.sum(y2*convol) + np.sum(y2[1:]*convol[:-1])) / (x2[1]-x2[0])

    # Elongation at break
    index = m.floor(len(dys2)/2)
    indices = np.where(np.abs(dy2)>10**4)[0]

    #indices = indices[indices>len(indices)/2]
    x_elong = indices[0]
    add = 10
    elongation_break = x2[x_elong]

    # Toughness calculation
    toughness = np.sum(y2[:x_elong+add]+dys2[:x_elong+add]/2) * (x[1]-x[0])

    # Maximum tensile strength
    max_strength = np.amax(stress)

    # Modulus 50%, 100%, 200%
    if elongation_break > 0.5:
        modulus50 = y2[np.where(x2>0.5)[0][0]]
    else:
        modulus50 = 0
    if elongation_break > 1:
        modulus100 = y2[np.where(x2>1)[0][0]]
    else:
        modulus100 = 0
    if elongation_break > 2:
        modulus200 = y2[np.where(x2>2)[0][0]]
    else:
        modulus200 = 0


    ### SAVING DATA
    tensiledata = {}
    tensiledata['elongation'] = x2[:x_elong+add]
    tensiledata['strain(MPa)'] = y2[:x_elong+add]/10**6
    td = pd.DataFrame.from_dict(tensiledata)
    td.to_csv(analyzed_folder+str(sample_name[i])+'_stress_strain.csv', index = False)

    analysis_data = {}
    analysis_data['Young moldulus (Pa)'] = [E]
    analysis_data['elongation at break (%)'] = [elongation_break*100]
    analysis_data['toughness (Pa)'] = [toughness]
    analysis_data['maximum strength (Pa)'] = [max_strength]
    analysis_data['Modulus at 50%'] = [modulus50]
    analysis_data['Modulus at 100%'] = [modulus100]
    analysis_data['Modulus at 200%'] = [modulus200]
    ad = pd.DataFrame.from_dict(analysis_data)
    ad.to_csv(analyzed_folder+n+'_detailed_data.csv', index = False)


    plt.plot(x-x[zero],y/10**6)
    plt.plot(x2[:x_elong+add],y2[:x_elong+add]/10**6)
    plt.plot(x2[:1000],y2[0]/10**6+E*x2[:1000]/10**6,',')
    plt.plot(elongation_break,max_strength/10**6,'+')
    plt.plot(0.5,modulus50/10**6,'x')
    plt.plot(1,modulus100/10**6,'x')
    plt.plot(2,modulus200/10**6,'x')
    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title(f'{sample_name[i]}')
    plt.savefig(analyzed_folder+n+'_stress_strain.png', format='png')
    plt.close()



