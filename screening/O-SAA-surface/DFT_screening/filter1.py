#!/usr/bin/env python3
import os
from ase.io import read
from ase import Atoms
from ase.build import minimize_rotation_and_translation
import numpy as np
import sys


def getdis(f1,f2,f3,f4):
#    f1=sys.argv[1]  # CONTCAR    source 
#    f2=sys.argv[2]  # POSCAR ; POSCAR.bak   refer
#    f3=sys.argv[3]  # POSCAR ; POSCAR.bak   target
#    f4=sys.argv[4]  # metal ; O ; metal_O ; doped_metal
    surf1=read(f1)
    surf2=read(f2)
    surf3=read(f3)
    
    #align
    minimize_rotation_and_translation(surf3,surf1)
    
    idss=surf1.constraints[0].index
    totalindex=list(range(len(surf1)))
    relaxindex=np.setdiff1d(totalindex,idss)
    indexs=[]
    if f4=='metal':
        indexs=[i for i in relaxindex if surf1[i].symbol!='O']
    elif f4=='O':
        indexs=[i for i in relaxindex if surf1[i].symbol=='O']
    elif f4=='doped_metal':
        indexs=[i for i in relaxindex if surf1[i].symbol not in ['O','Ag','Cu','Au']]
    else:
        indexs=relaxindex.copy()
    
    sum=0
    surfn=surf1.copy()
    surfn.extend(surf2)
    for i in indexs:
        d=surfn.get_distance(i,i+len(surf1),mic=1)
        sum+=d*d
    return sum/len(indexs)

with open('../pred_lt_0.5','r') as fp:
    data=fp.readlines()
ene={}
for i in data:
    tmp=i.strip().split()
    key=tmp[0]+'_'+tmp[1]
    if key in ene:
        ene[key].append([tmp[0],int(tmp[1]),int(tmp[2]),int(tmp[3]),float(tmp[4])])
    else:
        ene[key]=[[tmp[0],int(tmp[1]),int(tmp[2]),int(tmp[3]),float(tmp[4])]]

   
with open('names','r') as fp:
    names=fp.readlines()
names=[i.strip() for i in names]


fp1=open('filter1','w')
fp2=open('move1','w')
for name in names:
    if os.path.exists(name+'/POSCAR.bak'):
        d_o=getdis(name+'/CONTCAR',name+'/POSCAR.bak',name+'/POSCAR.bak','O')
        d_m=getdis(name+'/CONTCAR',name+'/POSCAR.bak',name+'/POSCAR.bak','doped_metal')
    else:
        d_o=getdis(name+'/CONTCAR',name+'/POSCAR',name+'/POSCAR','O')
        d_m=getdis(name+'/CONTCAR',name+'/POSCAR',name+'/POSCAR','doped_metal')
    if d_o<0.5 and d_m<0.5:
        fp2.write('{0:>50s} {1:>20.5f} {2:>20.5f}\n'.format(name,d_o,d_m))
        for tmp in ene[name]:
            fp1.write('{0:<50s} {1:>5d} {2:>5d} {3:>5d} {4:>20.6f}\n'.format(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]))        




fp1.close()
fp2.close()
