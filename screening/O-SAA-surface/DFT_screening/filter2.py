#!/usr/bin/env python3
import os
from ase.io import read
from ase import Atoms
from ase.build import minimize_rotation_and_translation
import numpy as np
import sys

def getdis(f1,f2,f3,f4):
    surf1=read(f1)
    surf2=read(f2)
    surf3=read(f3)
    surf1.pop(-1)
    surf1.pop(-1)
    surf1.pop(-1)
    surf1.pop(-1)
    surf1.pop(-1)
    surf2.pop(-1)
    surf2.pop(-1)
    surf2.pop(-1)
    surf2.pop(-1)
    surf2.pop(-1)
    surf3.pop(-1)
    surf3.pop(-1)
    surf3.pop(-1)
    surf3.pop(-1)
    surf3.pop(-1)


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


with open('filter1','r') as fp:
    names=fp.readlines()

fp1=open('filter2','w')
fp2=open('move2','w')
for name in names:
    surfname,site,Oid,Mid,ea_gbdt=name.strip().split()
    path=surfname+'_'+site+'/'+Oid+'_'+Mid+'/is'
    if os.path.exists(path+'/POSCAR.bak'):
        d_o=getdis(path+'/CONTCAR',path+'/POSCAR.bak',path+'/POSCAR.bak','O')
        d_m=getdis(path+'/CONTCAR',path+'/POSCAR.bak',path+'/POSCAR.bak','doped_metal')
    else:
        d_o=getdis(path+'/CONTCAR',path+'/POSCAR',path+'/POSCAR','O')
        d_m=getdis(path+'/CONTCAR',path+'/POSCAR',path+'/POSCAR','doped_metal')

    surf=read(path+'/CONTCAR')
    cindex=[j.index for j in surf if j.symbol=='C'][0]
    mindex=[j.index for j in surf if j.symbol not in  ['C','H','O']]
    discm=[[j,surf.get_distance(cindex,j,mic=1)] for j in mindex]
    discm2=sorted(discm,key=lambda x:x[1])
    res=discm2[0]
    disom=surf.get_distance(int(Oid),int(Mid),mic=1)
    if res[0]==int(Mid) and d_o<0.5 and d_m<0.5 and disom<4.0:
        fp1.write('{0:<50s} {1:>5d} {2:>5d} {3:>5d} {4:>20.6f}\n'.format(surfname,int(site),int(Oid),int(Mid),float(ea_gbdt)))
        fp2.write('{0:>50s} {1:>20.5f} {2:>20.5f} {3:>20.5f}\n'.format(surfname+'_'+site+'/'+Oid+'_'+Mid,d_o,d_m,disom))


fp1.close()
fp2.close()
