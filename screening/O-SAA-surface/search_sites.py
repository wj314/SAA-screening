#!/usr/bin/env python3
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, Structure, Lattice, ReconstructionGenerator
from pymatgen.analysis.adsorption import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from asetools.analysis.coordinationNumbers import coordination_numbers, generateRcDict, generateNeighborList
from ase.io import read as ioread
from ase.visualize import view
from ase.atom import Atom
from mendeleev import element
import json
from ase.visualize import view

source=['Au(111)','Ag(111)','Cu(111)','Au(100)','Ag(100)','Cu(100)','Au(110)','Ag(110)','Cu(110)','Au(211)','Ag(211)','Cu(211)']
subatom={'Au(111)':17,'Ag(111)':17,'Cu(111)':17,'Au(100)':17,'Ag(100)':17,'Cu(100)':17,'Au(110)':17,'Ag(110)':16,'Cu(110)':16,'Au(211)':22,'Ag(211)':22,'Cu(211)':19}
target=['Sc','Ti','V','Cr','Mn','Fe','Co','Zn','Ga','Ge',
        'Y','Zr','Nb','Mo','Tc','Ru','Cd','In','Sn',
        'Hf','Ta','W','Re','Os','Hg','Tl','Pb']



def deter_geo_limit(surfgeo,atomid):
    surf2=surfgeo.copy()
    loc=surf2.get_scaled_positions()[atomid]
    dis=[]
    for atom in surf2:
        if (atom.scaled_position[2]-loc[2])>1e-5 and atom.index!=atomid:
            atom.position[2]=surf2[atomid].position[2]
            dis.append(surf2.get_distance(atomid,atom.index,mic=True))
    dis.sort()
    dis_lt_1=[i for i in dis if i<1.0]
    dis_lt_2=[i for i in dis if i<2.0]
    if len(dis_lt_1)>=1 or len(dis_lt_2)>=2:
        return True
    else:
        return False


def is_same_atoms(geo, idx1, idx2, geo2=None, oid=-1, radius=5.0, ltol=0.2, stol=0.3, angle_tol=5):
    struct0 = Structure.from_ase_atoms(geo)
    if struct0[idx1].species != struct0[idx2].species:
        return False
    def get_local_environment(struct0, center_idx, radius):
        center = struct0[center_idx]
        neighbors = struct0.get_neighbors(center, radius)
        fc_center = center.frac_coords

        env_coords = []
        for neighbor_obj in neighbors:
            fc = neighbor_obj.frac_coords
            rel_frac = np.mod(fc - fc_center + 0.5, 1) - 0.5
            rel_cart = struct0.lattice.get_cartesian_coords(rel_frac)
            env_coords.append((neighbor_obj.species, rel_cart))

        return env_coords

    env1 = get_local_environment(struct0, idx1, radius)
    env2 = get_local_environment(struct0, idx2, radius)

    if len(env1) != len(env2):
        return False

    large_lattice = [[50, 0, 0], [0, 50, 0], [0, 0, 50]]

    def create_env_structure(env):
        species = [site[0] for site in env]
        coords = [site[1] for site in env]
        return Structure(large_lattice, species, coords, coords_are_cartesian=True)

    struct1 = create_env_structure(env1)
    struct2 = create_env_structure(env2)

    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False
    )
    res=sm.fit(struct1, struct2)
    if geo2!=None:
        dis1=geo2.get_distance(idx1,oid,mic=1)
        dis2=geo2.get_distance(idx2,oid,mic=1)

        if np.abs(dis1-dis2)<1e-5 and res==True:
            return True
        else:
            return False
    else:
        return res



def classify_same_atoms(geo, idlist, geo2=None, oid=-1):
    idlist.sort()
    atoms = {idlist[0]: 0}
    reslist = [[idlist[0]]]
    for i in range(1, len(idlist)):
        mark = -1
        for atom in atoms:
            if is_same_atoms(geo, idlist[i], atom, geo2, oid):
                mark = atoms[atom]
                break
        if mark >= 0:
            reslist[mark].append(idlist[i])
        else:
            atoms[idlist[i]] = len(reslist)
            reslist.append([idlist[i]])
    atomkey = []
    for i in atoms:
        dis = [[j, geo[j].position[2]] for j in reslist[atoms[i]]]
        atomkey.append(sorted(dis, key=lambda x: x[1], reverse=True)[0][0])
    return atomkey, atoms, reslist


def print_features(filename, name):
    dope_M,host_surf=name.strip().split('_')
    host_M=host_surf[:-5]
    struct = Structure.from_file(filename)
    asegeo = struct.to_ase_atoms()
    ads = AdsorbateSiteFinder(struct)
    ads_sites0 = ads.find_adsorption_sites(distance=0.0, positions=('bridge', 'hollow'))['all']
    ads_sites = ads.find_adsorption_sites(distance=0.5, positions=('bridge', 'hollow'))['all']  # o hollow bridge  metal otop
    top_sites0 = [s for s in range(len(ads.slab)) if ads.slab[s].properties["surface_properties"]=='surface']
    DE = -1.5

    for i in range(len(ads_sites0)):
        site0 = ads_sites0[i]
        site = ads_sites[i]

        surf0 = asegeo.copy()
        surf = asegeo.copy()
        surfn = asegeo.copy()
        surf0.append(Atom('O', position=site0))
        surf.append(Atom('O', position=site))
        surfn.append(Atom('O', position=site0))
        surfn.append(Atom('O', position=site))
        rv = surfn.get_distance(len(surfn)-2, len(surfn)-1, mic=1, vector=1)
        normal_rv = rv/np.linalg.norm(rv)

        
        cn = coordination_numbers(surf, probe=len(surf)-1)[0][len(surf)-1]
        try:
            neighbors = generateNeighborList(surf, [len(surf)-1], generateRcDict(surf, None, 1.2))[len(surf)-1]
        except:
            continue

        # the average of dOM = 0.12355009OCN+1.66749702rm+-0.4323462314736233

        rm = np.average([element(surf0[j].symbol).metallic_radius for j in neighbors])/100
       # print(cn,rm)
        d_aver_OM = 0.12355009*cn+1.66749702*rm-0.4323462314736233
        
        disom = np.average([surf0.get_distance(len(surf0)-1, j, mic=1) for j in neighbors])
   #     print(d_aver_OM,disom)
        while d_aver_OM <= disom:
            d_aver_OM = d_aver_OM+0.1
    #    print(d_aver_OM)
        h = np.sqrt(d_aver_OM*d_aver_OM-disom*disom)
        hv = normal_rv*h
        opos = site0+hv
        surfn2 = asegeo.copy()
        surfn2.append(Atom('O', position=opos))
        
        
    #    view(surfn2)
        # DE Eb_aver_OM XNv_M X_aver_MM Diff_X_aver_MM
     #   print(name,i,len(surfn2),generateNeighborList(surfn2, [len(surfn2)-1], generateRcDict(surfn2, None, 1.2)))
#        print(generateNeighborList(surfn2, [len(surfn2)-1], generateRcDict(surfn2, None, 1.2)))
        try:
            neighbors = generateNeighborList(surfn2, [len(surfn2)-1], generateRcDict(surfn2, None, 1.2))[len(surfn2)-1]
        except:
            continue
        if len(neighbors)==0:
            continue
  #      Eb_aver_OM = np.average([bond_energy['O-'+surfn2[j].symbol]/96.4853 for j in neighbors])

        target_M=[] #dope_M
        mark=1
        for j in neighbors:
            if surfn2[j].symbol==dope_M:
                mark=0
        if mark==0:
            target_M=[host_M,dope_M]
        else:
            target_M=[dope_M]
        oreac = len(surfn2)-1

        
        dism = [[atom.index, surfn2.get_distance(oreac, atom.index, mic=1), atom.symbol] for atom in surfn2 if atom.index != oreac]
        dism = sorted(dism, key=lambda x: x[1])
        dism2 = [j for j in dism if j[1] < 4.0]
        dism3 = [j[0] for j in dism2]
        classifygeo = asegeo.copy()
        res, _, _ = classify_same_atoms(classifygeo, dism3, surfn2, oreac)

        res2=[]
        for atomid in res:
            if deter_geo_limit(surfn2,atomid):
                continue
            else:
                res2.append(atomid)
        res2=[ k for k in res2 if k in top_sites0 and surfn2[k].symbol in target_M]

        if len(res2)==0:
            continue
        surfn2.write(name+'_'+str(i)+'.vasp')
        count = 0
        
        for mreac in res2:

            d_OM=surfn2.get_distance(mreac,len(surfn2)-1,mic=1)
            surfn3=surfn2.copy()
            surfn3.pop(-1)
            MMGCN=coordination_numbers(surfn3, probe=mreac,generalized=1,norm=12)[0][mreac]

            X_M = element(asegeo[mreac].symbol).electronegativity()
            XNv_M = element(asegeo[mreac].symbol).electronegativity()*element(asegeo[mreac].symbol).nvalence()
            try:
                M_neighbors = generateNeighborList(asegeo, [mreac], generateRcDict(asegeo, None, 1.2))[mreac]
            except:
                continue
            X_aver_MM = np.average([element(asegeo[k].symbol).electronegativity() for k in M_neighbors])
            Diff_XNv_aver_MM = np.average(np.abs([element(asegeo[mreac].symbol).electronegativity()*element(asegeo[mreac].symbol).nvalence() -
                                                element(asegeo[k].symbol).electronegativity()*element(asegeo[k].symbol).nvalence() for k in M_neighbors]))

            with open('res','a+') as fp:
                fp.write('{0:<50s} {1:>5d} {2:>5d} {3:>5d} {4:>20.6f} {5:>20.6f} {6:>20.6f} {7:>20.6f} {8:>20.6f} {9:>20.6f}\n'.format(
                name, i, oreac, mreac, DE, d_OM, MMGCN, XNv_M, X_aver_MM, Diff_XNv_aver_MM))
            count += 1

# DE d_OM MMGCN XNv_M X_aver_MM Diff_XNv_aver_MM

fp = open('../names', 'r')
data = fp.readlines()
fp.close()
#print_features('../mp-12065_Ag3Pt_110_0.vasp','mp-12065_Ag3Pt_110_0')
for name in data:
    print_features('../SAA-surface/'+name.strip()+'.vasp', name.strip())

