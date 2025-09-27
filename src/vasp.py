from vaspwfc import vaspwfc
import numpy as np
from aewfc import vasp_ae_wfc
import numpy as np
import xml.etree.ElementTree as ET

def get_vasp_psi(wavecar, poscar, potcar, nband, latt9)


    # read the wavefunction from wavecar file
    pswfc = vaspwfc(wavecar, lgamma = True)
    aewfc = vasp_ae_wfc(pswfc, poscar=poscar, potcar=potcar, aecut=-1)
    phi_ae_list = []
    for i in range(nband):
        phi_ae, phi_core_ae, phi_core_ps = aewfc.get_ae_wfc(
        iband=i+1, lcore=True, norm=True)
        phi_ae_list.append(phi_ae)

    # fft grid size
    grid = phi_ae.shape



    g2_mask = np.abs(phi_ae.real)+ np.abs(phi_ae.real) > 0

    rec_latt= 2. * np.pi * np.linalg.inv(latt9).T
    g2_vector=np.zeros(grid)
    x_range = [np.arange(grid[0]//2) , np.arange(-grid[0]//2, 0)]
    x_range = np.hstack(x_range)
    y_range = [np.arange(grid[1]//2) , np.arange(-grid[1]//2, 0)]
    y_range = np.hstack(y_range)
    z_range = [np.arange(grid[2]//2) , np.arange(-grid[2]//2, 0)]
    z_range = np.hstack(z_range)

    a=np.meshgrid(x_range, y_range, z_range)
    a=np.array(a)
    a=a.transpose(1,2,3,0)

    tmp = a[:, :, :, 0] * rec_latt[0] + a[:, :, :, 1] * rec_latt[1] + a[:, :, :, 2] * rec_latt[2]
    g2_vector=np.sum(tmp**2, axis=-1)


    return(phi_ae_list, grid, g2_vector, g2_mask)


def get_vasp_data(vasprun_xml):

    tree = ET.parse(st_xml )
    root = tree.getroot()
    
    varray = tree.findall("calculation/eigenvalues/array/set/set/set")[0]
    eigenvalues = list(varray.itertext())[1:-1:2]
    eigenvalues = np.array([i.strip().split() for i in eigenvalues], dtype=np.float64)
    occupations = eigenvalues[:,1]
    eigenvalues = eigenvalues[:,0]/27.2113846
    n_occupied = np.sum(occupations > 0.)

    varray = tree.findall("structure/crystal/varray")[0]
    latt9 = list(varray.itertext())[1:-1:2]
    latt9 = np.array([i.strip().split() for i in latt9], dtype=np.float64) / 0.5291772


    return(eigenvalues, n_occupied, latt9)
