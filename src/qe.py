import h5py
import numpy as np
import xml.etree.ElementTree as ET

def mm_to_nn(i,n):
    if i > n // 2:
        i=i-n
    return(i)

def read_qe_wavefunction(latt9, grid_point, nband, wfc_hdf5):

    # read the wavefunction from the hdf5 file
    wfc=h5py.File(wfc_hdf5,"r",)
    MillerIndices = np.array(wfc['MillerIndices'])

    psi = np.array(wfc["evc"]).reshape((nband,-1,2))
    pw_psi = psi[:,:,0]+psi[:,:,1]*1.j

    pw_psi_3d = np.zeros([nband]+grid_point, dtype=np.complex128)
    g2vector = np.zeros(grid_point, dtype=np.float64)-1.0
    rec_latt= 2.*np.pi * np.linalg.inv(latt9).T

    # when gamma point method is used, the psi(g)* = psi(-g), so we need to restore the psi(-g) as well
    for n,idx in enumerate(MillerIndices):
        if idx[0]<0:
            i=idx[0]+grid_point[0]
            ii = -idx[0]
        elif idx[0]>0:
            i=idx[0]
            ii = grid_point[0] - idx[0]
        else:
            i=0
            ii=0
        if idx[1]<0:
            j=idx[1]+grid_point[1]
            jj = -idx[1]
        elif idx[1]>0:
            j=idx[1]
            jj = grid_point[1] - idx[1]
        else:
            j=0
            jj=0
        if idx[2]<0:
            k=idx[2]+grid_point[2]
            kk=-idx[2]            
        elif idx[2]>0:
            k=idx[2]
            kk = grid_point[2] - idx[2]
        else:
            k=0
            kk=0            
        tmp=idx[0]*rec_latt[0] + idx[1]*rec_latt[1] + idx[2]*rec_latt[2]        
        g2vector[i,j,k]=np.sum(tmp**2)
        g2vector[ii,jj,kk]=np.sum(tmp**2)
        pw_psi_3d[:,i,j,k]=pw_psi[:,n]
        pw_psi_3d[:,ii,jj,kk]=np.conj(pw_psi[:,n])

    g2vector_mask = g2vector > -0.5
    op_coul = 4*np.pi / g2vector[g2vector_mask]
    # set up the coulomb potential as 0.0 for g^2=0
    op_coul[np.isinf(op_coul)] = 0.0

    return(pw_psi_3d*np.prod(grid_point)**0.5, op_coul, g2vector_mask)


def get_qe_data(schema_xml):

    tree = ET.parse(schema_xml )
    root = tree.getroot()
    a=root.findall("output/band_structure/ks_energies/eigenvalues")[0]
    eigenvalues = np.array(a.text.replace("\n","").strip().split(), dtype=np.float64)
    nband = len(eigenvalues)
    occupations = root.findall("output/band_structure/ks_energies/occupations")[0]
    occupations = np.array(occupations.text.strip().split(), dtype=np.float64)
    n_occupied = np.sum(occupations > 0.)

    cell = root.findall("input/atomic_structure/cell")[0]
    cell = list(cell.itertext())[1::2]
    latt9 = np.array([i.split() for i in cell], dtype=np.float64)

    grid = root.findall("output/basis_set/fft_grid")[0]
    grid = np.array([i[1] for i in grid.items()], dtype=np.int32).tolist()

    return(latt9, eigenvalues, n_occupied, grid)

