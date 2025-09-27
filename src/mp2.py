import numpy as np

def get_phi(psi_r, n_occupied, g2vector_mask):    
    '''
    calculate the overlap density element of between occupied-virtual pairs of orbitals
    eq(8) in the paper 'Journal of Chemical Theory and Computation 2023 19 (24), 9211-9227'
    '''
    # phi_ia = <psi_i|psi_a> = ifft(psi_i(r)*psi_a(r))
    phi_ia = np.zeros((n_occupied, len(psi_r)-n_occupied, np.sum(g2vector_mask)), dtype=np.complex128)
    for i in range(n_occupied):
        for a in range(n_occupied,len(psi_r)):
            phi_ia[i, a-n_occupied] = np.fft.fftn(np.conj(psi_r[i])*psi_r[a])[g2vector_mask]            
    return(phi_ia)

def get_eri(phi_ia, op_coul):
    """"
    calculate the electron repulsion integral between occupied-virtual pairs of orbitals
    eq(3) and eq(7) in the paper 'Journal of Chemical Theory and Computation 2023 19 (24), 9211-9227'
    """
    # when only gamma point is used, psi(-g) = psi(g)*, the ERI can be calculated as:
    # <ij|ab> = <psi_i|psi_a> * op_coul * <psi_j|psi_b>*
    shape = list(phi_ia.shape)
    eri = np.zeros(shape[:2]*2, dtype=np.complex128)
    for i in range(shape[0]):
        for a in range(shape[1]):
            for j in range(shape[0]):
                for b in range(shape[1]):
                    eri[i,a,j,b] = np.sum(phi_ia[i,a]*op_coul*np.conj(phi_ia[j,b]))
    
    return(eri)
    
def get_e(eri, eigenvalues, n_occupied):

    """
    calculate the MP2 correlation energy
    eq(2) in the paper 'Journal of Chemical Theory and Computation 2023 19 (24), 9211-9227'
    E_mp2 = <ij|ab>(2*<ij|ab>-<ij|ba>)*/(e_i + e_j - e_a - e_b)
    """

    shape = list(eri.shape)
    e_d =0.0
    e_x =0.0
    for i in range(shape[0]):
        for a in range(shape[1]):
            for j in range(shape[0]):
                for b in range(shape[1]):
                    e_d += np.real(eri[i,a,j,b]*np.conj(2*eri[i,a,j,b])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied])
                    e_x += np.real(eri[i,a,j,b]*np.conj(eri[i,b,j,a])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied]) 
    
    return(e_d.real, e_x.real)




def task_parting(n_task, n_thread):
    # divide the tasks into n_thread parts, each part has approximately n_task/n_thread tasks
    n = int( n_task / n_thread )
    n_re = n_task - n * n_thread    
    n_list = [0]+ [ n+1 for i in range(n_thread+1) if i < n_re] + [ n for i in range(n_thread+1) if i > n_re]    
    task_idx_list = np.cumsum(n_list)
    task_idx_list = [[task_idx_list[i], task_idx_list[i+1]] for i in range(n_thread)]
    return(task_idx_list)


# parallel verison of former functions
# In order to calculate with parallel processing, the functions are modified to accept a range of unoccupied orbitals
# and return the results for that range.
# All the functions below are designed to be used in a multiprocessing pool with the global variables 
# `psi_r`, `g2vector_mask`, `n_occupied`, and `eigenvalues` already defined.
def get_phi_pp(psi_r, n_occupied, g2vector_mask, un_occupied_range):
    n_calculated = un_occupied_range[1]-un_occupied_range[0]
    phi_ia = np.zeros((n_occupied, n_calculated, np.sum(g2vector_mask)), dtype=np.complex128)
    for i in range(n_occupied):
        for a in range(n_calculated):
            phi_ia[i, a] = np.fft.fftn(np.conj(psi_r[i])*psi_r[n_occupied+a+un_occupied_range[0]])[g2vector_mask]            
    return(phi_ia)

def get_eri_pp(phi_ia,op_coul, un_occupied_range):
    shape = list(phi_ia.shape)
    n_calculated = un_occupied_range[1]-un_occupied_range[0]
    eri = np.zeros([n_calculated,shape[0]]+shape[:2], dtype=np.complex128)
    for a in range(n_calculated):
        for i in range(shape[0]):
            for j in range(shape[0]):
                for b in range(shape[1]):
                    eri[a,i,j,b] = np.sum(phi_ia[i,a+un_occupied_range[0]]*op_coul*np.conj(phi_ia[j,b]))    
    return(eri)

def get_emp2_pp(eri, eigenvalues, un_occupied_range):
    shape = list(eri.shape)
    e_d =0.0
    e_x =0.0
    n_occupied = shape[0]
    for i in range(shape[0]):
        for a in range(un_occupied_range[0], un_occupied_range[1]):
            for j in range(shape[0]):
                for b in range(shape[1]):
                    e_d += np.real(eri[i,a,j,b]*np.conj(2*eri[i,a,j,b])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied])
                    e_x += np.real(eri[i,a,j,b]*np.conj(eri[i,b,j,a])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied])     
    return(e_d, e_x)