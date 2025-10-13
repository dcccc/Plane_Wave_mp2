import numpy as np
import  time

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

def get_mp2(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied):

    # transform psi from g-space to r-space
    psi_r = np.array([np.fft.ifftn(i) for i in psi_list])
    psi_list = []

    # calculate phi_ia
    time_start = time.time()
    phi_ia = get_phi(psi_r, n_occupied, g2vector_mask)
    time_end = time.time()
    print("Time used to calculate phi_ia:       {:10d} seconds".format(int(time_end - time_start)))


    psi_r = []
    # calculate electron repulsion integral
    time_start = time.time()
    eri = get_eri(phi_ia, op_coul)
    time_end = time.time()
    print("Time used to calculate eri:          {:10d} seconds".format(int(time_end - time_start)))

    phi_ia = []
    # calculate mp2 energy
    time_start = time.time()
    e_d, e_x = get_e(eri, eigenvalues, n_occupied)
    time_end = time.time()
    print("Time used to calculate mp2 energy:   {:10d} seconds".format(int(time_end - time_start)))

    return(e_d, e_x)


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


import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
def psi_g2r(psi_name, psi_r_name, psi_shape, range_list):

    shm_psi = SharedMemory(name = psi_name, create=False, size=np.prod(psi_shape)*16)
    psi_list = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psi.buf)

    shm_psir = SharedMemory(name = psi_r_name, create=False, size=np.prod(psi_shape)*16)
    psi_r = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psir.buf)

    for n in range(range_list[0],range_list[1]):
        psi_r[n] = np.fft.ifftn(psi_list[n])
    return(0)



def get_phi_pp(phi_name, phi_shape, psi_name, psi_shape,mask_name,  un_occupied_range):
    shm_phi = SharedMemory(name=phi_name, create=False, size=np.prod(phi_shape)*16)
    phi_ia = np.ndarray(phi_shape, dtype=np.complex128, buffer=shm_phi.buf)

    shm_psir = SharedMemory(name = psi_name, create=False, size=np.prod(psi_shape)*16)
    psi_r = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psir.buf)

    shm_mask0 = SharedMemory(name = mask_name, create=False, size=np.prod(psi_shape[1:]))
    g2vector_mask = np.ndarray(psi_shape[1:], dtype=bool, buffer=shm_mask0.buf)
    n_occupied = phi_shape[0]
    for i in range(n_occupied):
        for a in range(un_occupied_range[0],un_occupied_range[1]):
            phi_ia[i, a] = np.fft.fftn(np.conj(psi_r[i])*psi_r[n_occupied+a])[g2vector_mask]          

    return(0)


def get_eri_pp(eri_name, phi_name, op_name, phi_shape, un_occupied_range):
    n_occupied, n_unoccupied = phi_shape[0], phi_shape[1]
    shm_eri = SharedMemory(name=eri_name,create=False, size=np.prod(phi_shape[:2])**2*16)
    eri = np.ndarray([n_occupied, n_unoccupied]*2, dtype=np.complex128, buffer=shm_eri.buf)

    shm_phi = SharedMemory(name=phi_name, create=False, size=np.prod(phi_shape)*16)
    phi_ia = np.ndarray(phi_shape, dtype=np.complex128, buffer=shm_phi.buf)

    shm_op = SharedMemory(name=op_name, create=False, size=phi_shape[-1])
    op_coul = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)


    for a in range(un_occupied_range[0],un_occupied_range[1]):
        for i in range(n_occupied):
            for j in range(n_occupied):
                for b in range(n_unoccupied):
                    eri[i,a,j,b] = np.sum(phi_ia[i,a]*op_coul*np.conj(phi_ia[j,b]))    

    return(0)

def get_emp2_pp(eri_name, eri_shape, eigenvalues, un_occupied_range):
    shm_eri = SharedMemory(name=eri_name,create=False, size=np.prod(eri_shape)*16)
    eri = np.ndarray(eri_shape, dtype=np.complex128, buffer=shm_eri.buf)

    n_occupied, n_unoccupied = eri_shape[0], eri_shape[1]
    e_d =0.0
    e_x =0.0
    for i in range(n_occupied):
        for a in range(un_occupied_range[0], un_occupied_range[1]):
            for j in range(n_occupied):
                for b in range(n_unoccupied):
                    e_d += np.real(eri[i,a,j,b]*np.conj(2*eri[i,a,j,b])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied])
                    e_x += np.real(eri[i,a,j,b]*np.conj(eri[i,b,j,a])) / \
                    (eigenvalues[i]+eigenvalues[j]-eigenvalues[a+n_occupied]-eigenvalues[b+n_occupied])     
    return(e_d, e_x)



def get_mp2_parallel(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied, n_thread=4):

    psi_shape = psi_list.shape
    n_unoccupied = psi_shape[0] - n_occupied

    time_start = time.time()
    # create the shared memory for psi and psi_r
    shm_psir = SharedMemory(create=True, size=psi_list.nbytes)
    shm_psi  = SharedMemory(create=True, size=psi_list.nbytes)
    tmp_psi = np.ndarray(psi_list.shape, dtype=np.complex128, buffer=shm_psi.buf)
    tmp_psi[:] = psi_list[:]
    psi_list = []

    # transform psi from g-space to r-space
    task_idx_list = task_parting(n_unoccupied+n_occupied, n_thread)   
    pool = mp.Pool(n_thread)
    results = [pool.apply_async(psi_g2r, args=(shm_psi.name, shm_psir.name, psi_shape,task_idx_list[i],))  
                    for i in range(n_thread) ]    
    pool.close()
    pool.join()
    temp2 = np.array([p.get() for p in results])


    # release the shared memory for psi
    shm_psi.unlink()

    # divide the tasks into n_thread parts, each part has approximately n_task/n_thread tasks
    # the number of unoccupied orbitals is always larger than number of the occupied orbitals
    task_idx_list = task_parting(n_unoccupied, n_thread)

    # calculate phi_ia in parallel
    phi_shape = [n_occupied, n_unoccupied, np.sum(g2vector_mask)]
    
    # create the shared memory for phi_ia and g2vector_mask
    shm_phi  = SharedMemory(create=True, size=np.prod(phi_shape)*16)
    shm_mask = SharedMemory(create=True, size=g2vector_mask.nbytes)
    tmp_mask = np.ndarray(g2vector_mask.shape, dtype=bool, buffer=shm_mask.buf)
    tmp_mask[:]=g2vector_mask[:]
    g2vector_mask = []


    # calculate eri in parallel
    pool = mp.Pool(n_thread)
    results = [pool.apply_async(get_phi_pp, args=(shm_phi.name, phi_shape, shm_psir.name, 
                                                  psi_shape, shm_mask.name,  task_idx_list[i],))  
            for i in range(n_thread) ]    
    pool.close()
    pool.join()
    tmp = np.hstack([p.get() for p in results])

    # get the results
    time_end = time.time()
    print("Time used to calculate phi_ia:        {:10d} seconds".format(int(time_end - time_start)))

    # release the shared memory for psi_r and g2vector_mask
    shm_psir.unlink()
    shm_mask.unlink()

    # create the shared memory for eri and op_coul
    shm_eri = SharedMemory(create=True, size=np.prod(phi_shape[:2])**2*16)
    shm_op = SharedMemory(create=True, size=op_coul.nbytes)
    tmp_op = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)
    tmp_op[:]=op_coul[:]
    op_coul = []



    # calculate eri in parallel
    pool = mp.Pool(n_thread)
    time_start = time.time()
    results = [pool.apply_async(get_eri_pp, args=(shm_eri.name,shm_phi.name, shm_op.name, 
                                                  phi_shape, task_idx_list[i],))  
        for i in range(n_thread) ]    
    pool.close()
    pool.join()
    tmp = np.hstack([p.get() for p in results])
    # get the results
    time_end = time.time()
    print("Time used to calculate eri:           {:10d} seconds".format(int(time_end - time_start)))


    # release the shared memory for phi_ia and op_coul
    shm_phi.unlink()
    shm_op.unlink()

    # calculate energy in parallel
    time_start = time.time()
    pool = mp.Pool(n_thread)    
    results = [pool.apply_async(get_emp2_pp, args=(shm_eri.name, phi_shape[:2]*2,eigenvalues ,
                                                   task_idx_list[i],))  
        for i in range(n_thread) ]

    pool.close()
    pool.join()
    emp2 = np.array([p.get() for p in results])

    # release the shared memory for eri
    shm_eri.unlink()


    e_d, e_x = np.sum(emp2, axis=0) 
    time_end = time.time()
    print("Time used to get mp2 energy:          {:10d} seconds".format(int(time_end - time_start)))

    return(e_d, e_x)
