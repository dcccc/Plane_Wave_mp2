import numpy as np
import time
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
## stochastic orbitals mp2

def get_stochastic_orbitals(psi, eigenvalues, tau=0.0, num_of_orbitals=100):
    np.random.seed(int(time.time()*1000000)%1000000)
    grid = psi[0].shape
    n_psi = len(psi)
    p = np.random.rand(n_psi * num_of_orbitals * 2).reshape((num_of_orbitals, n_psi, 2)) - 0.5

    # As only gamma point here, the random uniform coefficients can be real, and in range of [-3^0.5, 3^0.5]
    # If the gamma point is not used, the random coefficients should be complex, both read and imaginary
    # parts p range of [-3^0.5/2, 3^0.5/2]
    p = p*(3)**0.5*2
    p = p[:,:,0] #+ p[:,:,1] * 1.0j
    p = p * np.exp(eigenvalues * tau / 2.).reshape((1,-1))    
    chi = np.dot(p, psi.reshape((len(psi),-1)))
    chi_r = [np.fft.ifftn(i.reshape(grid)) for i in chi]    
    return(chi_r)


# The psuedo code of the fig 1 in paper 'J. Chem. Phys. 148, 064103 (2018)'
# A simplified stochastic MP2 implementation
def get_stochastic_mp2(psi_name, mask_name, op_name, psi_shape, n_occupied, eigenvalues, 
                       tau_list, w_list, round_num = 10):

    # load shared memory for psi, g2vector_mask and op_coul
    shm_psi = SharedMemory(name=psi_name, create=False, size=np.prod(psi_shape)*16)
    psi_list = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psi.buf)

    shm_mask = SharedMemory(name = mask_name, create=False, size=np.prod(psi_shape[1:]))
    g2vector_mask = np.ndarray(psi_shape[1:], dtype=bool, buffer=shm_mask.buf)

    shm_op = SharedMemory(name=op_name, create=False, size=np.sum(g2vector_mask))
    op_coul = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)


    n_unoccupied = len(psi_list) - n_occupied

    num_of_orbitals = 1
    e_d = []
    e_x = []
    for n in range(round_num):
        tmp_x = 0.0
        tmp_d = 0.0
        for tau, w in zip(tau_list, w_list):
            # eq (12) in the paper
            psi_i   = get_stochastic_orbitals(psi_list[:n_occupied],  eigenvalues[:n_occupied], tau=tau, num_of_orbitals=num_of_orbitals)
            psi_j   = get_stochastic_orbitals(psi_list[:n_occupied],  eigenvalues[:n_occupied], tau=tau, num_of_orbitals=num_of_orbitals)
            psi_a   = get_stochastic_orbitals(psi_list[n_occupied:], -eigenvalues[n_occupied:], tau=tau, num_of_orbitals=num_of_orbitals)
            psi_b   = get_stochastic_orbitals(psi_list[n_occupied:], -eigenvalues[n_occupied:], tau=tau, num_of_orbitals=num_of_orbitals)
            
            for i in range(num_of_orbitals):           
                rho1 = np.fft.fftn(psi_a[i]*np.conj(psi_i[i]))[g2vector_mask]
                rho2 = np.fft.fftn(psi_b[i]*np.conj(psi_j[i]))[g2vector_mask]
                # eq(3)
                d = np.sum(rho1*np.conj(rho2)*op_coul)
                tmp_d += d*np.conj(d)*w

                rho3 = np.fft.fftn(psi_b[i]*np.conj(psi_i[i]))[g2vector_mask]
                rho4 = np.fft.fftn(psi_a[i]*np.conj(psi_j[i]))[g2vector_mask]
                # eq(4)
                x = np.sum(rho3*np.conj(rho4)*op_coul)   
                tmp_x += d*np.conj(x)*w


        e_d.append(tmp_d/num_of_orbitals)
        e_x.append(tmp_x/num_of_orbitals)

    return(np.mean(e_d).real, np.mean(e_x).real)



def get_stochastic_mp2_pp(psi_list, g2vector_mask, eigenvalues, op_coul, n_occupied,
                          tau_list,w_list, n_thread, round_num = 10):
    
    psi_shape = psi_list.shape
    # create shared memory for psi and g2vector_mask
    shm_psi  = SharedMemory(create=True, size=psi_list.nbytes)
    tmp_psi = np.ndarray(psi_list.shape, dtype=np.complex128, buffer=shm_psi.buf)
    tmp_psi[:] = psi_list[:]
    psi_list = []

    shm_mask = SharedMemory(create=True, size=g2vector_mask.nbytes)
    tmp_mask = np.ndarray(g2vector_mask.shape, dtype=bool, buffer=shm_mask.buf)
    tmp_mask[:]=g2vector_mask[:]
    g2vector_mask = []

    shm_op = SharedMemory(create=True, size=op_coul.nbytes)
    tmp_op = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)
    tmp_op[:]=op_coul[:]

    if n_thread > 1:
        part_step = round_num // n_thread
        pool = mp.Pool(n_thread)
        results = [pool.apply_async(get_stochastic_mp2, args=(shm_psi.name, shm_mask.name, shm_op.name, 
                                    psi_shape, n_occupied, eigenvalues, 
                                    tau_list, w_list, part_step))  
                   for i in range(n_thread) ]    
        pool.close()
        pool.join()


        e_mp2 = np.array([p.get() for p in results])
        e_d = e_mp2[:,0].reshape(-1,).mean()
        e_x = e_mp2[:,1].reshape(-1,).mean()

    else:
        e_d, e_x = get_stochastic_mp2(shm_psi.name, shm_mask.name, shm_op.name, psi_shape, 
                                      n_occupied, eigenvalues, 
                                      tau_list, w_list, round_num = round_num)
        e_d = e_d.mean()
        e_x = e_x.mean()


    # release the shared memory
    shm_psi.unlink()
    shm_mask.unlink()
    shm_op.unlink()

    return(e_d, e_x)

