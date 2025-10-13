import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import  time
from multiprocessing.shared_memory import SharedMemory

# The tau and w are calculated based on the paper:
# "J. Chem. Phys. 96, 489–494 (1992) 10.1063/1.462485"
def get_w(fx, x, t):
    
    xt = t.reshape((-1,1)) * x.reshape((1,-1))
    # deal with the overflow of np.float64
    xt[xt<-709.] = -709.
    # eq(18)
    a = np.dot(np.exp(-xt) / x.reshape((1,-1)), fx)
    
    xtt = (t.reshape((-1,1,1)) + t.reshape((1,-1,1))) * x.reshape((1, 1, -1))
    xtt[xtt<-709.] = -709.
    # eq(19)
    b = np.dot(np.exp(-xtt), fx)
    w = np.dot(a, np.linalg.inv(b))
    
    return(w)


def laplace_error(t, fx, x):
    
    w = get_w(fx, x, t)
    xt = x.reshape((-1,1)) * t.reshape((1,-1))
    # deal with the overflow of np.float64
    xt[xt<-709.] = -709.
    exp_xt = np.exp(-xt)
    
    # eq(16)    
    error = (1.0 / x - np.dot(exp_xt, w))
    de_dt =  np.dot(exp_xt.T, 2 * error * fx * x) * w
    
    return(np.sum(np.abs(error)**2*fx), de_dt)



def get_tau(tau_list=[], n_w=8, n_x=600, x_range=[0.1, 1000]):
    # the number of tau points is n_w, and the number of x points is n_x
    # the default x points are in the range of [0.1, 1000], and a 8 tau will be generated
    # which will give a values of tau and weigth list with a mean error of 0.0007
    if len(tau_list) == 0:
        tau_list = 2. ** np.arange(-1, n_w-1)
    x= np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), n_x)

    # Minimize the laplace error function with L-BFGS-B algorithm
    result = fmin_l_bfgs_b(laplace_error, tau_list, args=(np.ones(len(x)), x), approx_grad=False)   
    
    # optimized tau and weight
    tau_list = result[0]
    w_list = get_w(10./x, x, tau_list)
    
    # the error
    xt = x.reshape((-1,1)) * tau_list.reshape((1,-1))
    error = 1.0 / x - np.dot(np.exp(-xt), w_list)
    error = np.mean(np.abs(error))

    return(tau_list, w_list, error)





# The psuedo code of the fig 1 in paper 'J. Chem. Phys. 146, 104101 (2017)'
def get_w_psi(psi, phi_ia, eigenvalues, g2vector_mask, ng,  tau=0.0):
    w_psi_list = []
    n_occupied = phi_ia.shape[0]
    n_unoccupied = len(psi) - n_occupied
    for i in range(n_occupied):
        tmp_w = 0.0+0.0j
        for a in range(n_unoccupied):
            delta_e = eigenvalues[i] - eigenvalues[a+n_occupied]
            tmp_w += phi_ia[i, a][ng] * np.exp(delta_e * tau) * psi[a+n_occupied][g2vector_mask]       
        w_psi_list.append(tmp_w)        
    return(w_psi_list)



def get_lmp2(psi, w_list, coulomb_potential, g2vector_mask):
    
    e_d = 0.0
    rho_r = 0.0
    w_r_list   = []
    psi_r_list = []
    for i, w in enumerate(w_list):
        tmp=np.zeros(psi[0].shape, dtype=np.complex128)
        tmp[g2vector_mask] = w
        w_r = np.fft.ifftn(tmp)
        w_r_list.append(w_r)
        psi_r = np.fft.ifftn(psi[i])
        psi_r_list.append(psi_r)
        rho_r += w_r * np.conj(psi_r)
    rho = np.fft.fftn(rho_r)[g2vector_mask]
    e_d += np.sum(np.conj(rho)* rho * coulomb_potential)
    
    n_occupied = len(w_list)
    e_x = 0.0
    for i in range(n_occupied):
        for j in range(n_occupied):
            rho1_r = np.conj(psi_r_list[i]) * w_r_list[j]
            rho1   = np.fft.fftn(rho1_r)
            rho2_r = np.conj(psi_r_list[j]) * w_r_list[i]
            rho2   = np.fft.fftn(rho2_r)
            e_x += np.sum(np.conj(rho1[g2vector_mask]) * rho2[g2vector_mask] * coulomb_potential)
    
    
    return(e_d.real , e_x.real)

# serial implementation
def laplace_mp2_energy(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied, tau_list, w_list):

    from mp2 import get_phi
    e_x_total = 0.0
    e_d_total = 0.0

    time_start = time.time()
    # transform psi from g-space to r-space
    psi_r = np.array([np.fft.ifftn(i) for i in psi_list])

    # calculate phi_ia
    
    phi_ia = get_phi(psi_r, n_occupied, g2vector_mask)
    time_end = time.time()
    psi_r = []
    print("Time used to calculate phi_ia:        {:10d} seconds".format(int(time_end - time_start)))

  
    time_start = time.time()
    for tau, w in zip(tau_list, w_list):        
        for ng in range(len(op_coul)):
            w_psi_list = get_w_psi(psi_list, phi_ia, eigenvalues, g2vector_mask, ng,  tau=tau)
            e_d, e_x= get_lmp2(psi_list, w_psi_list, op_coul, g2vector_mask)
            e_d_total += e_d * w * op_coul[ng]
            e_x_total += e_x * w * op_coul[ng]
    time_end = time.time()
    print("Time used to calculate laplace mp2 energy: {:10d} seconds".format(int(time_end - time_start)))
    return(e_d_total, e_x_total)


import multiprocessing as mp
from mp2 import task_parting, psi_g2r, get_phi_pp

def get_lmp2_pp(psi_name, phi_name, mask_name, op_name, psi_shape, phi_shape, tau_list, w_list, 
                eigenvalues, ng_range=[]):

    e_d_total, e_x_total = 0.0, 0.0

    shm_psi = SharedMemory(name = psi_name, create=False, size=np.prod(psi_shape)*16)
    psi_list = np.ndarray(psi_shape, dtype=np.complex128, buffer=shm_psi.buf)

    n_occupied, n_unoccupied = phi_shape[0], phi_shape[1]
    shm_phi = SharedMemory(name=phi_name, create=False, size=np.prod(phi_shape)*16)
    phi_ia = np.ndarray(phi_shape, dtype=np.complex128, buffer=shm_phi.buf)

    shm_op = SharedMemory(name=op_name, create=False, size=phi_shape[-1])
    op_coul = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)

    shm_mask = SharedMemory(name = mask_name, create=False, size=np.prod(psi_shape[1:]))
    g2vector_mask = np.ndarray(psi_shape[1:], dtype=bool, buffer=shm_mask.buf)


    e_d_total = 0.
    e_x_total = 0.
    for tau, w in zip(tau_list, w_list):        
        for ng in range(ng_range[0], ng_range[1]):
            w_psi_list = get_w_psi(psi_list, phi_ia, eigenvalues, g2vector_mask, ng,  tau=tau)
            e_d, e_x= get_lmp2(psi_list, w_psi_list, op_coul, g2vector_mask)
            e_d_total += e_d * w * op_coul[ng]
            e_x_total += e_x * w * op_coul[ng]

    return(e_d_total.real, e_x_total.real)


def laplace_mp2_energy_pp(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied, 
                          tau_list, w_list, n_thread=4):

    from mp2 import get_phi_pp, task_parting, psi_g2r
    psi_shape = psi_list.shape
    n_unoccupied = psi_shape[0] - n_occupied

    time_start = time.time()
    # create the shared memory for psi and psi_r
    shm_psir = SharedMemory(create=True, size=psi_list.nbytes)
    shm_psi  = SharedMemory(create=True, size=psi_list.nbytes)
    tmp_psi = np.ndarray(psi_list.shape, dtype=np.complex128, buffer=shm_psi.buf)
    tmp_psi[:] = psi_list[:]

    # transform psi from g-space to r-space
    task_idx_list = task_parting(n_unoccupied+n_occupied, n_thread)   
    pool = mp.Pool(n_thread)
    results = [pool.apply_async(psi_g2r, args=(shm_psi.name, shm_psir.name, psi_shape,task_idx_list[i],))  
                    for i in range(n_thread) ]    
    pool.close()
    pool.join()
    temp2 = np.array([p.get() for p in results])

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
    shm_psir.unlink()


    # parallel implementation

    # create the shared memory for op_coul
    shm_op = SharedMemory(create=True, size=op_coul.nbytes)
    tmp_op = np.ndarray((-1,), dtype=np.float64, buffer=shm_op.buf)
    tmp_op[:]=op_coul[:]
    op_coul = []

    task_idx_list = task_parting(len(tmp_op), n_thread)

    time_start = time.time()
    pool = mp.Pool(n_thread)
    print(task_idx_list)
    results = [pool.apply_async(get_lmp2_pp, args=(shm_psi.name, shm_phi.name, shm_mask.name, shm_op.name, 
                                psi_shape, phi_shape, tau_list, w_list, 
                                eigenvalues, task_idx_list[i],))  
        for i in range(n_thread) ]

    pool.close()
    pool.join()
    emp2 = np.array([p.get() for p in results])
    e_d_total, e_x_total = np.sum(emp2, axis=0) 
    time_end = time.time()

    # release the shared memory
    shm_psi.unlink()
    shm_phi.unlink()
    shm_mask.unlink()
    shm_op.unlink()
    
    time_end = time.time()
    print("Time used to calculate laplace mp2 energy: {:10d} seconds".format(int(time_end - time_start)))
    return(e_d_total, e_x_total)