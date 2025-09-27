import numpy as np
from scipy.optimize import fmin_l_bfgs_b



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
    
    
def get_lmp2_pp(psi_list, phi_ia, tau_list, w_list, eigenvalues, g2vector_mask, op_coul, ng_range=[]):

    e_d_total = 0.
    e_x_total = 0.
    for tau, w in zip(tau_list, w_list):        
        for ng in range(ng_range[0], ng_range[1]):
            w_psi_list = get_w_psi(psi_list, phi_ia, eigenvalues, g2vector_mask, ng,  tau=tau)
            e_d, e_x= get_lmp2(psi_list, w_psi_list, op_coul, g2vector_mask)
            e_d_total += e_d * w * op_coul[ng]
            e_x_total += e_x * w * op_coul[ng]

    return(e_d_total.real, e_x_total.real)
