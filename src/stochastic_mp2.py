import numpy as np
import time

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
def get_stochastic_mp2(psi_list,n_occupied, tau_list, w_list, op_coul, g2_mask, eigenvalues, round_num = 1000):

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
                rho1 = np.fft.fftn(psi_a[i]*np.conj(psi_i[i]))[g2_mask]
                rho2 = np.fft.fftn(psi_b[i]*np.conj(psi_j[i]))[g2_mask]
                # eq(3)
                d = np.sum(rho1*np.conj(rho2)*op_coul)
                tmp_d += d*np.conj(d)*w

                rho3 = np.fft.fftn(psi_b[i]*np.conj(psi_i[i]))[g2_mask]
                rho4 = np.fft.fftn(psi_a[i]*np.conj(psi_j[i]))[g2_mask]
                # eq(4)
                x = np.sum(rho3*np.conj(rho4)*op_coul)   
                tmp_x += d*np.conj(x)*w


        e_d.append(tmp_d*2/num_of_orbitals)
        e_x.append(tmp_x/num_of_orbitals)

    return(np.array(e_d).real, np.array(e_x).real)