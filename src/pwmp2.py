import numpy as np
import argparse
import os, glob, time
from qe import read_qe_wavefunction, get_qe_data
import multiprocessing as mp
from functools import partial

parser = argparse.ArgumentParser(description='Calculate mp2 energy with quantum espresso calculaiton result ')

parser.add_argument('-m',       '--method',            default='mp2', 
                      help="Method to use, mp2, laplace_mp2 or stochastic_mp2 are valid. The default method is mp2")
parser.add_argument('-n_thread','--number_of_threads', default='1',   
                      help="Number of threads to use in calculaiton. The default number is 1 ")
parser.add_argument('-n_step',  '--number_of_stochastic_mp2_steps', default='10000',
                      help="Number of stochastic mp2 steps use in calculaiton. The default number is 10000 ")
args = parser.parse_args()
args.number_of_threads = int(args.number_of_threads)
args.number_of_stochastic_mp2_steps = int(args.number_of_stochastic_mp2_steps)


# check the input files
qe_xml = "./data-file-schema.xml"
assert os.path.isfile(qe_xml), "QE data-file-schema.xml file not found!"
wfc_hdf5 = "./wfc1.hdf5"
assert os.path.isfile(wfc_hdf5), "QE wfc1.hdf5 file not found!"

print("Reading QE calculation result from files ")
time_start0 = time.time()
# read wavefunction data form the xml and hdf5 files
latt9, eigenvalues, n_occupied, grid_point = get_qe_data(qe_xml)
psi_list, op_coul, g2vector_mask = read_qe_wavefunction(latt9, grid_point, len(eigenvalues), wfc_hdf5)
psi_list = np.array(psi_list)
n_unoccupied = len(psi_list) - n_occupied
volume = np.linalg.det(latt9)

print("\nWavefuncion information:")

print(f"Number of orbitals: {len(psi_list)} = {n_occupied} occupied + {n_unoccupied} unoccupied")

for  i in range(len(eigenvalues)):
    occ = "occupied" if i < n_occupied else "unoccupied"
    print(f"orbital {i:4d} : eigenvalue = {eigenvalues[i]:10.6f}  {occ}")

print("\n")
print(f"Grid points: {grid_point}")
print("cell volume: {:10.6f}".format(volume))
time_end = time.time()
print("Time used to read QE calculation result: {:6d} seconds".format(int(time_end - time_start0)))


if args.number_of_threads > 1:
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


# calculate tau and weight for laplace_mp2 and stochastic_mp2 method
if args.method in ['laplace_mp2', "stochastic_mp2"]:
    from laplace_mp2 import get_tau
    delta_eigen_max = eigenvalues[-1] - eigenvalues[0]
    delta_eigen_min = eigenvalues[n_occupied+1] - eigenvalues[n_occupied]

    # get the tau and weight list using the least square fitting method by
    # minimizing the error in the range [delta_eigen_min*0.9, delta_eigen_max+1]
    tau_list, w_list, error = get_tau(n_w=6, n_x=600, x_range=[delta_eigen_min*0.9, delta_eigen_max+1])
    
    print("Using laplace transformation with tau and weight: ")
    for tau, w in zip(tau_list, w_list):
        print("tau: {: 12.6f}   weight: {: 12.6f}".format(tau, w))

    print("The average error in range [{:10.6f},  {:10.6f}] is {:10.8f}".format(delta_eigen_min*0.9,
                                                     delta_eigen_max+1, error) )


if args.method == 'mp2':

    # serial implementation
    if args.number_of_threads == 1:
        from mp2 import get_mp2, get_mp2_parallel
        e_d, e_x = get_mp2(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied)
    else:
        from mp2 import get_mp2_parallel
        n_thread = args.number_of_threads
        from mp2 import get_phi_pp, task_parting
        e_d, e_x = get_mp2_parallel(psi_list, eigenvalues, g2vector_mask, op_coul, n_occupied)
        
    e_d = e_d / volume**2
    e_x = e_x / volume**2

    time_end = time.time()

    print(f"MP2 direct energy:   {e_d: 10.8f}")
    print(f"MP2 exchange energy: {-e_x: 10.8f}")
    print(f"MP2 total energy:    {e_d - e_x: 10.8f}")
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))

elif args.method == 'laplace_mp2':
    from laplace_mp2 import *

    # The psuedo code of the fig 1 in paper 'J. Chem. Phys. 146, 104101 (2017)'
    e_x_total = 0.0
    e_d_total = 0.0
    if args.number_of_threads == 1:
        # serial implementation
        from laplace_mp2 import laplace_mp2_energy
        e_d_total, e_x_total = laplace_mp2_energy(psi_list, eigenvalues, g2vector_mask, 
                                                  op_coul, n_occupied, tau_list, w_list)
    else:
        n_thread = args.number_of_threads
        # parallel implementation
        from laplace_mp2 import laplace_mp2_energy_pp

        e_d_total, e_x_total = laplace_mp2_energy_pp(psi_list, eigenvalues, g2vector_mask, op_coul, 
                                                     n_occupied, tau_list, w_list, n_thread)

    e_d_total = e_d_total / volume**2 * 2
    e_x_total = e_x_total / volume**2

    time_end = time.time()
    print(f"MP2 direct energy:   { e_d_total: 10.8f}")
    print(f"MP2 exchange energy: {-e_x_total: 10.8f}")
    print(f"MP2 total energy:    {e_d_total - e_x_total: 10.8f}")
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))

elif args.method == "stochastic_mp2":
    from stochastic_mp2 import *

    # The psuedo code of the fig 1 in paper 'J. Chem. Phys. 148, 064103 (2018)'
    n_stochastic = 1000
    n_repeat = int(args.number_of_stochastic_mp2_steps / n_stochastic)
    e_d_list = []
    e_x_list = []
    print(f"stochastic step     direct part      deltaE        exchange part   deltaE         total energy     deltaE ")

    n_thread = args.number_of_threads
    for i in range(n_repeat):
        
        e_d, e_x= get_stochastic_mp2_pp(psi_list, g2vector_mask, eigenvalues, op_coul, n_occupied,
                      tau_list,w_list, n_thread, round_num = n_stochastic)
        e_d_list.append(e_d)
        e_x_list.append(e_x)


        e_d_mean   = np.mean(e_d_list) / volume**2            
        e_x_mean   = np.mean(e_x_list) / volume**2
        
        if i==0:
            e_d_delta, e_x_delta = 0., 0.
        else:
            e_d_delta  = e_d_mean- np.mean(e_d_list[:i])  / volume**2
            e_x_delta  = e_x_mean - np.mean(e_x_list[:i])  / volume**2
        
        print("{: 10d}          {: 10.8f}   {: 10.8f}      {: 10.8f}   {: 10.8f}      {: 10.8f}   {: 10.8f}".format((i+1)*n_stochastic, e_d_mean, e_d_delta, 
          -e_x_mean, -e_x_delta, e_d_mean - e_x_mean, e_x_delta - e_d_delta  ))
 
    time_end = time.time()
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))
