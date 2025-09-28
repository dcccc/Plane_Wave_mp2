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
print("\n")

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



if args.method in ['laplace_mp2', "mp2"] and args.number_of_threads > 1:
    from mp2 import task_parting
    n_thread = args.number_of_threads
    def psi_g2r(range_list):
        return([np.fft.ifftn(i) for i in psi_list[range_list[0]:range_list[1]]])    
    
    task_idx_list = task_parting(n_unoccupied+n_occupied, n_thread)
    pool = mp.Pool(n_thread)

    results = [pool.apply_async(psi_g2r, args=(task_idx_list[i],))  
                   for i in range(n_thread) ]    
    pool.close()
    pool.join()

    psi_r = np.vstack([p.get() for p in results])
time_end = time.time()
print("Time used to calculate psi_r:       {:10d} seconds".format(int(time_end - time_start0)))


if args.method == 'mp2':
    psi_r = np.array([np.fft.ifftn(i) for i in psi_list])
    psi_list = []

    # serial implementation
    if args.number_of_threads == 1:
        from mp2 import get_phi, get_eri, get_e

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
    else:
        n_thread = args.number_of_threads
        from mp2 import get_phi_pp, task_parting
        # parallel implementation
        pool = mp.Pool(n_thread)

        # divide the tasks into n_thread parts, each part has approximately n_task/n_thread tasks
        # the number of unoccupied orbitals is always larger than number of the occupied orbitals
        task_idx_list = task_parting(n_unoccupied, n_thread)
        
        # calculate phi_ia in parallel
        time_start = time.time()
        results = [pool.apply_async(get_phi_pp, args=(psi_r, n_occupied, g2vector_mask, task_idx_list[i],))  
                   for i in range(n_thread) ]    
        pool.close()
        pool.join()
    
        # get the results
        phi_ia = np.hstack([p.get() for p in results])
        time_end = time.time()
        print("Time used to calculate phi_ia:        {:10d} seconds".format(int(time_end - time_start)))
        psi_r = []

        pool = mp.Pool(n_thread)

        from mp2 import get_eri_pp
        # calculate eri in parallel
        time_start = time.time()
        results = [pool.apply_async(get_eri_pp, args=(phi_ia, op_coul, task_idx_list[i],))  
               for i in range(n_thread) ]
    
        pool.close()
        pool.join()

        # get the results
        eri = np.vstack([p.get() for p in results])    
        eri = eri.transpose(1,0,2,3)
        phi_ia = []
        time_end = time.time()
        print("Time used to calculate eri:           {:10d} seconds".format(int(time_end - time_start)))



        from mp2 import get_emp2_pp

        time_start = time.time()
        pool = mp.Pool(n_thread)
        # calculate energy in parallel
        results = [pool.apply_async(get_emp2_pp, args=(eri, eigenvalues, task_idx_list[i],))  
               for i in range(n_thread) ]
    
        pool.close()
        pool.join()
        eri = []
        emp2 = np.array([p.get() for p in results])
        e_d, e_x = np.sum(emp2, axis=0) 
        time_end = time.time()
        print("Time used to get mp2 energy:          {:10d} seconds".format(int(time_end - time_start)))
        
    e_d = e_d / volume**2
    e_x = e_x / volume**2
    
    print(f"MP2 direct energy:   {e_d: 10.8f}")
    print(f"MP2 exchange energy: {-e_x: 10.8f}")
    print(f"MP2 total energy:    {e_d - e_x: 10.8f}")
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))

elif args.method == 'laplace_mp2':
    from laplace_mp2 import *
    from mp2 import task_parting, get_phi


    #psi_r = np.array([np.fft.ifftn(i) for i in psi_list])
    

    # The psuedo code of the fig 1 in paper 'J. Chem. Phys. 146, 104101 (2017)'
    e_x_total = 0.0
    e_d_total = 0.0
    if args.number_of_threads == 1:
        # calculate phi_ia
        time_start = time.time()
        phi_ia = get_phi(psi_r, n_occupied, g2vector_mask)
        time_end = time.time()
        psi_r = []
        print("Time used to calculate phi_ia:        {:10d} seconds".format(int(time_end - time_start)))


        # serial implementation

        time_start = time.time()
        for tau, w in zip(tau_list, w_list):        
            for ng in range(len(op_coul)):
                w_psi_list = get_w_psi(psi_list, phi_ia, eigenvalues, g2vector_mask, ng,  tau=tau)
                e_d, e_x= get_lmp2(psi_list, w_psi_list, op_coul, g2vector_mask)
                e_d_total += e_d * w * op_coul[ng]
                e_x_total += e_x * w * op_coul[ng]

        time_end = time.time()
        print("Time used to calculate laplace mp2 energy: {:10d} seconds".format(int(time_end - time_start)))

    else:
    
        n_thread = args.number_of_threads
        from mp2 import get_phi_pp, task_parting
        # parallel implementation
        pool = mp.Pool(n_thread)

        # divide the tasks into n_thread parts, each part has approximately n_task/n_thread tasks
        # the number of unoccupied orbitals is always larger than number of the occupied orbitals
        task_idx_list = task_parting(n_unoccupied, n_thread)
        
        # calculate phi_ia in parallel
        time_start = time.time()
        results = [pool.apply_async(get_phi_pp, args=(psi_r, n_occupied, g2vector_mask, task_idx_list[i],))  
                   for i in range(n_thread) ]    
        pool.close()
        pool.join()
    
        # get the results
        phi_ia = np.hstack([p.get() for p in results])
        psi_r = []
        time_end = time.time()
        print("Time used to calculate phi_ia:        {:10d} seconds".format(int(time_end - time_start)))
        
        time_start = time.time()
        n_thread = args.number_of_threads
        # parallel implementation
        pool = mp.Pool(n_thread)

        # divide the tasks into n_thread parts, each part has approximately n_task/n_thread tasks
        # the number of unoccupied orbitals is always larger than number of the occupied orbitals
        task_idx_list = task_parting(len(op_coul), n_thread)
        print(task_idx_list)
        # calculate phi_ia in parallel
        results = [pool.apply_async(get_lmp2_pp, 
                   args=(psi_list, phi_ia, tau_list, w_list, eigenvalues, g2vector_mask, op_coul, task_idx_list[i],))  
                   for i in range(n_thread) ]    
        pool.close()
        pool.join()

        emp2 = np.array([p.get() for p in results])
        e_d_total, e_x_total = np.sum(emp2, axis=0) 
        time_end = time.time()
        print("Time used to calculate laplace mp2 energy: {:10d} seconds".format(int(time_end - time_start)))

    e_d_total = e_d_total / volume**2 * 2
    e_x_total = e_x_total / volume**2

    print(f"MP2 direct energy:   { e_d_total: 10.8f}")
    print(f"MP2 exchange energy: {-e_x_total: 10.8f}")
    print(f"MP2 total energy:    {e_d_total - e_x_total: 10.8f}")
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))

elif args.method == "stochastic_mp2":
    from stochastic_mp2 import *
    from laplace_mp2 import get_tau

    # The psuedo code of the fig 1 in paper 'J. Chem. Phys. 148, 064103 (2018)'
    n_stochastic = 100
    n_repeat = int(args.number_of_stochastic_mp2_steps / n_stochastic)
    e_d_list = []
    e_x_list = []
    print(f"stochastic step     direct part      deltaE        exchange part   deltaE         total energy     deltaE ")
    if args.number_of_threads == 1:
        for i in range(n_repeat):
            
            e_d, e_x = get_stochastic_mp2(psi_list,n_occupied, tau_list, w_list, op_coul, g2vector_mask, eigenvalues, n_stochastic)
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
    else:
        n_thread = args.number_of_threads
        # calculate energy in parallel
        part_step = n_stochastic // n_thread
        for i in range(n_repeat):
            pool = mp.Pool(n_thread)
            results = [pool.apply_async(get_stochastic_mp2, args=(psi_list,n_occupied, tau_list, w_list, op_coul, g2vector_mask, eigenvalues, part_step))  
                       for i in range(n_thread) ]    
            pool.close()
            pool.join()
            
            e_mp2 = np.array([p.get() for p in results])
            e_d_list.append(e_mp2[:,0].reshape(-1,))
            e_x_list.append(e_mp2[:,1].reshape(-1,))


            e_d_mean   = np.mean(e_d_list) / volume**2            
            e_x_mean   = np.mean(e_x_list) / volume**2
            
            if i==0:
                e_d_delta, e_x_delta = 0., 0.
            else:
                e_d_delta  = e_d_mean - np.mean(e_d_list[:i])  / volume**2
                e_x_delta  = e_x_mean - np.mean(e_x_list[:i])  / volume**2
            
            print("{: 10d}          {: 10.8f}   {: 10.8f}      {: 10.8f}   {: 10.8f}      {: 10.8f}   {: 10.8f}".format((i+1)*n_stochastic, e_d_mean, e_d_delta, 
              -e_x_mean, -e_x_delta, e_d_mean - e_x_mean, e_x_delta - e_d_delta  ))

    time_end = time.time()
    print("Total time used in mp2 calculation:  {:10d} seconds".format(int(time_end - time_start0)))
