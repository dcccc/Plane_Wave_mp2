
Read this in other languages: [日本語](readme_jp.md).

## Plane Wave MP2
A code to calculate the mp2 energy with plane-wave basis set wavefunction provided by Quantum Espresso (QE)[1]. The code is written with python, and the purpose is to get a better understanding about plane-wave basis method.

## How To Run
Before running the code to calculate the mp2 energy, the wavefunction file should be prepared using QE. The QE compiled with HDF5 library [2] should be used to save wavefunction files. After QE calculation, enter into the directory where wavefunction file "wfc1.hdf5" locate and run the calculation. A sample QE input file can be referred in example directory.  

At present, canonical mp2 method [3], laplace transformed mp2 method [4] and stochastic orbit mp2 method [5] are implemented.  

There are some examples:  

```
# the canonical mp2 method in serial 
python src/pwmp2.py -m mp2 

# the canonical mp2 method in parallel using 24 process
python src/pwmp2.py -m mp2 -n_thread  24

# the laplace transformed mp2 method in in serial 
python src/pwmp2.py -m laplace_mp2 

# the laplace transformed mp2 method in parallel using 24 process
python src/pwmp2.py -m laplace_mp2 -n_thread 24


# the stochastic orbit mp2 method in serial 
python src/pwmp2.py -m stochastic_mp2 

# the stochastic orbit mp2 method in parallel using 24 process
python src/pwmp2.py -m stochastic_mp2 -n_thread  24 --number_of_stochastic_mp2_steps 10000

```



## Possible Bugs

1. The code is not optimized at all, the calculation can be very slow. So, don't used it for production calculations. Use vasp instead.

2. Only norm-conserving pseudopotentials should be used, PAW or ultrasoft pseudopotentials are not supported.  

3. As code is not tested extensively, so the calculation result may be wrong.  

## Reference
[1] https://gitlab.com/QEF/q-e  
[2] https://github.com/HDFGroup/hdf5  
[3] Journal of Chemical Theory and Computation 2023 19 (24), 9211-9227  
[4] J. Chem. Phys. 146, 104101 (2017)  
[5] J. Chem. Phys. 148, 064103 (2018)  

