


# run the qe calculation
pw.x < NH3_100.pw.in > NH3_100.pw.out

# enter the 'st' directory where the wavefunction files are stored
cd st

# run the pwmp2 code
python ../../src/pwmp2.py -m mp2 

