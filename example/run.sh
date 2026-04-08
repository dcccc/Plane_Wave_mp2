
# run the qe calculation
pw.x < NH3_100.pw.in > NH3_100.pw.out

# enter the 'NH3.save' directory where the wavefunction file stored
cd NH3.save

# run the pwmp2 code
python ../../src/pwmp2.py -m mp2  -n_thread 16


