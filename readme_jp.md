このコードはQuantum Espresso (QE)[1]生成した平面波関数を利用して、MP2エネルギー計算する。コードはpythonで完成した。MP2方法の実現を通して、平面波関数について方法のもと深い理解が得る  

## コードの実行
コードを実行する前に、まずはQE利用して波動関数を用意するべき。QEはHDF5のサポートが必要です。QE計算完了後、波動関数「wfc1.hdf5」が保存された位置へ移動し、コードを実行する。参考例のQE入力ファイルは「example」にある  

今まで、標準mp2法[3]、ラプラス変換mp2法[4]、確率軌道mp2法[5]が実現されている  

以下はコードを実行する例

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



## コードのバグ
1.このコードはただMP2計算方法が実現する、計算の適化されていないため、計算は非常に遅くかもしれない。本番計算の時はこのコードを使わないで、代わりに、VASPは正解  
2.ノルム保存擬似ポテンシャルのみを使用する必要がある。PAWとultrasoft擬似ポテンシャルはサポートさてはない  
3.	コードのテストは少ないため、計算結果は間違い可能性がある  

## 参考文献
[1] https://gitlab.com/QEF/q-e  
[2] https://github.com/HDFGroup/hdf5  
[3] Journal of Chemical Theory and Computation 2023 19 (24), 9211-9227  
[4] J. Chem. Phys. 146, 104101 (2017)  
[5] J. Chem. Phys. 148, 064103 (2018)  