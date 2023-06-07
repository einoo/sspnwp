# sspnwp
Sparse sensor placement for improving numerical weather prediction

### Sparse sensor placement

The codes of sparse sensor placement for selecing the observation stations of the SPEEDY-LETKF mode are in the *./src/* folder. 

- D_optimization_SSP.py corresponds to the D-optimization (maximize the determinant of Fisher information matrix) in sparse sensor placement.
- A_optimization_SSP.py corresponds to the A-optimization (maximize the trace of the Fisher information matrix) in sparse sensor placement.
- E_optimization_SSP.py corresponds to the E-optimization (maximize the minimum eigenvalue of the Fisher information matrix) in sparse sensor placement.

### SPEEDY-LETKF model

The existing observation stations *raob* is **staion_raob.tbl** in the *./data/* folder. 

### Data for reproducing the figures

The root mean square error and the standard deviation of the experiments are stored in the *./data/* folder.
The selected observation stations based on the uniform distribution are stored in the *./data/Unif/* folder.
The selected observation stations by sparse sensor placement are stored in the *./data/Network/* folder.

The scripts for reproducing the figures are stored in the *./src/fig_* scripts.
All the figures can be reproduced by those scripts.
The resulted figures are in the *./figure/* folder. 

I hope the data provide here could be of some help to the research for selecting optimal observation stations in scientific research. 
I always welcome any questions and discussions. Please email me: ouyang.mao@chiba-u.jp.
