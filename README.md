# Human Identification from ECG signals via sparse representation of local segments using optmization methods like ADMM and PROXIMAL Algorithm.


This project implements a method to extract compact and discriminative features from Electrocardiogram (ECG) signals for human identification based on the sparse representation of local segments.
Specifically, local segments are extracted from the ECG signals and projected to a small number of basic elements in a dictionary, which is an orthogonal random matrix.  
Finally, the features are extracted by performing max pooling procedure over all the spares coefficient vectors from the ECG signals.
Three optimization methods are used namely ADMM, Proximal Point Algrithm and final verified with CVX, to find the sparse coefficient vector from the ECG signals, and their performances are compared in this report.
This method achieves a 96.00% accuracy on a 10 subject dataset.  



## Getting Started

The code for the project implemented using ADMM is in file "Code_ADMM_method.mat" 
In the file "Final Code.mat", you can find code for optimization using all the method such as ADMM, Proximal Point Algrithm and CVX.


Enter the values for the following varible before running the main program, default values are given in program:

Nopatient:             Number of patients
Nseg:                  Total number of training and testing segments for patients
ntr:                   Number of train segments for patients
nte:                   Number of test segments for patients
K:                     Size of dictonary is d by K
st1,st2,st3:           Initial random states.
d:                     Length of window used to create a local segment
gp: number of samples a local segment differs from the next local segment.


### Results

A sample ECG signal segment:

![ECG image](https://github.com/alinstein/Human-Identification-with-ECG--/blob/master/observation/ecg.jpg)

Final sparse coefficient vector representation of the above ECG signal:

![Sparse coefficient](https://github.com/alinstein/Human-Identification-with-ECG--/blob/master/observation/maxpol2.jpg)
Following table shows the results obtained:

![Result image](https://github.com/alinstein/Human-Identification-with-ECG--/blob/master/Results.JPG)

## References

[1] J. Wang, M. She, S. Nahavandi and A. Kouzani, "Human Identification From ECG Signals Via Sparse Representation of Local Segments," in IEEE Signal Processing Letters, vol. 20, no. 10, pp. 937-940, Oct. 2013.
doi: 10.1109/LSP.2013.2267593
[2] W.-S. Lu, Course notes of Advanced Mathematical optimizations. 

## Authors

%Written by Alinstein Jose, University of Victoria.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


