# Tensor
This is a templated tensor class for tensor network calculations.
Tensor supports operation such as:
* arithmetic of tensors and numbers
* tensor contractions using either of: TBLIS, TCL or _GEMM (can be selected via environment variable)
* matrix decomposition such as: qr, svd, eig, eigh

Dependencies:
* TBLIS (https://github.com/devinamatthews/tblis)
* TCL (https://github.com/springer13/tcl)
* HTPP (https://github.com/springer13/hptt)
* BLAS
* LAPACK

A more thorough documentation will be written at a later stage 
