LibTORCH + LIBEFP


 Added routines :
=====================
nnlib -> contains the scripted neural network potential files
efpmd/src/torch.c
efpmd/src/torch.h
efpmd/torch -> contains the wrapper libraries for torch-gradient computation routine
CMakeLists.txt -> connection between torch and EFP


 Required modules :
=====================
 
cmake/3.18 or higher
openblas/0.3.8
netlib-lapack/3.8.0
gcc/9.3 or higher
 
The modules can be placed in module.sh. Run "source module.sh" to load these modules.


Setup LibTorch and LibEFP variables
===================================
 
>Install LibTorch (C++ frontend) of PyTorch. If not, just gathering the lib/ and include/ directories are good enough.
>Next we need to setup a few environment varibales. Libtorch and LibEFP directory
addresses needs to be changed. In "setup.sh" change the following :

> Turn the TORCH_SWITCH = ON for configuration with the torch-routines
>setenv LIBEFP_DIR "/PATH/TO/libefp"
>setenv LIBTORCH_INCLUDE_DIRS "/PATH/TO/libtorch/include/;/PATH/TO/libtorch/include/torch/csrc/api/include"
>setenv TORCH_INSTALLED_DIR "/PATH/TO/libtorch"

**for bash shell use export command instead of setenv>


Compile and run
===============

Run "./compile.sh" to compile.

Executable "efpmd" are in build/efpmd/efpmd

Run test files as :
  
"build/efpmd/efpmd tests/input.in"
