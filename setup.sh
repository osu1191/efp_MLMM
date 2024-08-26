#!/bin/csh

# setup.sh

# Prompt the user to enter the paths or set them manually here
setenv LIBEFP_DIR "/depot/lslipche/data/skp/torch_skp_branch/libefp"
setenv LIBTORCH_INCLUDE_DIRS "/depot/lslipche/data/skp/libtorch/include/;/depot/lslipche/data/skp/libtorch/include/torch/csrc/api/include"
setenv TORCH_INSTALLED_DIR "/depot/lslipche/data/skp/libtorch"

echo "Environment variables set:"
echo "LIBEFP_DIR=$LIBEFP_DIR"
echo "LIBTORCH_INCLUDE_DIRS=$LIBTORCH_INCLUDE_DIRS"
echo "TORCH_INSTALLED_DIR=$TORCH_INSTALLED_DIR"
