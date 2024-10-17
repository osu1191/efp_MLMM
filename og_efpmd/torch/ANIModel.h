#ifndef ANIMODEL_H
#define ANIMODEL_H

#include <torch/torch.h>
#include <torch/script.h>
#include <stdio.h>

//===== class ANIModel ==== //
 
class ANIModel {
public:
    //ANIModel() = default;
    ANIModel() {printf("\nANIModel() Constructor\n");}
    ~ANIModel() {printf("\nANIModel() Destructor\n");}
    void load_model(int model_type, const std::string &nn_path);
    void get_energy_grad(const torch::Tensor& coordinates, const torch::Tensor& species, float* atomic_energies, float* gradients, float* forces, int num_atoms);

private:
    torch::jit::Module module;
};

#endif // ANIMODEL_H
