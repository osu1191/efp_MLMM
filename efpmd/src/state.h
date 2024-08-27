#ifndef STATE_H
#define STATE_H

#include "../torch/c_libtorch.h"

typedef struct {
    ANIModel* model;
} ANIState;

extern ANIState global_state;

//ANIModel* ANIModel_new();
//void ANIModel_load(ANIModel* model, int model_type);
//void ANIModel_get_energy_grad(ANIModel* model, const torch::Tensor* coordinates, const torch::Tensor* species, float* atomic_energies, float* gradients, float* forces, int num_atoms);

#endif // STATE_H
