#ifndef C_LIBTORCH_H_
#define C_LIBTORCH_H_

#include <stdint.h>

//#include <torch/script.h>
//#include <torch/torch.h>

/*
class ANIModel {
public:
    ANIModel() = default;
    void load_model(int model_type);
    void get_energy_grad(const torch::Tensor& coordinates, 
                         const torch::Tensor& species, 
                         float* atomic_energies, 
                         float* gradients, 
                         float* forces, 
                         int num_atoms);

private:
    torch::jit::Module module;
};
*/

#ifdef __cplusplus
extern "C" {
#endif

struct TensorData;  // Opaque

// TODO: Remove
// SKP-==========//

struct Tensor{
    void* data;
    int64_t* sizes;
    int ndim;
    int type_id;
    int is_variable;
};


void *compute_gradient_c(float* data, int64_t* sizes, int ndim);
void destroy_tensor(struct Tensor *tensor);

typedef struct Net Net;
Net *createNet();
void destroyNet(Net *model);
void forward(Net* model, const float *inputs, float *output, int input_size, int output_size);
void trainModelWrapper(Net *model, const float **input_data, const float *target_data, int num_samples, int num_epochs, float learning_rate);
void generateEnergyWrapper(Net *model, const float **input_data, int batch_size, int input_size);


// June-6th

void *loadModelWrapper(const char *modelPath);
void generateEnergyForcesWrapper(const void* model, const float* const* coordinates, int num_atoms, float* energy, float* const* forces);
void generateSpeciesEnergyForcesWrapper(const void* model, const float* const* coordinates, const int* species, int num_atoms, float* energy, float* const* forces);

void get_torch_energy_grad(float* coordinates_data, int* species_data, int num_atoms, float *atomic_energies, float *gradients, float *forces, int model_type);


// ================== //
// ==== Wrappers ==== //

typedef struct ANIModel ANIModel;

ANIModel* ANIModel_new();
void load_ani_model(ANIModel* model, int model_type);
void get_ani_energy_grad(ANIModel* mode, float* coordinates, int* species, float* atomic_energies, float* gradients, float* forces, int num_atoms);
void ANIModel_delete(ANIModel* model);
 
//================// 


#ifdef __cplusplus
}
#endif


#endif // C_LIBTORCH_H_