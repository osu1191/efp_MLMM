/*-
 * Copyright (c) 2012-2015 Ilya Kaliman
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "torch.h"
//#include "common.h"
#include "cfg.h"
#include <stdio.h>
#include "state.h"

/*
struct torch {
    double energy;
    double *grad;
    size_t natoms;
    int *atom_types;
    double *atom_coords;
    int nn_type;
    ANIState global_state;
};
*/

struct torch *torch_create(void) {
    struct torch *torch;
    torch = calloc(1, sizeof(*torch));
    return (torch);
}

void get_torch_type(struct torch *torch, const char *str) {
    int file_type;
    if (strcmp(str, "ani1.pt") == 0) {
        file_type = 1;
        fprintf(stderr, "chosen nn_type: %s\n", str);
    } else if (strcmp(str, "ani2.pt") == 0) {
        file_type = 2;
        fprintf(stderr, "chosen nn_type: %s\n", str);
    } else {
        file_type = -1; // or any other default/error value
        fprintf(stderr, "Unknown filetype: %s\n", str);
    }
    torch->nn_type = file_type;
}

void torch_init(struct torch *torch, size_t natom) {
    torch->natoms = natom;
    torch->atom_coords = malloc(3*natom*sizeof(double));
    torch->atom_types = malloc(natom*sizeof(int));
    torch->grad = malloc(3*natom*sizeof(double));
    torch->elpot = malloc(natom*sizeof(double));
} 
 
//int torch_load_nn(struct torch *torch, const char *nn_name) {

void torch_get_atom_count(struct torch *torch , size_t natom) {
    natom = torch->natoms;
}

void torch_set_atom_count(struct torch *torch, size_t *natom) {
    torch->natoms = *natom;
}

void torch_get_atom_coord(struct torch *torch, size_t atom, double *coord) {
    assert(atom < torch->natoms);
    memcpy(coord, torch->atom_coords + (atom * 3), 3*sizeof(double));
}

void torch_set_atom_coord(struct torch *torch, size_t atom, const double *coord) {
    assert(atom < torch->natoms);
    memcpy(torch->atom_coords + (atom * 3), coord, 3*sizeof(double));
}

void torch_get_coord(struct torch *torch, double *coords) {
    memcpy(coords, torch->atom_coords, (3 * torch->natoms) * sizeof(double));
}

void torch_set_coord(struct torch *torch, const double *coords) {
    memcpy(torch->atom_coords, coords, (3 * torch->natoms) * sizeof(double));
}

void torch_set_elpot(struct torch *torch, const double *spec_elpot) {
    memcpy(torch->elpot, spec_elpot, torch->natoms * sizeof(double));
}


void torch_set_atom_species(struct torch *torch, const int *atom_z) {
    memcpy(torch->atom_types, atom_z, (torch->natoms) * sizeof(int));
}

/*
void torch_custom() {

    int num_atoms = 5;
    float *gradients, *forces;
    float energy;
   
    gradients = malloc(num_atoms * 3 * sizeof(float));
    forces = malloc(num_atoms * 3 * sizeof(float));

    float coordinates_data[15] = {
        0.03192167, 0.00638559, 0.01301679,
        -0.83140486, 0.39370209, -0.26395324,
        -0.66518241, -0.84461308, 0.20759389,
        0.45554739, 0.54289633, 0.81170881,
        0.66091919, -0.16799635, -0.91037834
    };
 
    int64_t species_data[5] = {1, 0, 0, 0, 0};
    float elecpots_data[5] = {1.0, 0.0, 2.0, 1.0, 2.0};

    //float gradients[15];
    //float forces[15];

    engrad3_custom_model_wrapper(coordinates_data, species_data, elecpots_data, num_atoms, &energy, gradients, forces);

    printf("Gradients:\n");
        for (int i = 0; i < num_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", gradients[i * 3 + j]);
            }
            printf("\n");
        }

        printf("Forces:\n");
        for (int i = 0; i < num_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", forces[i * 3 + j]);
            }
            printf("\n");
        }
	
     free(gradients);
    free(forces);

}
*/

void torch_custom_compute(struct torch *torch, int print) {

    size_t n_atoms = torch->natoms;
    float *gradients, *forces; 
    float *frag_coord;
    float *elecpots_data;
    float custom_energy;

    elecpots_data = malloc(n_atoms * sizeof(float));
    //energies = malloc(n_atoms * sizeof(float));
    gradients = malloc(n_atoms * 3 * sizeof(float));
    forces = malloc(n_atoms * 3 * sizeof(float));

    frag_coord = malloc(n_atoms*3* sizeof(float));

    for (size_t i=0; i<n_atoms; i++) {
        frag_coord[i*3] = (float)(torch->atom_coords[i*3] * BOHR_RADIUS);
        frag_coord[i*3+1] = (float)(torch->atom_coords[i*3+1] * BOHR_RADIUS);
        frag_coord[i*3+2] = (float)(torch->atom_coords[i*3+2] * BOHR_RADIUS);
	
	elecpots_data[i] = (float) torch->elpot[i];
    }

//    printf("=============TORCH ELPOT=============\n");
//    for (size_t j = 0; j < n_atoms; j++) {
//	printf("%12.6f %12.6f\n",torch->elpot[j], elecpots_data[j]);
//    } 
//    printf("====================================\n");

    int atomic_num[n_atoms];
    for (size_t i=0; i<n_atoms; i++) {
       atomic_num[i] = torch->atom_types[i];
    }
    
    int64_t frag_species[n_atoms];
    atomic_number_to_species(atomic_num, frag_species, n_atoms);    

    // feed torch->elpot hear
    // convert double to float
    // or form torch->elpot as a float

    //float elecpots_data[3] = {1.0, 0.0, 2.0};
//    float *elecpots_data;
//    elecpots_data = malloc(n_atoms * sizeof(float));

    engrad3_custom_model_wrapper(frag_coord, frag_species, elecpots_data, n_atoms, &custom_energy, gradients, forces);

    torch->energy = custom_energy;
//    printf("Energy in torch_custom2 = %12.6f \n",torch->energy);

    if (print > 1) {
        printf("Gradients:\n");
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", gradients[i * 3 + j]);
            }
            printf("\n");
        }

        printf("Forces:\n");
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", forces[i * 3 + j]);
            }
            printf("\n");
        }
    }

    double *tG_double = xcalloc(3 * n_atoms, sizeof(double));

    for (int i = 0; i < 3 * n_atoms; i++) {
        tG_double[i] = (double)(gradients[i] * BOHR_RADIUS);
    }

    memcpy(torch->grad, tG_double, (3 * n_atoms) * sizeof(double)); // Atomistic gradient for the EFP-ML fragment

    torch_print(torch);
    free(gradients);
    free(forces);
    free(frag_coord);
    free(tG_double);
 
}


void atomic_number_to_species(const int* atomic_num, int64_t* frag_species, size_t n_atoms) {

    for (size_t i = 0; i < n_atoms; i++) {
        switch (atomic_num[i]) {
            case 1:  // Hydrogen
                frag_species[i] = 0;
                break;
            case 6:  // Carbon
                frag_species[i] = 1;
                break;
            case 7:  // Nitrogen
                frag_species[i] = 2;
                break;
            case 8:  // Oxygen
                frag_species[i] = 3;
                break;
            default:

      	    frag_species[i] = -1; 
                printf("Warning: Unknown atomic number %d\n", atomic_num[i]);
                break;
        }
    }
}

// SKP's torch version
void torch_compute(struct torch *torch, int print) {

    // prepare data arrays 
    // msg("\n TORCH CALL \n---------------------------------\n");

    //struct torch_state *torch_state;

//    torch_custom();

    torch->global_state.model = ANIModel_new();
    load_ani_model(torch->global_state.model, torch->nn_type); 
 
    size_t n_atoms = torch->natoms;
    float *energies, *gradients, *forces, *frag_coord;

    energies = malloc(n_atoms * sizeof(float));
    gradients = malloc(n_atoms * 3 * sizeof(float));
    forces = malloc(n_atoms * 3 * sizeof(float));

    // copy from double to float
    frag_coord = malloc(n_atoms*3* sizeof(float));
    for (size_t i=0; i<n_atoms; i++) {
        frag_coord[i*3] = (float)(torch->atom_coords[i*3] * BOHR_RADIUS);
        frag_coord[i*3+1] = (float)(torch->atom_coords[i*3+1] * BOHR_RADIUS);
        frag_coord[i*3+2] = (float)(torch->atom_coords[i*3+2] * BOHR_RADIUS);
    }

    int frag_species[n_atoms];
        for (size_t i=0; i<n_atoms; i++) {
           frag_species[i] = torch->atom_types[i];
 	    //printf("atom_types in torch_compute %4d\n",torch->atom_types[i]);
    }

    double total_energy = 0.0;

    // call function
    get_torch_energy_grad(frag_coord, frag_species, n_atoms, energies, gradients, forces, torch->nn_type);
 
    get_ani_energy_grad(torch->global_state.model, frag_coord, frag_species, energies, gradients, forces, n_atoms);     
 
    //get_ani_energy_grad(torch->ani_model, frag_coord, frag_species, energies, gradients, forces, n_atoms);
   // torch-ani_model  instead of torch->global_state.model

    // printf("   Special fragment Atomic Energies, Coordinates, Gradients in H, A, H/A \n----------------------------------\n");
    for (int i = 0; i < n_atoms; ++i) {

        // printf("%4d   %12.6f     %12.6f %12.6f %12.6f    %12.6f %12.6f %12.6f\n",
        //     torch->atom_types[i],energies[i], 
        //    frag_coord[3*i], frag_coord[3*i+1], frag_coord[3*i+2],
        //    gradients[3*i], gradients[3*i+1], gradients[3*i+2]);
        
        total_energy  += (double)energies[i];
    }
    //printf("   Total TORCH energy %12.6f\n", total_energy);
    //printf("----------------------------------\n\n");
    
    torch->energy = total_energy;

    if (print > 1) {
        printf("Gradients:\n");
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", gradients[i * 3 + j]);
            }
            printf("\n");
        }

        printf("Forces:\n");
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f\t", forces[i * 3 + j]);
            }
            printf("\n");
        }
    }

    // save data in energy and grad
    // convert the gradients from float to double and to Hartree/Bohr
    double *tG_double = xcalloc(3 * n_atoms, sizeof(double));
    
    for (int i = 0; i < 3 * n_atoms; i++) {
        tG_double[i] = (double)(gradients[i] * BOHR_RADIUS);
    }

    memcpy(torch->grad, tG_double, (3 * n_atoms) * sizeof(double)); // Atomistic gradient for the EFP-ML fragment

    ANIModel_delete(torch->global_state.model);
    // if (print> 0) torch_print(torch);
    torch_print(torch);
    free(energies);
    free(gradients);
    free(forces);
    free(frag_coord);
    free(tG_double);
    //free(torch_state);
}


// LS's version
/*
void torch_compute(struct torch *torch, int do_grad) {
    static int iter = 0;

    if (iter == 0) {
        for (size_t g=0; g<torch->natoms*3; g++) {
            torch->grad[g] = 0.1;
        }
        torch->energy = -55.0;
    }
    else {
        for (size_t g = 0; g<torch->natoms * 3; g++) {
            torch->grad[g] = -torch->grad[g] * 0.5;
        }
        torch->energy = torch->energy - 0.1 / (iter+1);
    }
    iter++;
}
*/

double torch_get_energy(struct torch *torch) {
    return torch->energy;
}
void torch_get_gradient(struct torch *torch, double *grad) {
    memcpy(grad, torch->grad, (3 * torch->natoms) * sizeof(double));
}

void torch_free(struct torch *torch) {
    if (torch) {
        free(torch->grad);
        free(torch->atom_coords);
        free(torch->atom_types);
        free(torch);
    }
}

void torch_print(struct torch *torch) {
    if (torch) {
        printf("\n TORCH INFO \n");
	printf("-----------\n");
        printf("\n Special fragment coordinates (Angstroms) \n", torch->natoms);
	printf("-----------------------------------------------------------\n");
	printf("  Atom            X                 Y                Z\n");
        for (size_t i=0; i< torch->natoms; i++) {
            printf("%4d      %12.6f      %12.6f     %12.6f\n", torch->atom_types[i], torch->atom_coords[i*3] * BOHR_RADIUS,
                   torch->atom_coords[i*3+1] * BOHR_RADIUS, torch->atom_coords[i*3+2] * BOHR_RADIUS);	    
        }
	printf("-----------------------------------------------------------\n");
        printf("\n Special fragment atom gradients \n", torch->natoms);
	printf("-----------------------------------------------------------\n");
            printf("  Atom            X                 Y                Z\n");
        for (size_t i=0; i< torch->natoms; i++) {
            printf("%4d      %12.6f      %12.6f     %12.6f\n",i+1,torch->grad[i*3],
                   torch->grad[i*3+1], torch->grad[i*3+2]);
        }
	printf("------------------------------------------------------------\n");
        printf("\n Torch energy %lf \n\n", torch->energy);
    }
}
