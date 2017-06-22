/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

// includes, system
#include <iostream>
#include <stdlib.h>
#include <memory>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" bool runTest(const int argc, const char **argv,
                        char *data, int2 *data_int2, unsigned int len);
extern "C" bool runAcousticTest(const int argc, const char **argv,
	float3 *rays, unsigned int n_rays, float3 *atm_layers, unsigned int n_atm_layers);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    // input data
    int len = 16;
    // the data has some zero padding at the end so that the size is a multiple of
    // four, this simplifies the processing as each thread can process four
    // elements (which is necessary to avoid bank conflicts) but no branching is
    // necessary to avoid out of bounds reads
    char str[] = { 82, 111, 118, 118, 121, 42, 97, 121, 124, 118, 110, 56,
                   10, 10, 10, 10
                 };

	// emulate pattern from sample code
	// init angles
	int n_rays = 6;
	float angles[6] = {0., 15., 30., 45., 60., 75.};
	float3 prop_rays[6];
	for (int i = 0; i < n_rays; i++) {
		prop_rays[i].x = angles[i] * 3.14159265359f / 180.f;
		prop_rays[i].y = 0.;            // init propagated attenuation to 0f for all angles
		prop_rays[i].z = 0.;            // init propagated radii to 0f for all angles
	}
    std::unique_ptr<float3[]> rays(new float3[n_rays]);
	// init layers
	int n_atm_layers = 2;
	float wavefront_velocities[2] = {325., 361.};   // m/s
	float layer_attn[2] = {0.000424f, 0.000362f};	// dB/m @ 1atm
	float h_atm[2] = { 2000.f, 3000.f };            // m
	float3 atm_layers[2];
	for (int i = 0; i < n_atm_layers; i++) {
		atm_layers[i].x = wavefront_velocities[i];
		atm_layers[i].y = layer_attn[i];
        atm_layers[i].z = h_atm[i];
	}

    // Use int2 showing that CUDA vector types can be used in cpp code
    int2 i2[16];

    for (int i = 0; i < len; i++)
    {
        i2[i].x = str[i];
        i2[i].y = 10;
    }

    bool bTestResult;

    // run the device part of the program
    bTestResult = runAcousticTest(argc, (const char **)argv, prop_rays, n_rays, atm_layers, n_atm_layers);
        
    for (int i_ray = 0; i_ray < n_rays; i_ray++)
    {
        std::printf("%f %f %f %f\n", angles[i_ray], prop_rays[i_ray].x, prop_rays[i_ray].y, prop_rays[i_ray].z);
    }

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
