////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

extern "C" void computeGold(char *reference, char *idata, const unsigned int len);
extern "C" void computeGold2(int2 *reference, int2 *idata, const unsigned int len);

///////////////////////////////////////////////////////////////////////////////
//! refract ray with Snell's
//! @param theta1  incident angle of incident ray
//! @param v1 wavefront velocity of incident ray
//! @param v2 wavefront velocity of refracted ray
///////////////////////////////////////////////////////////////////////////////
__device__ float
refract(float theta1, float v1, float v2)
{
    float arg_asin = v2 * sin(theta1) / v1;
    return asin(arg_asin);
}

///////////////////////////////////////////////////////////////////////////////
//! Demonstration of acoustic ray trace
//! @param rays  rays to trace (in and out)
//! @param atms
//! @param n_atms
///////////////////////////////////////////////////////////////////////////////
__global__ void
acoustic_trace_kernel(float3 *rays, float3 *atms, const int n_atms)
{
	// get current ray ID
	const unsigned int tid = threadIdx.x;
	float3 ray = rays[tid];
    if (tid == 5) {
        std::printf("%f %f %f\n", ray.x, ray.y, ray.z);
    }
	for (int i_atm = 0; i_atm < n_atms; i_atm++) {
        float h_layer = atms[i_atm].z;
        float d_layer = h_layer / cos(ray.x);
        // update attentuation
        ray.y -= d_layer * atms[i_atm].y + 20.0 * log10(d_layer);
        // update projected radii/ ground distance
        ray.z += sqrt(d_layer*d_layer - h_layer*h_layer);
        // update angle
        if (i_atm < n_atms - 1)
            ray.x = refract(ray.x, atms[i_atm].x, atms[i_atm + 1].x);
        if (tid == 5) {
            std::printf("%f %f %f\n", ray.x, ray.y, ray.z);
        }
	}
    // write data to global memory
    rays[tid] = ray;
}

////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////
extern "C" bool
runAcousticTest(const int argc, const char **argv, 
	float3 *h_rays, unsigned int n_rays, float3 *h_atm_layers, unsigned int n_atm_layers)	
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, (const char **)argv);

	const unsigned int num_threads = n_rays;
	const unsigned int rays_mem_size = sizeof(float3) * n_rays;
	const unsigned int atms_mem_size = sizeof(float3) * n_atm_layers;

	// allocate device memory
	float3 *d_rays;
	checkCudaErrors(cudaMalloc((void **)&d_rays, rays_mem_size));
	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_rays, h_rays, rays_mem_size,
		cudaMemcpyHostToDevice));
	// allocate device memory for int2 version
	float3 *d_atm_layers;
	checkCudaErrors(cudaMalloc((void **)&d_atm_layers, atms_mem_size));
	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_atm_layers, h_atm_layers, atms_mem_size,
		cudaMemcpyHostToDevice));

	// setup execution parameters
	dim3 grid(1, 1, 1);
	dim3 threads(num_threads, 1, 1);
							  
    // execute the kernel
    acoustic_trace_kernel <<< grid, threads >>>(d_rays, d_atm_layers, n_atm_layers);
	
	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy results from device to host
	checkCudaErrors(cudaMemcpy(h_rays, d_rays, rays_mem_size,
		cudaMemcpyDeviceToHost));

	// cleanup memory
	checkCudaErrors(cudaFree(d_rays));
	checkCudaErrors(cudaFree(d_atm_layers));
	
	return true;
}
