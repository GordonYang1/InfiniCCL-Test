#ifndef RUNTIME_API_H
#define RUNTIME_API_H

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#define GPU_MALLOC(ptr_addr, sz) cudaMalloc((void **)(ptr_addr), (sz))
#define GPU_FREE(ptr) cudaFree(ptr)
#define GPU_MEMCPY_H2D(d, s, n) cudaMemcpy(d, s, n, cudaMemcpyHostToDevice)
#define GPU_MEMCPY_D2H(d, s, n) cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost)
#define GPU_SYNC() cudaDeviceSynchronize()
#define GPU_SET_DEVICE(id) cudaSetDevice(id)
#define GPU_GET_DEVICE_PROPS(p, id) cudaGetDeviceProperties(p, id)
#define GPU_MALLOC_HOST(ptr, sz) cudaMallocHost((void **)&(ptr), (sz))
#define GPU_FREE_HOST(ptr) cudaFreeHost(ptr)
typedef cudaDeviceProp gpuProp_t;
#define GPU_PLATFORM "NVIDIA"
#define CHECK_DEVICE(call)                                                     \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#elif defined(USE_MACA)
#include <mcr/mc_runtime_api.h>
#define GPU_MALLOC(ptr_addr, sz) mcMalloc((void **)(ptr_addr), (sz))
#define GPU_FREE(ptr) mcFree(ptr)
#define GPU_MEMCPY_H2D(d, s, n) mcMemcpy(d, s, n, mcMemcpyHostToDevice)
#define GPU_MEMCPY_D2H(d, s, n) mcMemcpy(d, s, n, mcMemcpyDeviceToHost)
#define GPU_SYNC() mcDeviceSynchronize()
#define GPU_SET_DEVICE(id) mcSetDevice(id)
#define GPU_GET_DEVICE_PROPS(p, id) mcGetDeviceProperties(p, id)
#define GPU_MALLOC_HOST(ptr, sz) malloc(sz)
#define GPU_FREE_HOST(ptr) free(ptr)
typedef mcDeviceProp_t gpuProp_t;
#define GPU_PLATFORM "MetaX"
#define CHECK_DEVICE(call)                                                     \
  do {                                                                         \
    mcError_t err = (call);                                                    \
    if (err != mcSuccess) {                                                    \
      std::cerr << "MetaX error in " << __FILE__ << ":" << __LINE__ << ": "    \
                << mcGetErrorString(err) << std::endl;                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#endif // RUNTIME_API_H
