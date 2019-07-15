#include <cuda.h>


void cuda_info();
void cuda_error_info(const char* msg, cudaError_t error);

int shared_memory_per_block(int device_id);
