#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

template <typename T>
__device__ bool greater(const T& a, const  T& b, const int32& index_a, const int32& index_b) {
                if (a  == b ) {
                        return index_a < index_b;
                }
                return a > b;
	};
template <typename T>
__device__ void swap(T& a, T& b){
                 T temp = a;
                 a = b;
                 b = temp;
          };
//GPU heap definition
template  <typename T>
struct Heap{
	__device__ void init(const int max_size, T* values_memory, int32* indices_memory){
		this->max_size = max_size;
		this->values = values_memory;
		this->indices = indices_memory;
		n = 0;
	}

	__device__ void add(T element, int32 index_set){
		if(n < max_size){	
			values[n] =  element;
			indices[n] = index_set;
			build_heap(n);
			n++; 
		} else{
			T temp_value = element;
			int32 temp_indice = index_set;
			if(greater(temp_value,  get_min(), temp_indice, get_min_indice())){
				values[0] = temp_value;
				indices[0] = temp_indice;
				rebuild_heap(0);	
			}
		}	
	}
	__device__ T get_min() { return values[0]; }
	__device__ int32 get_min_indice() {return indices[0]; }
	__device__ void clear() { n = 0; }
	__device__ void build_heap(const int start_index){
			int index = start_index;
			while(index > 0 && greater(values[(index - 1) / 2], values[index], indices[(index - 1) / 2 ], indices[index])){
				swap(values[index], values[(index - 1) / 2]);
				swap(indices[index], indices[(index - 1) /2]);
				index = (index - 1) / 2;
			}
		}
	__device__ void rebuild_heap(const int start){
			 int index = start;
			 int left = index * 2 + 1;
                         int right = left + 1;
			 if(right >= n) right = left;

                         while(right < n ){
                 		if(greater(values[left],  values[index], indices[left], indices[index]) && 
                                  	greater(values[right], values[index], indices[right], indices[index])) { break; }

                     		if(greater(values[right],  values[left], indices[right], indices[left])){
                                	swap(values[index], values[left]);
                                       	swap(indices[index], indices[left]);
                                     	index = left;
                             	}else{
                                      	swap(values[index], values[right]);
                                       	swap(indices[index], indices[right]);
                                        index = right;
                                }
                                        left = index * 2 + 1;
                                        right = left + 1;
                      }          

		}	 
	T* values;
	int32* indices;
        int n;
        int max_size;   
};

// run heap with top k elem on GPU
template <typename T>
__device__ void run_heap(bool sorted, const int k, const int index, const int last_size, const T* in, int32* indices, T* values){
	Heap<T> heap;
	heap.init(k, values, indices);	
	for(int i = 0; i < last_size; ++i){ heap.add(in[i], i);	}
//make heap sort
	if(sorted){	
		while(heap.n > 0){
			swap(heap.values[0], heap.values[heap.n-1]);
			swap(heap.indices[0], heap.indices[heap.n-1]);
			heap.n--;
			heap.rebuild_heap(0);
		}
	}
}

// Define the CUDA kernel.
template <typename T>
__global__ void CudaKernel(bool sorted, const int N, const int k, const int last_size, const T* in, int32* indices, T* values) {
	for(int i  = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
		run_heap(sorted, k, i, last_size, &in[i*last_size], &indices[i * k], &values[i * k]);	
	} 
}

// Define the GPU implementation that launches the CUDA kernel.
template<typename T>
void run(bool sorted, const int N, const int k, const int last_size, const T* in, int32* indices, T* values){
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
	int block_count = 65535;
	int thread_per_block = 1024;
 	CudaKernel<T>
     		 <<<block_count, thread_per_block>>>(sorted, N, k, last_size, in, indices, values);
}

template void run<int32>(bool sorted, const int N, const int k, const int last_size, const int32* in, int32* indices, int32* values);
template void run<double>(bool sorted, const int N, const int k, const int last_size, const double* in, int32* indices, double* values);
template void run<float>(bool sorted, const int N, const int k, const int last_size, const float* in, int32* indices, float* values);

#endif  // GOOGLE_CUDA

