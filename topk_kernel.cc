#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

using namespace tensorflow;


REGISTER_OP("MyTopK")
	.Attr("T: {float, double, int32}")
	.Attr("k: int")
	.Attr("sorted: bool = false")
	.Input("input: T")
	.Output("values: T")
	.Output("indices: int32");

//function lounching GPU kernel
//N number of rows, last_size number of elem in k dimension
template<typename T>
void run(bool sorted, const int N, const int k, const int last_size, const T* in, int32* indices, T* values);

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename T>
class MyTopKOp : public OpKernel{
public:
	explicit MyTopKOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
        	OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
	}

	void Compute(OpKernelContext* context) override {
		int k = k_;
		const auto& k_in = context->input(1);

		OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
		const auto& input = context->input(0);
    		OP_REQUIRES(context, input.dims() >= 1,
                		errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input.shape().DebugString()));
    		OP_REQUIRES(context, input.dim_size(input.dims() - 1) >= k,
				errors::InvalidArgument("input must have at least k columns"));
        
        	if(k == 0) {return; }

        	TensorShape output_shape = input.shape();
    		output_shape.set_dim(input.dims() - 1, k);
    		Tensor* values_out = nullptr;
    		OP_REQUIRES_OK(context,
                   	context->allocate_output(0, output_shape, &values_out));
    		Tensor* indices_out = nullptr;
    		OP_REQUIRES_OK(context,
			context->allocate_output(1, output_shape, &indices_out));
              /*------------------------------------------------------------------------------------------- */
	        const int input_dims_number = input.dims();
		const int dim_last_size = input.dim_size(input_dims_number - 1);
        	const int N = input.flat<T>().size() / dim_last_size;
        	const T *input_data = input.flat<T>().data();
        	T *values_data = values_out->flat<T>().data();
        	int32 *indices_data = indices_out->flat<int32>().data();
              /*----------------------------------------------------------------------------------------------- */
    		run<T>(sorted_, N, k, dim_last_size, input_data, 
			indices_data, values_data);	
	}
private:
    bool sorted_;
    int k_;
};
// Register the GPU kernels.

                                       

    
REGISTER_KERNEL_BUILDER(                                       
	Name("MyTopK")
	.Device(DEVICE_GPU)
	.TypeConstraint<float>("T"), 
	MyTopKOp<float>);

REGISTER_KERNEL_BUILDER(
        Name("MyTopK")
        .Device(DEVICE_GPU)
        .TypeConstraint<double>("T"),
        MyTopKOp<double>);

REGISTER_KERNEL_BUILDER(
        Name("MyTopK")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T"),
        MyTopKOp<int32>);





