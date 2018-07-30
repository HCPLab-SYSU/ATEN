OUT_DIR   = ./build

TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_NSYNC = $(TF_INC)/external/nsync/public
TF_SHAD = $(TF_INC)/tensorflow/core/util
CUDA_HOME = /usr/local/cuda

GPUFLAGS = -I $(TF_INC) -I$(TF_NSYNC) -I$(CUDA_HOME)/include -I/usr/local -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
CFLAGS = -I $(TF_INC) -I$(TF_NSYNC) -I$(TF_SHAD) -fPIC -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -L$(TF_LIB) -ltensorflow_framework

all: flow_warp.so

flow_warp_gpu.o:
	nvcc -std=c++11 -c --expt-relaxed-constexpr --gpu-architecture=sm_52 \
		-o $(OUT_DIR)/flow_warp_gpu.o flow_warp/flow_warp.cu.cc \
		$(GPUFLAGS)
flow_warp_grad_gpu.o:
	nvcc -std=c++11 -c --expt-relaxed-constexpr --gpu-architecture=sm_52 \
		-o $(OUT_DIR)/flow_warp_grad_gpu.o flow_warp/flow_warp_grad.cu.cc \
		$(GPUFLAGS)
flow_warp.so: flow_warp_gpu.o flow_warp_grad_gpu.o
	g++ -std=c++11 -shared \
		-o $(OUT_DIR)/flow_warp.so \
		flow_warp/flow_warp_op.cc flow_warp/flow_warp.cc flow_warp/flow_warp_grad.cc \
		$(OUT_DIR)/flow_warp_gpu.o $(OUT_DIR)/flow_warp_grad_gpu.o \
		$(CFLAGS)

clean:
	rm -f $(OUT_DIR)/*




