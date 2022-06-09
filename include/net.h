#ifndef NET_H
#define NET_H

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <functional>
#include "cuda_runtime.h"
#include <nvToolsExt.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <pthread.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <signal.h>
#include <memory>
#include <stdlib.h>
#include <cstdio>
#include <c10/cuda/CUDAFunctions.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sched.h>

struct Dummy : torch::jit::Module{};

typedef struct _gpu
{
	int g_index;
	std::vector<at::cuda::CUDAStream> g_stream;
	torch::Device g_device;
	int n_net;
	int all_api;	//필요?
	int load;
	int last_cnt;
	int total_weight;	//필요?
	int worker_load;
	_gpu() : g_device({c10::DeviceType::CUDA, 0}){

	}
}Gpu;

typedef struct _layer
{
	at::Tensor output;
	std::string name;	//layer name
	torch::jit::Module layer;
	bool exe_success;	//layer operation complete or not
	bool check_concat;
	bool round_last; // last layer check for round (CONST_P)
	std::vector<int> from_idx;	//concat
	std::vector<int> branch_idx;	// last layer idx of branch for eventrecord
	int input_idx; 	//network with branch
	int event_idx;	//network with branch
	int skip;	//inception skip num in a branch
	int stream_idx;	//stream index of current layers
	int l_kernel;	//num of kernel
	int l_cuevent;	//num of cuda event
	int l_cubind;	//num of cudabind/unbind
	int l_api;
	int l_next; //next layer index
	int l_prev; //prev layer index
	int all_api;
	int l_load;
	int l_identity; //for resnet
	// float q_mean;
	float l_mean;	//mean of i-th layertime
	float l_mem;	//memory (kb)
	/*for record time only in Q*/
	// bool dequeue;
	struct timespec q_start,q_end;
	double q_time;
	float l_time;
}Layer;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	at::Tensor identity;	//resnet
	std::vector<cudaEvent_t> record;
	std::vector<at::Tensor> chunk; //shuffle
	std::string name;	//network name
	bool change_gid;	//check change gpu index
	bool warming;
	bool loadbalancing;
	int nice; //nice value per net
	int weight;
	int index; //layer index
	int index_n; //network index
	int index_s; // stream index
	int index_b; // index of stream for branch layer 
	int n_all; // all network num
	int flatten; //flatten layer index
	int last; 	//last layer index
	int cur_round_last;	//current round last layer index
	int next_round_last;	//next round last layer index
	Gpu *device;
	int all_api;
	int q_all_api;
	int g_index;
	float all_mem;
	//c10::DeviceIndex device;	//device index
	float timeslice;
	float q_mean;
	FILE *fp; // for recording
	// int all_kernels;
	// int all_cuevent;
	// int all_cubind;
}Net;


// typedef struct _netlayer
// {
// 	Net *net;
// }netlayer;

#endif
