#ifndef TEST_H
#define TEST_H

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <functional>
#include "cuda_runtime.h"
#include <nvToolsExt.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <unistd.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <pthread.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <utility>
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
#include <algorithm>

#include "net.h"
#include "thpool.h"
#include "alex.h"
#include "vgg.h"
#include "resnet.h"
#include "densenet.h"
#include "inception.h"
#include "efficient.h"

#define n_threads 3
#define N_GPU 4
#define ITERATION 1000
#define HOMO 1

#define n_streamPerPool 32
#define CPU_PINNING 0
#define CORE4 0
#define RT_THREAD 0
#define RECORD 1
#define L_RECORD 1
#define Q_OVERHEAD 1
#define MEM_CPY 0
#define MEM_RECORD 0
#define L_SYNC 0
#define NVTX 0

#define NANO 1000000000
#define MILLI 1000000

const float CPY_a=0.07787;
const float CPY_b=1.10798;

extern threadpool thpool[N_GPU];
extern pthread_mutex_t *mutex_g;
extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern pthread_cond_t cond_l;
extern pthread_mutex_t mutex_l;
extern int *cond_i;
extern bool check_load; 
extern std::vector<std::vector<at::cuda::CUDAStream>> streams;
extern c10::DeviceIndex GPU_NUM;
extern std::vector<int> gpu_idx;
extern std::vector<int> available_gpu;
extern int gpu_n;
extern int THRESHOLD;
extern bool nice_0_end;

extern pthread_cond_t *cond_p;
extern int *cond_p_i;
extern std::vector<std::vector<torch::jit::IValue>> inputs;
extern std::vector<std::vector<torch::jit::IValue>> inputs2;
extern std::vector<std::vector<torch::jit::IValue>> inputs3;
extern std::vector<Gpu> gpu_list;
extern int all_api_0,all_api_1,all_api_2,all_api_3;
extern FILE *fp_before,*fp_after;
extern time_t ti;
extern cpu_set_t cpuset;

extern float CONST_P;
int nice2prio(int nice);
float cal_timeslice(int total_weight,int my_weight);

static const int prio_to_weight[40] = {  
 /* -20 */     88761,     71755,     56483,     46273,     36291,  
 /* -15 */     29154,     23254,     18705,     14949,     11916,  
 /* -10 */      9548,      7620,      6100,      4904,      3906,  
 /*  -5 */      3121,      2501,      1991,      1586,      1277,  
 /*   0 */      1024,       820,       655,       526,       423,  
 /*   5 */       335,       272,       215,       172,       137,  
 /*  10 */       110,        87,        70,        56,        45,  
 /*  15 */        36,        29,        23,        18,        15,  
};


#endif
