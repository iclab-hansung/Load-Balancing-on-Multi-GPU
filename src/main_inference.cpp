#include "test.h"
#include <cuda_profiler_api.h>

// #define n_threads 1
#define WARMING 4 //4
// #define N_GPU 3

extern void *predict_densenet_warming(Net *dense);
extern void *predict_resnet_warming(Net *res);

extern void *predict_densenet(std::vector<Net*> *vec_dense);
extern void *predict_resnet(std::vector<Net*> *vec_res);


extern void *check_loadbalancing();

namespace F = torch::nn::functional;
using namespace std;

// FILE *fp_before = fopen("../0510-R30-iter1000-thresavg-sleep15-wbalancing.txt","w");
// FILE *fp_after = fopen("../0509-D27-iter1000-thresavg-sleep15-after.txt","w");


threadpool thpool[N_GPU];

pthread_mutex_t* mutex_g;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;

//Load check mutex
pthread_mutex_t mutex_l;
pthread_cond_t cond_l;
bool check_load;

std::vector<std::vector<at::cuda::CUDAStream>> streams;

c10::DeviceIndex GPU_NUM=0;

#if CPU_PINNING
  cpu_set_t cpuset;
  int n_cpu = (27 - n_threads);
  vector<int> cpu_list = {21,22,23,24};
#endif

float CONST_P = 15; //ms 


vector<vector<float>> input_layer_values(const std::string& filePath){
  std::ifstream fs(filePath);

  if (true == fs.fail())
  {
      throw std::ifstream::failure("fail to open file");
  }
  vector<vector<float>> indexkernel;
  vector<float> data;

  while(!fs.eof()){

    string str_buf;
    float tmp;
    getline(fs,str_buf);
    size_t prev=0;
    size_t current;
    string substring;
    current = str_buf.find(',');

    while(current != string::npos){
      substring=str_buf.substr(prev,current-prev);
      prev = current + 1;
      current = str_buf.find(',',prev);
      tmp = std::stof(substring);
      data.push_back(tmp); 
    }
    substring=str_buf.substr(prev,current-prev);  //last
    tmp = std::stof(substring);
    data.push_back(tmp);
    indexkernel.push_back(data);
    data.clear();
  }
  fs.close();
  return indexkernel;
}


int nice2prio(int nice){
  int prio = nice+20;
  return prio;
}
float cal_timeslice(int total_weight,int my_weight){
  float timeslice = (float)(((float)my_weight/(float)total_weight)*CONST_P);
  return timeslice;
}


#if RECORD || Q_OVERHEAD || MEM_RECORD
string result_path;
#endif

vector<vector<torch::jit::IValue>> inputs(N_GPU);
vector<vector<torch::jit::IValue>> inputs2(N_GPU);
vector<vector<torch::jit::IValue>> inputs3(N_GPU);

vector<Gpu> gpu_list;
vector<int> gpu_idx={0,1,2,3};
// vector<int> gpu_num={0,3,1,2,0,2,1,2,2,1,3,1,3,0,1,3,0,3,0,3,0,1,2,0,2,1,3,1,3,1};
vector<int> gpu_num_dense={0, 0, 1, 2, 1, 2, 3, 1, 3, 2, 2, 3, 3, 0, 0, 0, 2, 3, 1, 1, 0, 0, 2, 2, 0, 0, 3, 1, 2, 2, 1, 3};
vector<int> gpu_num_res={3, 1, 0, 1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 0, 1, 2, 2, 2, 1, 1, 2, 0, 3, 2, 1, 3, 3, 2, 0, 3};

int gpu_n = gpu_idx.size();

int THRESHOLD = 10000; //초기값
time_t ti;
bool nice_0_end = false;

int main(int argc, const char* argv[]) {
  ti= time(NULL);
  GPU_NUM=atoi(argv[1]);
  c10::cuda::set_device(GPU_NUM);
  torch::Device device = {at::kCUDA,GPU_NUM};

  #if RECORD || Q_OVERHEAD || MEM_RECORD
    result_path = argv[2];
  #endif

  int n_dense=atoi(argv[3]);
  int n_res=atoi(argv[4]);
  int n_alex=atoi(argv[5]);
  int n_vgg=atoi(argv[6]);
  int n_wide=atoi(argv[7]);
  int n_squeeze=atoi(argv[8]);
  int n_mobile=atoi(argv[9]);
  int n_mnasnet=atoi(argv[10]);
  int n_inception=atoi(argv[11]);
  int n_shuffle=atoi(argv[12]);
  int n_resX=atoi(argv[13]);
  int n_efficient=atoi(argv[14]);
  
  #if RECORD || Q_OVERHEAD || MEM_RECORD
    std::string filename;
    if(n_dense) filename += "D"+to_string(n_dense);
    if(n_res) filename += "R"+to_string(n_res);
    if(n_alex) filename += "A"+to_string(n_alex);
    if(n_vgg) filename += "V"+to_string(n_vgg);
    if(n_wide) filename += "W"+to_string(n_wide);
    if(n_mobile) filename += "M"+to_string(n_mobile);
    if(n_mnasnet) filename += "N"+to_string(n_mnasnet);
    if(n_inception) filename += "I"+to_string(n_inception);
    if(n_resX) filename += "X"+to_string(n_resX);
    if(n_efficient) filename += "E"+to_string(n_efficient);
    //X5 - X1 : resX 5개 중 index 1 record
  #endif

  srand(time(NULL));

  int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX + n_efficient;

  static int stream_index_H = 0;
  static int branch_index_H = 31;

  for(int g=0;g<N_GPU;g++){
    thpool[g] = thpool_init(n_threads,g);
  }

  vector<int> nice_list = {0,5,10};

  /*struct init*/
  streams.resize(N_GPU); // streams[][] 형식으로 사용할 것

  for(int i=0; i<gpu_idx.size(); i++){
    for(int j=0; j<n_streamPerPool;j++){
      streams[gpu_idx[i]].push_back(at::cuda::getStreamFromPool(true,gpu_idx[i]));//(c10::DeviceIndex)i));
    }
  }
  
  gpu_list.resize(N_GPU);
  
  for(int i=0;i<gpu_idx.size();i++){ //struct 사이즈로 malloc 잡으면 C++ 에서 bad alloc 발생
    std::cout<<"***** GPU "<<gpu_idx[i]<<" INIT *****\n";
    Gpu g;
    g.g_index = gpu_idx[i];
    g.all_api = 0;
    g.n_net = 0;
    g.last_cnt = 0; //필요?
    g.total_weight = 0;
    g.load = 0;
    g.g_device = {at::kCUDA,(c10::DeviceIndex)g.g_index};
    g.g_stream = streams[gpu_idx[i]];
    gpu_list[gpu_idx[i]] = g;
  } 

  torch::jit::script::Module denseModule[N_GPU];
  torch::jit::script::Module resModule[N_GPU];

  try {
    for(int i=0; i<gpu_n; i++){
      if(n_dense){
        denseModule[gpu_idx[i]] = torch::jit::load("../model/densenet201_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
      if(n_res){
        resModule[gpu_idx[i]] = torch::jit::load("../model/resnet152_model.pt",gpu_list[gpu_idx[i]].g_device);
      }
    }
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";

  //Network Mutex
  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);

  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }

  //Load check Mutex
  pthread_cond_init(&cond_l, NULL);
  pthread_mutex_init(&mutex_l, NULL);
  check_load = false;  
  

//GPU Mutex
  mutex_g = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) *N_GPU);

  for(int i=0;i<N_GPU;i++)
  {
    pthread_mutex_init(&mutex_g[i], NULL);
  }

  for(int i=0;i<gpu_n;i++){
    torch::Tensor x = torch::ones({1,3,224,224}).to(gpu_list[gpu_idx[i]].g_device); 
    inputs[gpu_list[gpu_idx[i]].g_index].push_back(x);
    
    if(n_inception){
      torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(gpu_list[gpu_idx[i]].g_device);

      auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
      auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
      auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
        
      x_ch0.to(gpu_list[gpu_idx[i]].g_device);
      x_ch1.to(gpu_list[gpu_idx[i]].g_device);
      x_ch2.to(gpu_list[gpu_idx[i]].g_device);

      auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(gpu_list[gpu_idx[i]].g_device);
      inputs2[gpu_list[gpu_idx[i]].g_index].push_back(x_cat);
    }

    if(n_efficient){
      torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(gpu_list[gpu_idx[i]].g_device);
      inputs3[gpu_list[gpu_idx[i]].g_index].push_back(x3);
    }
  }

  at::Tensor out;
  int total_w = 0;

  vector<Net*> multig_dense;
  vector< vector<Net*> > net_input_dense;
  Net dense[n_dense][N_GPU];
  pthread_t networkArray_dense[n_dense];
  multig_dense.resize(N_GPU);
  net_input_dense.resize(n_dense);


  vector<Net*> multig_res;
  vector< vector<Net*> > net_input_res;
  Net res[n_res][N_GPU];
  pthread_t networkArray_res[n_res];
  multig_res.resize(N_GPU);
  net_input_res.resize(n_res);

  for(int i=0;i<n_dense;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_densenet(denseModule[g], dense[i][g]);
          dense[i][g].flatten = dense[i][g].layers.size()-1;
          dense[i][g].device = &gpu_list[g];
          // dense[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : dense[i][gpu_idx[0]].g_index;
          dense[i][g].name = "DenseNet";
          dense[i][g].index_n = i;
          if(gpu_idx.size()>1){
            dense[i][g].g_index = gpu_num_dense[dense[i][g].index_n];
          }else{
            dense[i][g].g_index = gpu_idx[0];
          }
          dense[i][g].index_s = stream_index_H;
          dense[i][g].nice = nice_list[dense[i][g].index_n%nice_list.size()];
          dense[i][g].weight = prio_to_weight[nice2prio(dense[i][g].nice)];
          dense[i][g].timeslice = CONST_P;
          dense[i][g].last = 801;
          dense[i][g].input = inputs[g];
          dense[i][g].change_gid = false;
          dense[i][g].loadbalancing = false;
          // dense[i][g].device->n_net += 1;
          dense[i][g].warming = false;
          dense[i][g].all_api = 0;
          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            dense[i][g].fp = fopen((result_path+"dense/"+filename+"-"+"D"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==1){
            indexkernel = input_layer_values("../layer_values/new_onlydense_th1.csv");
          }else if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlydense_th3.csv");  
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlydense_th6.csv");  
          }
          int l_prev = 0;
          for(int l=0;l<dense[i][g].layers.size();l++){
            dense[i][g].layers[l].l_api = 0;
            dense[i][g].layers[l].l_mean = 0.0;
            dense[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                dense[i][g].layers[l].l_mean = indexkernel[w][1];
                dense[i][g].layers[l].l_api = (int)indexkernel[w][2];
                dense[i][g].layers[l].l_mem = indexkernel[w][3];
                dense[i][g].layers[l].l_prev = l_prev;
                dense[i][g].layers[l_prev].l_next = l;
                if(l == dense[i][g].last){
                  dense[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_densenet_warming(&dense[i][g]);
          }
          dense[i][g].input = inputs[dense[i][g].g_index];
          dense[i][g].warming = true;
          multig_dense[g]=(&(dense[i][g])); //하나의 net_index 0,1,2,3 
        }
      }
    }
    total_w += dense[i][0].weight;
    stream_index_H+=1;
    net_input_dense[i]=multig_dense;
  }

  for(int i=0;i<n_res;i++){
    for(int g=0;g<N_GPU;g++){
      for(int j=0;j<gpu_n;j++){
        if(g==gpu_idx[j]){
          get_submodule_resnet(resModule[g], res[i][g]);
          res[i][g].flatten = res[i][g].layers.size()-1;
          res[i][g].device = &gpu_list[g];
          // res[i][g].g_index = g==gpu_idx[0] ? gpu_idx[rand()%gpu_n] : res[i][gpu_idx[0]].g_index;
          res[i][g].name = "ResNet";
          res[i][g].index_n = i + n_dense;
          if(gpu_idx.size()>1){
            res[i][g].g_index = gpu_num_res[res[i][g].index_n];
          }else{
            res[i][g].g_index = gpu_idx[0];
          }
          res[i][g].index_s = stream_index_H;
          res[i][g].nice = nice_list[res[i][g].index_n%nice_list.size()];
          res[i][g].weight = prio_to_weight[nice2prio(res[i][g].nice)];
          res[i][g].timeslice = CONST_P;
          res[i][g].last = 308;
          res[i][g].input = inputs[g];
          res[i][g].change_gid = false;
          res[i][g].loadbalancing = false;
          // res[i][g].device->n_net += 1;
          res[i][g].warming = false;
          res[i][g].all_api = 0;
          #if RECORD || Q_OVERHEAD || MEM_RECORD
            /*=============FILE===============*/
            res[i][g].fp = fopen((result_path+"res/"+filename+"-"+"R"+to_string(i)+".txt").c_str(),"a"); 
          #endif

          vector<vector<float>> indexkernel;
          if(n_threads==1){
            indexkernel = input_layer_values("../layer_values/new_onlyres_th1.csv");
          }else if(n_threads==3){
            indexkernel = input_layer_values("../layer_values/new_onlyres_th3.csv"); //각 레이어의 커널 개수 
          }else{
            indexkernel = input_layer_values("../layer_values/new_onlyres_th6.csv"); //각 레이어의 커널 개수 
          }
          int l_prev = 0;
          for(int l=0;l<res[i][g].layers.size();l++){
            res[i][g].layers[l].l_api = 0;
            res[i][g].layers[l].l_mean = 0.0;
            res[i][g].layers[l].l_mem = 0.0;

            for(int w=0;w<indexkernel.size();w++){
              if(l==(int)indexkernel[w][0]){
                res[i][g].layers[l].l_mean = indexkernel[w][1];
                res[i][g].layers[l].l_api = (int)indexkernel[w][2];
                res[i][g].layers[l].l_mem = indexkernel[w][3];
                res[i][g].layers[l].l_identity = indexkernel[w][4];
                res[i][g].layers[l].l_prev = l_prev;
                res[i][g].layers[l_prev].l_next = l;
                if(l == res[i][g].last){
                  res[i][g].layers[l].l_next = l; //for last
                }
                l_prev = l;
                break;
              }
            }
          }
          for(int j=0;j<WARMING;j++){
            predict_resnet_warming(&res[i][g]);
            
          }
          res[i][g].input = inputs[res[i][g].g_index];
          res[i][g].warming = true;
          multig_res[g]=(&(res[i][g])); //하나의 net_index 0,1,2,3 
        }
      }
    }
    total_w += res[i][0].weight;
    stream_index_H+=1;
    net_input_res[i]=multig_res;
  }



  std::cout<<"\n==================WARM UP END==================\n";

  THRESHOLD = int(total_w / N_GPU);
  // std::cout<<"THRESHOLD : "<<THRESHOLD<<"total : "<<total_w<<"N_GPU"<<N_GPU<<std::endl;
  
  for(int g=0;g<gpu_n;g++){
    gpu_list[gpu_idx[g]].load = 0;
    gpu_list[gpu_idx[g]].all_api = 0;
  }

  //cudaDeviceSynchronize();
  cudaProfilerStart();

  cudaEvent_t t_start, t_end;
  float t_time;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);

  cudaEventRecord(t_start);


  for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &(net_input_dense[i])) < 0){
      perror("thread error");
      exit(0);
    }//std::cout<<"dense device num = "<<c10::cuda::current_device()<<"\n";
    // std::cout<<"thread "<<i<<std::endl;
    // #if CPU_PINNING

    //   #if CORE4
    //   // std::cout<<cpu_list[net_input_dense[i][0]->index_n%(cpu_list.size())]<<std::endl;
    //   CPU_SET(cpu_list[net_input_dense[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif

		// 	pthread_setaffinity_np(networkArray_dense[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  } 

  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &(net_input_res[i])) < 0){
      perror("thread error");
      exit(0);
    }
    // #if CPU_PINNING
    //   #if CORE4
    //   CPU_SET(cpu_list[net_input_res[i][0]->index_n%(cpu_list.size())], &cpuset);
    //   #else
    //   CPU_SET(n_cpu, &cpuset); //only this thread has the affinity for the 'n'-th CPU	
    //   #endif
		// 	pthread_setaffinity_np(networkArray_res[i], sizeof(cpu_set_t), &cpuset);
    //   n_cpu--;
    // #endif
  }

  /*Load balancing thread*/

  pthread_t p_sched;
  if (pthread_create(&p_sched,NULL,(void *(*)(void*))check_loadbalancing,NULL)<0){
      perror("thread error");
      exit(0);
  }

  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL);
  }
  for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&t_time, t_start, t_end);

	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
  cudaProfilerStop();

  fclose(fp_before);

}
