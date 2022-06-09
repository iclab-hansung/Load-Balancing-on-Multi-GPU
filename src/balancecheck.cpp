#include "balancecheck.h"

float find_time_value(Net *net,int api_num){
  float Q_time = 0.0;
  float Q_mean = 0.0;
  /*Q time*/
  #if HOMO
    if(n_threads == 1){
      if(net->name == "DenseNet"){
        Q_mean = 0.0136;
      }
      else if(net->name == "ResNet"){
        Q_mean = 0.0102;
      }
      else if(net->name == "AlexNet"){
        Q_mean = 0.0166;
      }
      else{
        Q_mean = 0.0047;
      }
    }
    else {//if(n_threads == 3){      
      if(net->name == "DenseNet"){
        if(gpu_list[net->g_index].all_api == 0){
          Q_time = 3.427001;
        }else{
          Q_mean = 0.007132;
          Q_time = Q_mean * api_num;
        }        
      }
      else if(net->name == "ResNet"){
        if(gpu_list[net->g_index].all_api == 0){
          Q_time = 1.497702;
        }else{
          Q_mean = 0.006254;
          Q_time = Q_mean * api_num;
        }
        // std::cout<<net->index_n<<" api num : "<<api_num<<" Q mean : "<<Q_mean<<"Q time :"<<Q_time<<std::endl;
      }
      else if(net->name == "EfficientNet"){
        if(gpu_list[net->g_index].all_api == 0){
          Q_time = 3.482466;
        }else{
          Q_mean = 0.007416;
          Q_time = Q_mean * api_num;
        }
      }
      else{
        if(gpu_list[net->g_index].all_api == 0){
          Q_time = 5.534114;
        }else{
          Q_mean = 0.003263;
          Q_time = Q_mean * api_num;
        }
      }
    }
    // else{ //n_threads==6      
    //   if(net->name == "DenseNet"){
    //     if(gpu_list[net->g_index].all_api == 0){
    //       Q_time = 0.315743;
    //     }else{
    //       Q_mean = 0.00600;
    //       Q_time = Q_mean * api_num;
    //     }        
    //   }
    //   else if(net->name == "ResNet"){
    //     if(gpu_list[net->g_index].all_api == 0){
    //       Q_time = 2.180298;
    //     }else{
    //       Q_mean = 0.00746;
    //       Q_time = Q_mean * api_num;
    //     }
    //     // std::cout<<net->index_n<<" api num : "<<api_num<<" Q mean : "<<Q_mean<<"Q time :"<<Q_time<<std::endl;
    //   }
    //   else if(net->name == "EfficientNet"){
    //     if(gpu_list[net->g_index].all_api == 0){
    //       Q_time = 0.87961;
    //     }else{
    //       Q_mean = 0.00669;
    //       Q_time = Q_mean * api_num;
    //     }
    //   }
    //   else{
    //     if(gpu_list[net->g_index].all_api == 0){
    //       Q_time = 2.65369;
    //     }else{
    //       Q_mean = 0.00359;
    //       Q_time = Q_mean * api_num;
    //     }
    //   }
    // }
  #else
  //if(n_threads==3){
    if(gpu_list[net->g_index].all_api==0){
      Q_time = 0.075183;
    }else{
      if(net->name == "DenseNet"){
        Q_mean = 0.003428;
        Q_time = Q_mean * api_num;
      }
      else if(net->name == "ResNet"){
        Q_mean = 0.004337;
        Q_time = Q_mean * api_num;
      }
      else if(net->name == "EfficientNet"){
        Q_mean = 0.003537;
        Q_time = Q_mean * api_num;
      }
      else{ //inception
        Q_mean = 0.003795;
        Q_time = Q_mean * api_num;
      }
    }
  //}
  // }else{  //n_threads==6
  //   if(gpu_list[net->g_index].all_api==0){
  //     Q_time = 0.687931;
  //   }else{
  //     if(net->name == "DenseNet"){
  //       Q_mean = 0.005105;
  //       Q_time = Q_mean * api_num;
  //     }
  //     else if(net->name == "ResNet"){
  //       Q_mean = 0.005096;
  //       Q_time = Q_mean * api_num;
  //     }
  //     else if(net->name == "EfficientNet"){
  //       Q_mean = 0.005099;
  //       Q_time = Q_mean * api_num;
  //     }
  //     else{ //inception
  //       Q_mean = 0.005201;
  //       Q_time = Q_mean * api_num;
  //     }
  //   }
  // }

  #endif
  // Q_time = Q_mean * api_num;
  return Q_time;
}

int get_lowest_load_idx(Net *net){
  std::vector< std::pair<int,int> >g_load;

  for(int g=0;g<gpu_idx.size();g++){
    pthread_mutex_lock(&mutex_g[gpu_idx[g]]);
    g_load.push_back(std::make_pair(gpu_list[gpu_idx[g]].g_index,gpu_list[gpu_idx[g]].load));
    pthread_mutex_unlock(&mutex_g[gpu_idx[g]]);
  }
  
  auto L_min = *min_element(g_load.begin(),g_load.end(),[](const auto& lhs, const auto& rhs) {
    return lhs.second < rhs.second;    
  });
  auto L_idx = L_min.first;

  // fprintf((net->fp),"%d,%d,%d,%d,%d,",net->index_n,g_load[0].second,g_load[1].second,g_load[2].second,g_load[3].second);

  return (int)L_idx;
}

void select_queue(Net *net,int my_gid,int L_gid,float mem_size){
  int my_api = gpu_list[my_gid].all_api;
  int L_api = gpu_list[L_gid].all_api;
  float cpy_time = 0.0;

  if(mem_size == 0){
    cpy_time = 0;
  }else{
    cpy_time = ((CPY_a * mem_size) + CPY_b)/1000; 
  }

  float my_Q = find_time_value(net,my_api);
  float L_Q = find_time_value(net,L_api)+cpy_time;

  
  if(my_Q>L_Q){
    net->change_gid = true;
    net->q_mean = L_Q;
    net->g_index = L_gid;
  }else{
    net->q_mean = my_Q;
  }
}

void cal_kernels_enqueue(Net *net){
    pthread_mutex_lock(&mutex_g[net->device->g_index]);
    
    gpu_list[net->device->g_index].all_api += net->all_api;
    net->q_all_api = gpu_list[net->device->g_index].all_api;
    
    pthread_mutex_unlock(&mutex_g[net->device->g_index]);
}

void cal_kernels_dequeue(Net *net,int index){
    pthread_mutex_lock(&mutex_g[net->device->g_index]);
    gpu_list[net->device->g_index].load -= net->weight;
    gpu_list[net->device->g_index].worker_load += net->weight;
    pthread_mutex_unlock(&mutex_g[net->device->g_index]);
}

void* check_loadbalancing(){
  std::vector< std::pair<int,int> >g_load;

  int sched_cnt=0;

  while(1){
    check_load = true;

    for(int g=0;g<gpu_idx.size();g++){
      pthread_mutex_lock(&mutex_g[gpu_idx[g]]);
      g_load.push_back(std::make_pair(gpu_list[gpu_idx[g]].g_index,gpu_list[gpu_idx[g]].load));
      // g_worker_load.push_back(std::make_pair(gpu_list[gpu_idx[g]].g_index,gpu_list[gpu_idx[g]].load));
      pthread_mutex_unlock(&mutex_g[gpu_idx[g]]);
    }
    // fprintf((fp_before),"%d,%d,%d,%d,%d\n",sched_cnt,g_load[0].second,g_load[1].second,g_load[2].second,g_load[3].second);

    auto L_max = *max_element(g_load.begin(),g_load.end(),[](const auto& lhs, const auto& rhs) {
      return lhs.second < rhs.second;    
    });
    auto L_maxid = L_max.first;

    auto L_min = *min_element(g_load.begin(),g_load.end(),[](const auto& lhs, const auto& rhs) {
      return lhs.second < rhs.second;    
    });
    auto L_minid = L_min.first;

    if(!(L_max.second <= THRESHOLD)){
      auto capacity = abs(L_max.second - THRESHOLD);
      if(capacity){
        int tmp = 0;
        int cntt=1;
        job* job = thpool[L_maxid]->jobqueue.front;
        time_t t = time(NULL);
        struct tm tm=*localtime(&t);

        while(job){
          tmp += (((th_arg*)job->arg)->arg)->weight;
          if((tmp <= capacity)&& (tmp+L_min.second <= (1.5*THRESHOLD))){
            pthread_mutex_lock(&mutex_t[(((th_arg*)job->arg)->arg)->index_n]);
            (((th_arg*)job->arg)->arg)->loadbalancing = true;
            (((th_arg*)job->arg)->arg)->change_gid = true;
            (((th_arg*)job->arg)->arg)->g_index = L_minid;
            pthread_mutex_unlock(&mutex_t[(((th_arg*)job->arg)->arg)->index_n]);
          }
          else{
            tmp -= (((th_arg*)job->arg)->arg)->weight;
          }
          job = job->prev;
          cntt++;
        }
        // fprintf((fp_after),"%d,%d,%d,%d,%d,%d,%d,%d,%d\n",sched_cnt,g_load[0].second,g_load[1].second,g_load[2].second,g_load[3].second,capacity,tmp,L_maxid,L_minid);
      }
    }
  
    g_load.clear();
    sched_cnt++;
    check_load=false;

    // fprintf((fp_load),"%d,%d,%d,%d\n",g_load[0].second,g_load[1].second,g_load[2].second,g_load[3].second);

    usleep(1500000);
  }
}
