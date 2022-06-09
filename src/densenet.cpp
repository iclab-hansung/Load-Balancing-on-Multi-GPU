
#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

// void get_submodule_densenet(torch::jit::script::Module module,Net &net){
//     Layer t_layer;
//     Dummy concat;
// 	for(auto child : module.named_children()){
//         if(child.value.children().size()==0){ //classifier
// 			t_layer.layer = child.value;
//             if(child.name.find("classifier") != std::string::npos) t_layer.name = "classifier";
// 			net.layers.push_back(t_layer);
// 		}else{  //features
//             for(auto block : child.value.named_children()){    //conv0 ,norm0 , relu0, pool, denseblock , transition
// 				if(child.name == "features"){
//                     if(block.value.children().size() == 0){    //conv0 ,norm0 , relu0, pool 
//                         t_layer.layer = block.value;
//                         if(block.name.find("conv") != std::string::npos) t_layer.name = "conv";
//                         else if(block.name.find("norm") != std::string::npos) t_layer.name = "norm";
//                         else if(block.name.find("relu") != std::string::npos) t_layer.name = "relu";
//                         else if(block.name.find("pool") != std::string::npos) t_layer.name = "pool";
// 			            net.layers.push_back(t_layer);
                    
//                     }else{  //Denseblock, Transition
//                         for(auto layer : block.value.named_children()){
//                             if(layer.value.children().size() == 0){  // layers in transition block
//                                 t_layer.layer = layer.value;
//                                 if(layer.name.find("conv")!= std::string::npos) t_layer.name = "conv";
//                                 else if(layer.name.find("norm") != std::string::npos) t_layer.name = "norm";
//                                 else if(layer.name.find("relu") != std::string::npos) t_layer.name = "relu";
//                                 else if(layer.name.find("pool") != std::string::npos) t_layer.name = "pool";
//                                 net.layers.push_back(t_layer);
//                             }else{  //denselayer
//                                 if(layer.name.find("denselayer") != std::string::npos){
//                                     t_layer.from_idx = {-1};
//                                     t_layer.layer = concat;
//                                     t_layer.name = "concat";
//                                     net.layers.push_back(t_layer);
//                                     for(auto in_layer : layer.value.named_children()){  //layers in denselayer
//                                         t_layer.from_idx.clear();
//                                         t_layer.layer = in_layer.value;
//                                         if(in_layer.name.find("conv") != std::string::npos) t_layer.name = "conv";
//                                         else if(in_layer.name.find("norm") != std::string::npos) t_layer.name = "norm";
//                                         else if(in_layer.name.find("relu") != std::string::npos) t_layer.name = "relu";
//                                         else if(in_layer.name.find("pool") != std::string::npos) t_layer.name = "pool";
// 			                            net.layers.push_back(t_layer);
//                                     }
//                                     t_layer.from_idx = {-7, -1};
//                                     t_layer.layer = concat;
//                                     t_layer.name = "concat";
//                                     net.layers.push_back(t_layer);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

void get_submodule_densenet(torch::jit::script::Module module,Net &net){
    Layer t_layer;
    Dummy concat;
	for(auto child : module.named_children()){
        if(child.value.children().size()==0){ //classifier
			t_layer.layer = child.value;
            if(child.name.find("classifier") != std::string::npos) t_layer.name = "classifier";
			net.layers.push_back(t_layer);
		}else{  //features
            for(auto block : child.value.named_children()){    //conv0 ,norm0 , relu0, pool, denseblock , transition
				if(child.name == "features"){
                    if(block.value.children().size() == 0){    //conv0 ,norm0 , relu0, pool 
                        t_layer.layer = block.value;
                        if(block.name.find("conv") != std::string::npos) t_layer.name = "conv";
                        else if(block.name.find("norm") != std::string::npos) t_layer.name = "norm";
                        else if(block.name.find("relu") != std::string::npos) t_layer.name = "relu";
                        else if(block.name.find("pool") != std::string::npos) t_layer.name = "pool";
			            net.layers.push_back(t_layer);
                    
                    }else{  //Denseblock, Transition
                        for(auto layer : block.value.named_children()){
                            if(layer.value.children().size() == 0){  // layers in transition block
                                t_layer.layer = layer.value;
                                if(layer.name.find("conv")!= std::string::npos) t_layer.name = "conv";
                                else if(layer.name.find("norm") != std::string::npos) t_layer.name = "norm";
                                else if(layer.name.find("relu") != std::string::npos) t_layer.name = "relu";
                                else if(layer.name.find("pool") != std::string::npos) t_layer.name = "pool";
                                net.layers.push_back(t_layer);
                            }else{  //denselayer
                                if(layer.name.find("denselayer") != std::string::npos){
                                    t_layer.from_idx = {-1};
                                    t_layer.layer = concat;
                                    t_layer.name = "concat";
                                    net.layers.push_back(t_layer);
                                    for(auto in_layer : layer.value.named_children()){  //layers in denselayer
                                        t_layer.from_idx.clear();
                                        t_layer.layer = in_layer.value;
                                        if(in_layer.name.find("conv") != std::string::npos) t_layer.name = "conv";
                                        else if(in_layer.name.find("norm") != std::string::npos){
                                            t_layer.name = "norm";
                                            t_layer.from_idx = {-4,-1};
                                        }
                                        else if(in_layer.name.find("relu") != std::string::npos) t_layer.name = "relu";
                                        else if(in_layer.name.find("pool") != std::string::npos) t_layer.name = "pool";
			                            net.layers.push_back(t_layer);
                                    }
                                    t_layer.from_idx = {-7, -1};
                                    t_layer.layer = concat;
                                    t_layer.name = "concat";
                                    net.layers.push_back(t_layer);
                                    t_layer.from_idx.clear();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void *predict_densenet_warming(Net *dense){
    {
        at::cuda::CUDAGuard guard(dense->device->g_device);
        int i;
        Net *net = dense;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        std::cout<<"warming_dense\n\n";
        // for(int iter=0;iter<ITERATION;iter++){
            
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            th_arg th;
            net->input=inputs[dense->device->g_index];
            net->cur_round_last = 0;

            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[dense->index_n]);
                cond_i[dense->index_n] = 1;
                net->index = i;
                net->all_api = 0;

                //multi_GPU 시 loss 비교 후 device[] 선택

                pthread_mutex_lock(&mutex_g[net->device->g_index]);

                net->device->load += net->weight;
                net->layers[net->index].l_load = net->device->load; //왼쪽이 net->load 여도되지않나..?
                
                pthread_mutex_unlock(&mutex_g[net->device->g_index]);

                //net->timeslice = cal_timeslice(dense->device->load,net->weight);
                // std::cout<<"net : "<<net->index_n<<" , my weight : "<<dense->weight<<" , timeslice : "<<net->timeslice<<std::endl<<std::endl;
                int j=i;
                do{
                    l_sum += net->layers[j].l_mean;
                    net->all_api += net->layers[j].l_api;
                    // std::cout<<"l_sum : "<<l_sum<<" , l_api : "<<net->layers[j].l_api<<std::endl;
                    if(j == net->last){
                        net->cur_round_last = j;
                        break;
                    }else if(l_sum > net->timeslice){
                        l_sum -= net->layers[j].l_mean;
                        net->cur_round_last = net->layers[j].l_prev;
                        net->all_api -= net->layers[j].l_api;
                        break;
                    }else if(l_sum == net->timeslice){
                        net->cur_round_last = j;
                        break;
                    }
                    j=net->layers[j].l_next;
                }while(j<net->layers.size());
                
                // std::cout<<net->index_n<<" this round :"<<round<<" , timeslice : "<<net->timeslice<<" , cur index : "<<i<<" , cur_round_last : "<<net->cur_round_last<<std::endl<<std::endl;
                // net->layers[net->cur_round_last].round_last = true; //이번 round 가 끝났는지 확인을 위한 flag 필요??????
                th.arg = net;
                // std::cout<<"layer index : "<<net->index<<" last : " <<net->cur_round_last<<std::endl;
                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_densenet,&th);

                while (cond_i[net->index_n] == 1){
                    // std::cout<<"layer signal wait\n\n";
                    pthread_cond_wait(&cond_t[net->index_n], &mutex_t[net->index_n]);
                }

                l_sum = 0.0;
                real_sum = 0.0;
                round += 1;
                i = net->index;
                net->input.clear();
                net->input.push_back(net->layers[i].output);
                pthread_mutex_unlock(&mutex_t[net->index_n]);
            }
        // }
        cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);

        std::cout << "\n*****"<<dense->name<<" "<<dense->index_n<<" result  "<<time<<"ms ***** \n";
    }
} 

void *predict_densenet(std::vector<Net*> *vec_dense){
    {
        at::cuda::CUDAGuard guard((*vec_dense)[(*vec_dense)[gpu_idx[0]]->g_index]->device->g_device);
        Net *densenet = (*vec_dense)[(*vec_dense)[gpu_idx[0]]->g_index];

        Net *net = densenet;
        float t_time,real;
        cudaEvent_t start, end,real_start;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventCreate(&real_start);

        int iter = 0;
        int round = 0;

        pthread_mutex_lock(&mutex_g[net->device->g_index]);
        gpu_list[net->device->g_index].load += net->weight;
        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
        
        cudaEventRecord(real_start);
        
        for(iter=0;iter<ITERATION;iter++){
            std::vector< std::pair<int,at::Tensor> >vv;
            int i;
            float l_sum = 0.0;
            float real_sum = 0.0;
            bool next_concat = false;
            net->cur_round_last = 0;
            th_arg th;
            cudaEventRecord(start);
            
            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[densenet->index_n]);
                cond_i[densenet->index_n] = 1;
                net->index = i;
                net->all_api = 0;
                float cpy_mem = 0.0; 
                float q_pred = 0.0;
                // multi_GPU 시 loss 비교 후 device[] 선택
                // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                if(!(iter == 0 && i == 0)){
                    if(i != 0){
                        vv.clear();
                        if(((net->layers[net->index].name == "concat") || (net->layers[net->index].name == "norm" && net->layers[net->layers[net->index].l_prev].name == "norm"))){
                            for(int j=0;j<net->layers[net->index].from_idx.size();j++){
                                // std::cout<<(net->index) + net->layers[(net->index)].from_idx[j]<<std::endl;
                                cpy_mem += net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].l_mem;
                                vv.push_back(std::make_pair((net->index) + net->layers[(net->index)].from_idx[j],net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].output));
                            }    
                        }else{  //나중에 합치자.
                            cpy_mem += net->layers[net->layers[i].l_prev].l_mem;
                            // std::cout<<cpy_mem<<std::endl;
                            vv.push_back(std::make_pair((net->index-1),net->layers[(net->index-1)].output));
                            // std::cout<<"h÷ere?222\n";
                        }
                    }
                    // pthread_mutex_lock(&mutex_l);
                    //check_load=true;
                    // int L_idx = get_lowest_load_idx(net);  //net 넘겨줘야할지도...?
                    //check_load=false;
                    // pthread_mutex_unlock(&mutex_l);
                    
                    if(net->loadbalancing){
                        //time_t t = time(NULL);
                        //struct tm tm=*localtime(&t);
                        pthread_mutex_lock(&mutex_g[net->device->g_index]);
                        gpu_list[net->device->g_index].load -= net->weight;
                        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                        net->loadbalancing=false;
                        q_pred=0;   //필요..?
                        //fprintf((fp_load),"AFTER LOADBALANCING current : %d , net : %d , time : %d-%d-%d\n",net->g_index,net->index_n,tm.tm_hour,tm.tm_min,tm.tm_sec);
                    }else{
                        //int L_idx = get_lowest_load_idx(net);  //net 넘겨줘야할지도...?
                        // int selec_q = select_queue(net,net->device->g_index,L_idx,cpy_mem);
                        //select_queue(net,net->g_index,L_idx,cpy_mem);
                        //q_pred = net->q_mean;
                    }
                    if(net->change_gid){    //GPU index 가 바뀌었을 경우 to
                        net->change_gid = false;
                        densenet = (*vec_dense)[net->g_index];
                        net=densenet;

                        pthread_mutex_lock(&mutex_g[net->device->g_index]);
                        gpu_list[net->device->g_index].load += net->weight;
                        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                        
                        net->index = i;
                        if(i!=0){
                            for(int j=0;j<vv.size();j++){
                                // std::cout<<" vv first : "<<vv[j].first<<std::endl;
                                net->layers[vv[j].first].output = vv[j].second.to(net->device->g_device);   //-1 이 결국 input 
                            }
                            net->input.clear();
                            net->input.push_back(net->layers[i-1].output);
                        }else{
                             net->input.clear();
                            net->input = inputs[net->device->g_index];
                        }
                       
                        // std::cout<<"weight : "<<net->weight<<std::endl;
                    }
                }
              
                pthread_mutex_lock(&mutex_g[net->device->g_index]);
                net->layers[net->index].l_load = gpu_list[net->device->g_index].load; //왼쪽이 net->load 여도되지않나..?
                pthread_mutex_unlock(&mutex_g[net->device->g_index]);

                net->timeslice = cal_timeslice(gpu_list[net->device->g_index].load,net->weight);

                int j=i;
                do{
                    l_sum += net->layers[j].l_mean;
                    net->all_api += net->layers[j].l_api;
                    if(j == net->last){
                        net->cur_round_last = j;
                        break;
                    }else if(l_sum > net->timeslice){
                        l_sum -= net->layers[j].l_mean;
                        net->cur_round_last = net->layers[j].l_prev;
                        net->all_api -= net->layers[j].l_api;
                        break;
                    }else if(l_sum == net->timeslice){
                        net->cur_round_last = j;
                        break;
                    }
                    j=net->layers[j].l_next;
                }while(j<net->layers.size());
                
                th.arg = net;
                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_densenet,&th);

                while (cond_i[net->index_n] == 1){
                    pthread_cond_wait(&cond_t[net->index_n], &mutex_t[net->index_n]);
                }

                #if RECORD
                    if(net->warming==true){
                        // real_sum += net->layers[i].l_time;
                        #if Q_OVERHEAD
                            //fprintf((net->fp),"%d,%lf\n",i,eq_time);
                            //std::cout<<"write index : "<<i<<"   "<<net->layers[i].q_time<<std::endl;
                            #if L_RECORD
                                //if(!nice_0_end){
                                    fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",nice_0_end,round,i,net->nice,net->weight,net->device->g_index,net->layers[i].l_load,net->timeslice,q_pred,net->layers[i].q_time,l_sum,net->layers[i].l_time);
                                //}
                                // fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",round,i,net->all_api,net->q_all_api,net->device->g_index,net->layers[i].l_load,net->timeslice,q_pred,net->layers[i].q_time,l_sum,net->layers[i].l_time);
                                    // std::cout<<"\nround : "<<round<<" weight : "<<net->weight<<" index : "<<i<<" my api : "<<net->all_api<< " q api :"<<net->q_all_api<<" load : "<<net->layers[i].l_load<<" ts : "<<net->timeslice<<" Q pred : "<<q_pred<<" Q time : "<<net->layers[i].q_time<<" l_sum : "<<l_sum<<" runtime :"<<net->layers[i].l_time<<std::endl;
                            #else
                                //fprintf((net->fp),"%d,%d,%lf\n",i,net->layers[i].all_api,net->layers[i].q_time);
                            #endif
                        #else
                            //fprintf((net->fp),"%d,%d,%lf\n",i,net->layers[i].all_api);
                        #endif
                    }
                #endif

                l_sum = 0.0;
                // real_sum = 0.0;
                round += 1;
                int tmp = i;
                i = net->index;
                pthread_mutex_unlock(&mutex_t[net->index_n]);
            }
            net->index=0;
            // round = 0;
            net->input=inputs[densenet->device->g_index];
    
            cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&t_time, start, end);
            cudaEventElapsedTime(&real, real_start, end);

            // #if RECORD
            //     if(densenet->warming == true){
            //         fprintf((fp_res),"DenseNet,%d,%d,%d,%d,%lf,%lf\n",densenet->index_n,iter,net->nice,net->weight,time,real);
            //     }   
            // #endif
            
            std::cout << "\n*****"<<densenet->name<<" "<<densenet->index_n<<" result  "<<t_time<<"ms ***** \n";
            // std::cout << (densenet->layers[net->last].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
        }
        pthread_mutex_lock(&mutex_g[net->device->g_index]);
        gpu_list[net->device->g_index].load -= net->weight;
        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
        
        if(net->weight==1024){
            if(!nice_0_end) nice_0_end = true;
            fprintf((fp_before)," %d end\n",net->index_n);
        }
    }
}

void forward_densenet(th_arg *th){
    // pthread_mutex_lock(&mutex_t[th->arg->index_n]);
	{
        //at::cuda::CUDAStreamGuard guard(streams[th->arg->g_index][(th->arg->index_s)%(n_streamPerPool)]);
        at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[(th->arg->index_s)%(n_streamPerPool)]);
        pthread_mutex_lock(&mutex_t[th->arg->index_n]);
        // std::cout<<th->arg->index_n<<" in forward g_index= "<<th->arg->device->g_index<<"\n";
        // std::cout<<"forward start\n";
        #if L_RECORD
            cudaEvent_t l_start, l_end;
            float l_time;
            cudaEventCreate(&l_start);
            cudaEventCreate(&l_end);
            cudaEventRecord(l_start);
        #endif

        #if NVTX
        char str[30];
        sprintf(str, "Dense layer - %d", th->arg->index);
        nvtxRangeId_t id1 = nvtxRangeStartA(str);
        #endif

        Net *net = th->arg;
        // std::cout<<"*JOB Forward "<<net->index_n<<"  index : "<<net->index<<" , name : "<<net->layers[net->index].name<<" , tensor device : "<<net->input[0].toTensor().device()<<"*\n";

        int k = net->index;
        for(k;k<=net->cur_round_last;k++){
            std::vector<torch::jit::IValue> inputs = net->input;    
            at::Tensor out;
            // std::cout<<"*In JOB "<<net->index_n<<"  index : "<<k<<" , name : "<<net->layers[k].name<<" , tensor device : "<<inputs[0].toTensor().device()<<"*\n";
            // std::cout<<"k : "<<k<<"  , name :"<<net->layers[k].name<<std::endl;
        
            if(k == net->flatten){ 
                // std::cout<<"flatten :"<<k<<std::endl;
                out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
                out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
                out = out.view({out.size(0), -1});
                inputs.clear();
                inputs.push_back(out);
                out = net->layers[k].layer.forward(inputs).toTensor();
            }
            else if(net->layers[k].name == "concat"){
                std::vector<at::Tensor> cat_input;
                // std::cout<<"start concat\n\n";
                for(int i=0;i<net->layers[k].from_idx.size();i++){
                    cat_input.push_back(net->layers[k + net->layers[k].from_idx[i]].output);
                }
                // std::cout<<"befor concat\n";
                out = torch::cat(cat_input, 1);
            }
            else{
                out = net->layers[k].layer.forward(inputs).toTensor();
                if(k+1 < net->layers.size() && net->layers[k+1].name == "norm" && net->layers[k+1].name != "pool"){
                    net->layers[k].output = out;
                    k++;
                    inputs.clear();
                    inputs.push_back(out);
                    out = net->layers[k].layer.forward(inputs).toTensor();
                }
                if(k+1 < net->layers.size() && net->layers[k+1].name == "relu"){
                    net->layers[k].output = out;
                    k++;
                    inputs.clear();
                    inputs.push_back(out);
                    out = net->layers[k].layer.forward(inputs).toTensor();
                }
                if(k+1 < net->layers.size() && net->layers[k+1].name == "conv"){
                    net->layers[k].output = out;
                    k++;
                    inputs.clear();
                    inputs.push_back(out);
                    out = net->layers[k].layer.forward(inputs).toTensor();
                }
            }
            net->layers[k].output = out;    //필요?
            net->input.clear();
            net->input.push_back(net->layers[k].output);
            // std::cout<<"k : "<<k<<"end"<<std::endl;
        } k--;
        #if L_SYNC
            cudaStreamSynchronize(net->device.streams[net->index_s%(n_streamPerPool)]); // 나중에 지워야함
        #endif
                
        #if NVTX
        nvtxRangeEnd(id1);
        #endif

        #if L_RECORD
			cudaEventRecord(l_end);
			cudaEventSynchronize(l_end);
			cudaEventElapsedTime(&l_time, l_start, l_end);
            net->layers[net->index].l_time = l_time;
		#endif

        net->index = k;
        // std::cout<<th->arg->index_n<<" forward END , g_index= "<<th->arg->device->g_index<<"\n";

    cond_i[th->arg->index_n]=0;
    // std::cout<<"here\n";
    pthread_cond_signal(&cond_t[th->arg->index_n]);
	pthread_mutex_unlock(&mutex_t[th->arg->index_n]);
    }
}