
#include "resnet.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_resnet(torch::jit::script::Module module, Net &net){
    Layer t_layer;
    t_layer.round_last = false;
    for(auto child : module.named_children()){
        if(child.value.children().size() == 0){
            t_layer.layer = child.value;
            if(child.name == "conv1") t_layer.name = "conv1";
            else if(child.name == "bn1") t_layer.name = "bn_relu";
            else if(child.name == "relu") continue;//t_layer.name = "relu";
            else if(child.name == "maxpool") t_layer.name = "maxpool";
            else if(child.name == "avgpool")t_layer.name = "avgpool";
            else if(child.name == "fc") t_layer.name = "fc";
            net.layers.push_back(t_layer);
        }else{  //layer
            for(auto block : child.value.named_children()){ //Basicblock, Bottleneck
                for(auto layer : block.value.named_children()){ // in block
                    t_layer.layer = layer.value;
                    if(layer.name == "conv1") t_layer.name = "conv1";
                    else if(layer.name == "conv2") t_layer.name = "conv2";
                    else if(layer.name == "conv3") t_layer.name = "conv3";
                    else if(layer.name == "relu") continue;
                    else if(layer.name == "bn1" || layer.name == "bn2") t_layer.name = "bn_relu";
                    else if(layer.name == "bn3") t_layer.name = "bn3";
                    else if(layer.name == "downsample") t_layer.name = "downsample";
                    net.layers.push_back(t_layer);
                }
            }
        }
    }
}

void *predict_resnet_warming(Net *res){
    {
        at::cuda::CUDAGuard guard(res->device->g_device);
        int i=0;
        Net *net = res;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        // for(int iter=0;iter<ITERATION;iter++){
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            th_arg th;
            net->input=inputs[net->device->g_index];
            net->cur_round_last = 0;

            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[net->index_n]);
                // std::cout<<"i = "<<i<<" g_inddex : "<<net->device->g_index<<std::endl;
                cond_i[net->index_n] = 1;
                net->index = i;
                net->all_api = 0;

                pthread_mutex_lock(&mutex_g[net->device->g_index]);

                gpu_list[net->device->g_index].load += net->weight;
                net->layers[net->index].l_load = gpu_list[net->device->g_index].load; //왼쪽이 net->load 여도되지않나..?
                
                pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                // net->timeslice = cal_timeslice(gpu_list[net->device->g_index].load,net->weight);

                int j=i;
                do{
                    l_sum += net->layers[j].l_mean;
                    net->all_api += net->layers[j].l_api;
                    // std::cout<<"layer index : "<<j<<" l_api : "<<net->layers[j].l_api<<std::endl;
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
                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_resnet,&th);

                while (cond_i[net->index_n] == 1){
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
        
        std::cout << "\n*****"<<res->name<<" "<<res->index_n<<" result  "<<time<<"ms ***** \n";
    }
} 

void *predict_resnet(std::vector<Net*> *vec_res){
    {
        at::cuda::CUDAGuard guard(gpu_list[(*vec_res)[gpu_idx[0]]->g_index].g_device);
        Net *resnet = (*vec_res)[(*vec_res)[gpu_idx[0]]->g_index];
        Net *net = resnet;
        // std::cout<<net->layers[29].l_mem<<std::endl;
        // nice(net->nice);
        float time,real;
        cudaEvent_t start, end,real_start;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventCreate(&real_start);

        int iter;
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
            // std::cout<<"------------- res "<<net->index_n<<" ITER "<<iter<<"------------------\n";
            // net->input=inputs[resnet->device->g_index];
            // std::cout<<"Net index : "<<net->index_n<<" , input gid : "<<resnet->device->g_index<<std::endl;
            // std::cout<<"net index : "<<net->index_n<<"  /  current device : "<<(int)c10::cuda::current_device()<<std::endl; 
            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[resnet->index_n]);
                cond_i[resnet->index_n] = 1;
                net->index = i;
                net->all_api = 0;
                float cpy_mem = 0.0; 
                float q_pred = 0.0;
                at::Tensor prev_out;
                at::Tensor prev_identity;
                // multi_GPU 시 loss 비교 후 device[] 선택
                // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                if(!(iter==0 && i ==0)){
                    // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                    if(i!=0){
                        if(net->layers[i].name != "conv1"){
                            prev_identity = net->identity;
                            cpy_mem += net->layers[net->layers[i].l_prev].l_identity;
                        }
                        prev_out = net->layers[i-1].output;
                        cpy_mem += net->layers[net->layers[i].l_prev].l_mem;
                        // std::cout <<std::endl<<net->index_n<<" layer index : "<<net->index<<" , cpy mem : "<<cpy_mem<<" l_mem : "<<net->layers[net->layers[i].l_prev].l_mem<<std::endl;
                    }

                    // if(i!=0){
                    //     if((i!=net->last) && ((net->layers[net->index].name=="concat")||(net->layers[net->index].name=="norm"))){
                    //         // next_concat = true;
                    //         // std::cout<<"here?\n";
                    //         vv.clear();
                    //         for(int j=0;j<net->layers[net->index].from_idx.size();j++){
                    //             cpy_mem += net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].l_mem;
                    //             vv.push_back(std::make_pair((net->index) + net->layers[(net->index)].from_idx[j],net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].output));
                    //         }    
                    //     }else{  //나중에 합치자.
                    //         vv.clear();
                    //         cpy_mem += net->layers[(net->index-1)].l_mem;
                    //         std::cout<<cpy_mem<<std::endl;
                    //         vv.push_back(std::make_pair((net->index-1),net->layers[(net->index-1)].output));
                    //         // std::cout<<"h÷ere?222\n";
                    //     }
                    // }
                    // int L_idx; //= get_lowest_load_idx();
                    // int selec_q ;//= select_queue(net,net->g_index,L_idx,cpy_mem);
                    // q_pred = net->q_mean;
                    // std::cout<< "q_pred : "<<q_pred<<"\n\n";
                    //std::cout<<"Predict "<<net->index_n<<"  index : "<<net->index<<" , name : "<<net->layers[net->index].name<<" , change : "<<net->change_gid<<std::endl;
                    if(net->loadbalancing){
                        //time_t t = time(NULL);
                        //struct tm tm=*localtime(&t);
                        pthread_mutex_lock(&mutex_g[net->device->g_index]);
                        gpu_list[net->device->g_index].load -= net->weight;
                        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                        net->loadbalancing=false;
                        q_pred=0;   //필요..?
                        //fprintf((fp_load),"AFTER LOADBALANCING current : %d , net : %d , time : %d-%d-%d\n",net->g_index,net->index_n,tm.tm_hour,tm.tm_min,tm.tm_sec);
                    }
                    
                    if(net->change_gid){    //GPU index 가 바뀌었을 경우 to
                        net->change_gid = false;
                        resnet = (*vec_res)[net->g_index];
                        net=resnet;

                        pthread_mutex_lock(&mutex_g[net->device->g_index]);
                        gpu_list[net->device->g_index].load += net->weight;
                        pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                        
                        net->index = i;
                        if(i!=0){
                            if(net->layers[i].name != "conv1"){
                                net->identity = prev_identity.to(gpu_list[net->device->g_index].g_device);
                            }
                            // net->identity = prev_identity.to(gpu_list[net->g_index].g_device);
                            net->layers[i-1].output = prev_out.to(gpu_list[net->device->g_index].g_device);
                            net->input.clear();
                            net->input.push_back(net->layers[i-1].output);
                        }else{
                            net->input.clear();
                            net->input=inputs[net->device->g_index];
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
                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_resnet,&th);

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
                                fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",nice_0_end,round,i,net->nice,net->weight,net->device->g_index,net->layers[i].l_load,net->timeslice,q_pred,net->layers[i].q_time,l_sum,net->layers[i].l_time);
                                    // std::cout<<"\nround : "<<round<<" weight : "<<net->weight<<" index : "<<i<<" my api : "<<net->all_api<< " q api :"<<net->q_all_api<<" load : "<<net->layers[i].l_load<<" ts : "<<net->timeslice<<" Q pred : "<<q_pred<<" Q time : "<<net->layers[i].q_time<<" l_sum : "<<l_sum<<" runtime :"<<net->layers[i].l_time<<std::endl;
                                // fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",round,net->weight,i,net->all_api,net->q_all_api,net->layers[i].l_load,net->timeslice,q_pred,net->layers[i].q_time,l_sum,net->layers[i].l_time);
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
            round = 0;
            net->input=inputs[net->device->g_index]; //필요?
    
            // cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
            cudaStreamSynchronize(streams[net->device->g_index][net->index_s%(n_streamPerPool)]);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, start, end);
            cudaEventElapsedTime(&real, real_start, end);

            // #if RECORD
                // if(resnet->warming == true){
                    // fprintf((fp_res),"ResNet,%d,%d,%d,%d,%lf,%lf\n",resnet->index_n,iter,net->nice,net->weight,time,real);
                // }   
            // #endif
            
            std::cout << "\n*****"<<resnet->name<<" "<<resnet->index_n<<" result  "<<time<<"ms ***** \n";
            // std::cout << (resnet->layers[net->last].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
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

void forward_resnet(th_arg *th){
    pthread_mutex_lock(&mutex_t[th->arg->index_n]);
    {
        at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[th->arg->index_s%(n_streamPerPool)]);
        #if L_RECORD
            cudaEvent_t l_start, l_end;
            float l_time;
            cudaEventCreate(&l_start);
            cudaEventCreate(&l_end);
            cudaEventRecord(l_start);
        #endif

        #if NVTX
            char str[30];
            sprintf(str, "Resnet layer - %d", th->arg->index);
            nvtxRangeId_t id1 = nvtxRangeStartA(str);
        #endif
        Net *net = th->arg;
        int k = net->index;
        // std::cout<<"k : "<<k<<" last : "<<net->cur_round_last<<std::endl;
        for(k;k<=net->cur_round_last;k++){
            // std::cout<<"layer "<<k<<" , last : "<<net->layers[k].round_last<<std::endl;
            std::vector<torch::jit::IValue> inputs = net->input; 
            at::Tensor identity;
            vector<torch::jit::IValue> inputs_cpy;
            at::Tensor out;
            if(net->layers[k].name == "conv1"){ 
                identity = inputs[0].toTensor(); 
            }else{
                identity = net->identity;
            }
            if(k == net->flatten) //flatten
            {	 
                out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
                inputs.clear();
                inputs.push_back(out);
                out = net->layers[k].layer.forward(inputs).toTensor();
            } 
            else if(net->layers[k].name == "conv3"){ 
                out = net->layers[k].layer.forward(inputs).toTensor();
                if(k+1 < net->layers.size() && net->layers[k+1].name == "bn3"){
                    net->layers[k].output = out;
                    k++;
                    inputs.clear();
                    inputs.push_back(out);
                    out = net->layers[k].layer.forward(inputs).toTensor();
                }
                if(net->layers[k+1].name != "downsample" ){
                    out += identity;
                    out = torch::relu(out);
                }
            }
            else if(net->layers[k].name == "downsample"){   // downsample
                    inputs_cpy.clear();
                    inputs_cpy.push_back(identity); 
                    identity = net->layers[k].layer.forward(inputs_cpy).toTensor();
                    out = net->layers[k-1].output;
                    out += identity;
                    out = torch::relu(out);
                }         
            else{ 
                out = net->layers[k].layer.forward(inputs).toTensor();
                if(net->layers[k+1].name=="bn_relu"){
                    net->layers[k].output = out;
                    k++;
                    inputs.clear();
                    inputs.push_back(out);
                    out = net->layers[k].layer.forward(inputs).toTensor();  //bn1,2
                    out = torch::relu(out); // relu after bn1,2
                }
            }
            net->layers[k].output = out;    //필요?
            net->identity = identity;
            net->input.clear();
            net->input.push_back(net->layers[k].output);
        } k--; //because of for(;k++)

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
            //std::cout<<"layer index : "<<net->index<<"      "<<net->layers[net->index].l_time<<std::endl;
		#endif

        net->index = k;
    }
    cond_i[th->arg->index_n]=0;
	pthread_cond_signal(&cond_t[th->arg->index_n]);
	pthread_mutex_unlock(&mutex_t[th->arg->index_n]);
}
