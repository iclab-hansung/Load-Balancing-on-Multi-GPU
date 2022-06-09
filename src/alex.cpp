#include "alex.h"

namespace F = torch::nn::functional;

void get_submodule_alexnet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	for(auto child : module.named_children()){
		if(child.value.children().size()==0){	//avgpool
			t_layer.layer = child.value;
			t_layer.name = "avgpool";
			net.layers.push_back(t_layer);
		}
		else{	//feature , classifier
			for(auto ch : child.value.named_children()){
				if(child.name == "features"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" || ch.name == "6" || ch.name == "8" || ch.name == "10"){
						t_layer.name = "conv";
					}
					else if(ch.name == "1" || ch.name == "4" || ch.name == "7" || ch.name == "9" || ch.name == "11"){
						t_layer.name = "relu";
					}
					else if(ch.name == "2" || ch.name == "5" || ch.name == "12"){
						t_layer.name = "maxpool";
					}
					net.layers.push_back(t_layer);
				}
				else if(child.name == "classifier"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" ) continue;		//dropout
					else if(ch.name == "1" || ch.name == "4" || ch.name == "6"){
						t_layer.name = "linear";
					}
					else if(ch.name == "2" || ch.name == "5"){
						t_layer.name = "relu";
					}
					net.layers.push_back(t_layer);
				}
			}
		}
	}
}

void *predict_alexnet_warming(Net *alex){
    {   
        at::cuda::CUDAGuard guard(alex->device->g_device);

        Net *net = alex;
        cudaEvent_t start, end;
        float time; //eventrecord
        // struct timespec start_ts,end_ts;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        // for(int iter=0;iter<ITERATION;iter++){
        
            int i;
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            th_arg th;
            net->input=inputs[alex->device->g_index];
            net->cur_round_last = 0;

        //  #if RECORD
		// 	cudaEvent_t q_end;
		// 	float q_time;
        //     cudaEventCreate(&q_end);
		// #endif

            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[alex->index_n]);
                cond_i[alex->index_n] = 1;
                net->index = i;
                net->all_api = 0;

                pthread_mutex_lock(&mutex_g[net->device->g_index]);

                gpu_list[net->device->g_index].load += net->weight;
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
                
                // std::cout<<net->index_n<<" this round :"<<round<<" , timeslice : "<<net->timeslice<<" , cur index : "<<i<<" , cur_round_last : "<<net->cur_round_last<<std::endl<<std::endl;
                // net->layers[net->cur_round_last].round_last = true; //이번 round 가 끝났는지 확인을 위한 flag 필요??????
                th.arg = net;

                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_alexnet,&th);

                while (cond_i[net->index_n] == 1){
                    // std::cout<<"layer signal wait\n\n";
                    pthread_cond_wait(&cond_t[net->index_n], &mutex_t[net->index_n]);
                }
                // net->device->total_weight -= net->weight;
                l_sum = 0.0;
                real_sum = 0.0;
                round += 1;
                i = net->index;
                net->input.clear();
                net->input.push_back(net->layers[i].output);
                pthread_mutex_unlock(&mutex_t[net->index_n]);
            }
        
        cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);

        std::cout << "\n*****"<<alex->name<<" "<<alex->index_n<<" result  "<<time<<"ms ***** \n";
		// std::cout << (alex->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
    }
}

void *predict_alexnet(std::vector<Net*> *vec_alex){
    {
        at::cuda::CUDAGuard guard(gpu_list[(*vec_alex)[0]->g_index].g_device);
        Net *alexnet = (*vec_alex)[(*vec_alex)[0]->g_index];
        Net *net = alexnet;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        int iter;

        for(iter=0;iter<ITERATION;iter++){
            std::vector< std::pair<int,at::Tensor> >vv;
            int i;
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            bool next_concat = false;
            net->cur_round_last = 0;
            th_arg th;
            cudaEventRecord(start);
            // std::cout<<"------------- alex "<<net->index_n<<" ITER "<<iter<<"------------------\n";
            // net->input=inputs[alexnet->device->g_index];
            // std::cout<<"Net index : "<<net->index_n<<" , input gid : "<<alexnet->device->g_index<<std::endl;
            // std::cout<<"net index : "<<net->index_n<<"  /  current device : "<<(int)c10::cuda::current_device()<<std::endl; 
            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[alexnet->index_n]);
                cond_i[alexnet->index_n] = 1;
                net->index = i;
                net->all_api = 0;
                float cpy_mem = 0.0; 
                float q_pred = 0.0;
                at::Tensor prev_out;
                //at::Tensor prev_identity;
                // multi_GPU 시 loss 비교 후 device[] 선택
                // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                if(!(iter==0 && i ==0)){
                    // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                    if(i!=0){
                        cpy_mem += net->layers[i-1].l_mem;
                        //prev_identity = net->identity;
                        prev_out = net->layers[i-1].output;
                    }
                    int L_idx; //= get_lowest_load_idx();
                    int selec_q;// = select_queue(net,net->g_index,L_idx,cpy_mem);
                    q_pred = net->q_mean;
                    // std::cout<< "q_pred : "<<q_pred<<"\n\n";
                    //std::cout<<"Predict "<<net->index_n<<"  index : "<<net->index<<" , name : "<<net->layers[net->index].name<<" , change : "<<net->change_gid<<std::endl;
                    if(net->change_gid){    //GPU index 가 바뀌었을 경우 to
                        net->change_gid = false;
                        //at::Tensor prev_out= net->input[0].toTensor();
                        alexnet = (*vec_alex)[selec_q];
                        alexnet->g_index = selec_q;
                        // prev_out = prev_out.to(alexnet->device->g_device);
                        net=alexnet;
                        net->index = i;
                        net->input.clear();
                        if(i!=0){
                            net->layers[i-1].output = prev_out.to(gpu_list[net->g_index].g_device);
                            net->input.push_back(net->layers[i-1].output);
                        }else{
                            net->input=inputs[net->g_index];
                        }
                    }
                }
                pthread_mutex_lock(&mutex_g[net->g_index]);

                gpu_list[net->g_index].load += net->weight;
                net->layers[net->index].l_load = gpu_list[net->g_index].load; //왼쪽이 net->load 여도되지않나..?
                
                pthread_mutex_unlock(&mutex_g[net->g_index]);

                net->timeslice = cal_timeslice(gpu_list[net->g_index].load,net->weight);

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
                
                thpool_add_work(thpool[net->g_index],(void(*)(void *))forward_alexnet,&th);

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
                        
                                    // std::cout<<"\nround : "<<round<<" weight : "<<net->weight<<" index : "<<i<<" my api : "<<net->all_api<< " q api :"<<net->q_all_api<<" load : "<<net->layers[i].l_load<<" ts : "<<net->timeslice<<" Q pred : "<<q_pred<<" Q time : "<<net->layers[i].q_time<<" l_sum : "<<l_sum<<" runtime :"<<net->layers[i].l_time<<std::endl;
                                fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",round,net->weight,i,net->all_api,net->q_all_api,net->layers[i].l_load,net->timeslice,q_pred,net->layers[i].q_time,l_sum,net->layers[i].l_time);
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
            net->input=inputs[net->g_index]; //필요?
    
            // cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
            cudaStreamSynchronize(streams[net->g_index][net->index_s%(n_streamPerPool)]);
            cudaEventRecord(end);
            // if(net->warming == true){
            //     pthread_mutex_lock(&mutex_g[net->device->g_index]);
                
            //     net->device->n_net -= 1;

            //     pthread_mutex_unlock(&mutex_g[net->device->g_index]);
            // }
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, start, end);

            #if RECORD
                if(alexnet->warming == true){
                    // fprintf((fp_res),"%d,%d,%d,%lf\n",alexnet->index_n,iter,net->weight,time);
                }   
            #endif
            
            std::cout << "\n*****"<<alexnet->name<<" "<<alexnet->index_n<<" result  "<<time<<"ms ***** \n";
            std::cout << (alexnet->layers[net->last].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
        }
    }
}

void forward_alexnet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->index_n]);
	{
		at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[(th->arg->index_s)%(n_streamPerPool)]);
		
		#if NVTX
			char str[30];
			sprintf(str, "Alex layer - %d", th->arg->index);
			nvtxRangeId_t id1 = nvtxRangeStartA(str);
		#endif

		#if L_RECORD
			cudaEvent_t l_start, l_end;
			float l_time;
			cudaEventCreate(&l_start);
			cudaEventCreate(&l_end);
			cudaEventRecord(l_start);
		#endif

		Net *net = th->arg;
		int k = net->index;
        for(k;k<=net->cur_round_last;k++){
            std::cout<<net->index_n<<"  index : "<<k<<" name : "<<net->layers[k].name<<" g_index : "<<net->g_index<<net->input[0].toTensor().device()<<std::endl;
            std::vector<torch::jit::IValue> inputs = net->input;
            at::Tensor out;

            if(k==net->flatten){	//flatten
                std::cout<<"flatten\n";
                out = inputs[0].toTensor().view({net->layers[k-1].output.size(0), -1});
                inputs.clear();
                inputs.push_back(out);
                //out = net->layers[k].layer.forward(inputs).toTensor();
            }
            out = net->layers[k].layer.forward(inputs).toTensor();
            if(k+1 < net->layers.size() && net->layers[k+1].name == "relu"){
                net->layers[k].output = out;
                k++;
                inputs.clear();
                inputs.push_back(out);
                out = net->layers[k].layer.forward(inputs).toTensor();
            }
            net->layers[k].output = out;    //필요?
            net->input.clear();
            net->input.push_back(net->layers[k].output);
        } k--;           
            #if L_SYNC
                cudaStreamSynchronize(net->device.streams[net->index_s%(n_streamPerPool)]);
            #endif
        
            #if L_RECORD
                cudaEventRecord(l_end);
                cudaEventSynchronize(l_end);
                cudaEventElapsedTime(&l_time, l_start, l_end);
                net->layers[net->index].l_time = l_time;
            #endif

            #if NVTX
                nvtxRangeEnd(id1);
            #endif

            net->index = k;
	}
	cond_i[th->arg->index_n]=0;
	pthread_cond_signal(&cond_t[th->arg->index_n]);
	pthread_mutex_unlock(&mutex_t[th->arg->index_n]);
}