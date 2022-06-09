
#include "vgg.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_vgg(torch::jit::script::Module module, Net &net){
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
					if(ch.name == "0" || ch.name == "2" || ch.name == "5" || ch.name == "7" || ch.name == "10" || ch.name == "12" || ch.name == "14" ||
						ch.name == "17" || ch.name == "19" || ch.name == "21" || ch.name == "24" || ch.name == "26" || ch.name == "28"){
							t_layer.name = "conv";
					}
					else if(ch.name == "1" || ch.name == "3" || ch.name == "6" || ch.name == "8" || ch.name == "11" || ch.name == "13" || ch.name == "15" ||
							ch.name == "18" || ch.name == "20" || ch.name == "22" || ch.name == "25" || ch.name == "27" || ch.name == "29"){
								t_layer.name = "relu";
					}
					else if(ch.name == "4" || ch.name == "9" || ch.name == "16" || ch.name == "23" || ch.name == "30"){
						t_layer.name = "maxpool";
					}
					net.layers.push_back(t_layer);
				}
				else if(child.name == "classifier"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" || ch.name == "6"){
						t_layer.name = "linear";
					}
					else if(ch.name == "1" || ch.name == "4"){
						t_layer.name = "relu";
					}
					else if(ch.name == "2" || ch.name == "5" ){	//dropout
						continue;
					}
					net.layers.push_back(t_layer);
				}
			}
		}
	}
}

void *predict_vgg(Net *vgg){
    {   
        at::cuda::CUDAGuard guard(vgg->device->g_device);
        int i;
        Net *net = vgg;
        th_arg th;
        cudaEvent_t start, end;
        float time; //eventrecord
        struct timespec start_ts,end_ts;
        float l_sum = 0.0;
        float real_sum = 0.0;
        int round = 0;
        net->cur_round_last = 0;
        net->next_round_last = 0;
        
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

         #if RECORD
			cudaEvent_t q_end;
			float q_time;
            cudaEventCreate(&q_end);
		#endif

        for (i=0;i<net->layers.size();i++){
            pthread_mutex_lock(&mutex_t[vgg->index_n]);
            cond_i[vgg->index_n] = 1;
            net->index = i;
            //net->next_round = false;
            net->all_api = 0;

            //multi_GPU 시 loss 비교 후 device[] 선택

            net->device->total_weight += net->weight;
            net->timeslice = cal_timeslice(vgg->device->total_weight,net->weight);
            // std::cout<<"net : "<<net->index_n<<" , total_weight : "<<vgg->device->total_weight<<" , my weight : "<<vgg->weight<<" , timeslice : "<<net->timeslice<<std::endl<<std::endl;

            int j=i;
            do{
                l_sum += net->layers[j].l_mean;
                net->all_api += net->layers[j].l_api;
                if(l_sum > net->timeslice){
                    l_sum -= net->layers[j].l_mean;
                    net->cur_round_last = net->layers[j].l_prev;
                    net->all_api -= net->layers[j].l_api;
                    break;
                }else if(l_sum == net->timeslice){
                    net->cur_round_last = j;
                    break;
                }else if(j == net->last){
                    net->cur_round_last = j;
                    break;
                }
                j=net->layers[j].l_next;
            }while(j<net->layers.size());
            
            // std::cout<<net->index_n<<" this round :"<<round<<" , timeslice : "<<net->timeslice<<" , cur index : "<<i<<" , cur_round_last : "<<net->cur_round_last<<std::endl<<std::endl;
            // net->layers[net->cur_round_last].round_last = true; //이번 round 가 끝났는지 확인을 위한 flag 필요??????
            th.arg = net;

            thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_vgg,&th);

            while (cond_i[net->index_n] == 1){
                // std::cout<<"layer signal wait\n\n";
                pthread_cond_wait(&cond_t[net->index_n], &mutex_t[net->index_n]);
            }
            net->device->total_weight -= net->weight;
            #if RECORD
                if(net->warming==true){
                    real_sum += net->layers[i].l_time;
                    #if Q_OVERHEAD
                        //fprintf((net->fp),"%d,%lf\n",i,eq_time);
                        //std::cout<<"write index : "<<i<<"   "<<net->layers[i].q_time<<std::endl;
                        #if L_RECORD                                    
                            fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf\n",round,net->weight,i,net->all_api,net->q_all_api,net->layers[i].l_load,net->timeslice,net->layers[i].q_time,l_sum,net->layers[i].l_time);
                        #else
                            fprintf((net->fp),"%d,%d,%lf\n",i,net->layers[i].all_api,net->layers[i].q_time);
                        #endif
                    #else
                        fprintf((net->fp),"%d,%d,%lf\n",i,net->layers[i].all_api);
                    #endif
                }
            #endif

            l_sum = 0.0;
            real_sum = 0.0;
            round += 1;
            i = net->index;
            net->input.clear();
            net->input.push_back(net->layers[i].output);
            pthread_mutex_unlock(&mutex_t[net->index_n]);
        }
        cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);

        if(net->warming == true){
            pthread_mutex_lock(&mutex_g[net->device->g_index]);
            
            net->device->n_net -= 1;

            pthread_mutex_unlock(&mutex_g[net->device->g_index]);
        }

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        
        std::cout << "\n*****"<<vgg->name<<" "<<vgg->index_n<<" result  "<<time<<"ms ***** \n";
    }
}


void forward_vgg(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->index_n]);
	{
		at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[(th->arg->index_n)%(n_streamPerPool)]);
		
		#if L_RECORD
			cudaEvent_t l_start, l_end;
			float l_time;
			cudaEventCreate(&l_start);
			cudaEventCreate(&l_end);
			cudaEventRecord(l_start);
		#endif

		#if NVTX
		char str[30];
		sprintf(str, "VGG layer - %d", th->arg->index);
		nvtxRangeId_t id1 = nvtxRangeStartA(str);
		#endif
		
		Net *net = th->arg;
		int k = net->index;
        for(k;k<=net->cur_round_last;k++){

            std::vector<torch::jit::IValue> inputs = net->input;
            at::Tensor out;
        
            if(k == net->flatten){
                out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
                inputs.clear();
                inputs.push_back(out);
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
	}
	cond_i[th->arg->index_n]=0;
	pthread_cond_signal(&cond_t[th->arg->index_n]);
	pthread_mutex_unlock(&mutex_t[th->arg->index_n]);		
}
