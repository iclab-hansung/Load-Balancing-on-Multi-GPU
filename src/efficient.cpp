#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "efficient.h"

namespace F = torch::nn::functional;

// void get_submodule_efficientnet(torch::jit::script::Module module,Net &net){
//     Layer t_layer;
//     Dummy residual;
// 	for(auto children : module.named_children()){
//         if(children.value.children().size() == 0){     //avgpool
//             t_layer.layer = children.value;
//             t_layer.name = "avgpool";
//             net.layers.push_back(t_layer);
//         }
//         else{   //children.name: "features", "classifier"
//             if(children.name == "features"){
//                 for(auto child : children.value.named_children()){  //child.name: 0,1,..,8
//                     if(child.name == "0" || child.name == "8"){
//                         t_layer.layer = child.value;
//                         t_layer.name = "ConvBnSiLU";
//                         net.layers.push_back(t_layer);
//                     }
//                     else{
//                         for(auto MBConv : child.value.named_children()){     //MBConv: 0,1,..
//                             for(auto block : MBConv.value.named_children()){    //block: "block", "stochastic_depth"
//                                 if(block.name == "block"){
//                                     if(child.name == "1"){
//                                         for(auto in_block : block.value.named_children()){
//                                             t_layer.layer = in_block.value;
//                                             if(in_block.name == "0"){
//                                                 // t_layer.layer = in_block.value;
//                                                 t_layer.name = "ConvBnSiLU";
//                                                 net.layers.push_back(t_layer);
//                                             }
//                                             else if(in_block.name == "1"){
//                                                 for(auto layer : in_block.value.named_children()){
//                                                     t_layer.layer = layer.value;
//                                                     if(layer.name == "avgpool")     t_layer.name = "avgpool";
//                                                     else if(layer.name == "fc1")    t_layer.name = "ConvSiLU";
//                                                     else if(layer.name == "fc2")    t_layer.name = "ConvSigmoid";
//                                                     else if(layer.name == "activation" || layer.name == "scale_activation") continue;
//                                                     net.layers.push_back(t_layer);
//                                                 }
//                                             }
//                                             else if(in_block.name == "2"){
//                                                 // t_layer.layer = in_block.value;
//                                                 t_layer.name = "ConvBn";
//                                                 net.layers.push_back(t_layer);
//                                             }
//                                         }
//                                         if(MBConv.name != "0"){
//                                             t_layer.layer = residual;
//                                             t_layer.name = "Residual";
//                                             t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS_1};
//                                             net.layers.push_back(t_layer);
//                                         }
//                                     }
//                                     else{   //child.name: 2,3,..,7
//                                         for(auto in_block : block.value.named_children()){
//                                             t_layer.layer = in_block.value;
//                                             if(in_block.name == "0" || in_block.name == "1"){
//                                                 t_layer.name = "ConvBnSiLU";
//                                                 net.layers.push_back(t_layer);
//                                             }
//                                             else if(in_block.name == "2"){
//                                                 for(auto layer : in_block.value.named_children()){
//                                                     t_layer.layer = layer.value;
//                                                     if(layer.name == "avgpool")     t_layer.name = "avgpool";
//                                                     else if(layer.name == "fc1")    t_layer.name = "ConvSiLU";
//                                                     else if(layer.name == "fc2")    t_layer.name = "ConvSigmoid";
//                                                     else if(layer.name == "activation" || layer.name == "scale_activation") continue;
//                                                     net.layers.push_back(t_layer);
//                                                 }
//                                             }
//                                             else if(in_block.name == "3"){
//                                                 t_layer.name = "ConvBn";
//                                                 net.layers.push_back(t_layer);
//                                             }
//                                         }
//                                         if(MBConv.name != "0"){
//                                             t_layer.layer = residual;
//                                             t_layer.name = "Residual";
//                                             t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS};
//                                             net.layers.push_back(t_layer);
//                                         }
//                                     }
//                                 }else if(block.name == "stochastic_depth")   continue;
//                             }
//                         }
//                     }
//                 }
//             }
//             else{   //children.name: "classifier"
//                 for(auto child : children.value.named_children()){
//                     if(child.name == "0")   continue;   //dropout
//                     t_layer.layer = child.value;
// 	    			t_layer.name = "linear";
// 	    			net.layers.push_back(t_layer); 
//                 }
//             }
//         }
//     }
// }

void get_submodule_efficientnet(torch::jit::script::Module module,Net &net){
    Layer t_layer;
    Dummy residual;
	for(auto children : module.named_children()){
        if(children.value.children().size() == 0){     //avgpool
            t_layer.layer = children.value;
            t_layer.from_idx = {-1};
            t_layer.name = "avgpool";
            net.layers.push_back(t_layer);
        }
        else{   //children.name: "features", "classifier"
            if(children.name == "features"){
                for(auto child : children.value.named_children()){  //child.name: 0,1,..,8
                    if(child.name == "0" || child.name == "8"){
                        t_layer.layer = child.value;
                        t_layer.from_idx = {-1};
                        t_layer.name = "ConvBnSiLU";
                        net.layers.push_back(t_layer);
                    }
                    else{
                        for(auto MBConv : child.value.named_children()){     //MBConv: 0,1,..
                            for(auto block : MBConv.value.named_children()){    //block: "block", "stochastic_depth"
                                if(block.name == "block"){
                                    if(child.name == "1"){
                                        for(auto in_block : block.value.named_children()){
                                            t_layer.layer = in_block.value;
                                            if(in_block.name == "0"){
                                                t_layer.name = "ConvBnSiLU";
                                                t_layer.from_idx = {-1};
                                                net.layers.push_back(t_layer);
                                            }
                                            else if(in_block.name == "1"){
                                                for(auto layer : in_block.value.named_children()){
                                                    t_layer.layer = layer.value;
                                                    if(layer.name == "avgpool"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -2};
                                                        t_layer.name = "avgpool";
                                                    }
                                                    else if(layer.name == "fc1"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -3};
                                                        t_layer.name = "ConvSiLU";
                                                    }
                                                    else if(layer.name == "fc2"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -4};
                                                        t_layer.name = "ConvSigmoid";
                                                    }
                                                    else if(layer.name == "activation" || layer.name == "scale_activation") continue;
                                                    net.layers.push_back(t_layer);
                                                }
                                            }
                                            else if(in_block.name == "2"){
                                                if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -5};
                                                t_layer.name = "ConvBn";
                                                net.layers.push_back(t_layer);
                                            }
                                        }
                                        if(MBConv.name != "0"){
                                            t_layer.layer = residual;
                                            t_layer.name = "Residual";
                                            t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS_1};
                                            net.layers.push_back(t_layer);
                                        }
                                    }
                                    else{   //child.name: 2,3,..,7
                                        for(auto in_block : block.value.named_children()){
                                            t_layer.layer = in_block.value;
                                            if(in_block.name == "0" || in_block.name == "1"){
                                                if(in_block.name == "0" || MBConv.name == "0")    t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -2};
                                                t_layer.name = "ConvBnSiLU";
                                                net.layers.push_back(t_layer);
                                            }
                                            else if(in_block.name == "2"){
                                                for(auto layer : in_block.value.named_children()){
                                                    t_layer.layer = layer.value;
                                                    if(layer.name == "avgpool"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -3};
                                                        t_layer.name = "avgpool";
                                                    }
                                                    else if(layer.name == "fc1"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -4};
                                                        t_layer.name = "ConvSiLU";
                                                    }
                                                    else if(layer.name == "fc2"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -5};
                                                        t_layer.name = "ConvSigmoid";
                                                    }
                                                    else if(layer.name == "activation" || layer.name == "scale_activation") continue;
                                                    net.layers.push_back(t_layer);
                                                }
                                            }
                                            else if(in_block.name == "3"){
                                                if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -6};
                                                t_layer.name = "ConvBn";
                                                net.layers.push_back(t_layer);
                                            }
                                        }
                                        if(MBConv.name != "0"){
                                            t_layer.layer = residual;
                                            t_layer.name = "Residual";
                                            t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS};
                                            net.layers.push_back(t_layer);
                                        }
                                    }
                                }else if(block.name == "stochastic_depth")   continue;
                            }
                        }
                    }
                }
            }
            else{   //children.name: "classifier"
                for(auto child : children.value.named_children()){
                    if(child.name == "0")   continue;   //dropout
                    t_layer.layer = child.value;
                    t_layer.from_idx = {-1};
	    			t_layer.name = "linear";
	    			net.layers.push_back(t_layer); 
                }
            }
        }
    }
}

void *predict_efficientnet_warming(Net *efficientnet){
	{
		at::cuda::CUDAGuard guard(efficientnet->device->g_device);
        Net *net = efficientnet;

		int i;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
        // for(int iter=0;iter<500;iter++){
        //     efficientnet->input = inputs;
        // for(int iter=0;iter<ITERATION;iter++){
        
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            th_arg th;
            net->input=inputs3[net->device->g_index];
            net->cur_round_last = 0;

            for(i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[net->index_n]);
                cond_i[net->index_n] = 1;

                net->index = i;
                net->all_api = 0;
                
                //필요해?
                pthread_mutex_lock(&mutex_g[net->device->g_index]);

                gpu_list[net->device->g_index].load += net->weight;
                // std::cout<<"load : "<<gpu_list[net->device->g_index].load<<" , weight : "<<net->weight<<std::endl;
                net->layers[net->index].l_load = gpu_list[net->device->g_index].load; //왼쪽이 net->load 여도되지않나..?
                
                pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                
                // net->timeslice = cal_timeslice(gpu_list[net->device->g_index].load,net->weight);
                // std::cout<<"net : "<<net->index_n<<" , total_weight : "<<alex->device->total_weight<<" , my weight : "<<alex->weight<<" , timeslice : "<<net->timeslice<<std::endl<<std::endl;
                net->q_all_api = gpu_list[net->device->g_index].all_api;
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
                // cal_kernels_enqueue(efficientnet);

                #if RECORD
                    cudaEvent_t exe_start, exe_end,q_end;
                    float exe_time,q_time;
                    cudaEventCreate(&q_end);
                    cudaEventCreate(&exe_start);
                    cudaEventCreate(&exe_end);
                    cudaEventRecord(exe_start);
                #endif

                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_efficientnet,&th);
                
                while (cond_i[efficientnet->index_n] == 1)
                {
                    pthread_cond_wait(&cond_t[efficientnet->index_n], &mutex_t[efficientnet->index_n]);
                }

                // cal_kernels_dequeue(efficientnet,i);

                #if RECORD
                    if(net->warming==true){
                        real_sum += net->layers[i].l_time;
                        #if Q_OVERHEAD
                            //fprintf((net->fp),"%d,%lf\n",i,eq_time);
                            //std::cout<<"write index : "<<i<<"   "<<net->layers[i].q_time<<std::endl;
                            #if L_RECORD                                    
                                // fprintf((net->fp),"%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf\n",round,net->weight,i,net->all_api,net->q_all_api,net->layers[i].l_load,net->timeslice,net->layers[i].q_time,l_sum,net->layers[i].l_time);
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
        // }
        cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);

        #if RECORD
            if(efficientnet->warming == true)   fprintf((net->fp),"%d,%lf\n",net->index_n,time);
        #endif

        std::cout << "\n*****"<<efficientnet->name<<" "<<efficientnet->index_n<<" result  "<<time<<"ms ***** \n";
		// std::cout << (efficientnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	// }
    }
}

void *predict_efficientnet(std::vector<Net*> *vec_efficient){
    {
        at::cuda::CUDAGuard guard((*vec_efficient)[(*vec_efficient)[gpu_idx[0]]->g_index]->device->g_device);
        Net *efficientnet = (*vec_efficient)[(*vec_efficient)[gpu_idx[0]]->g_index];
        Net *net = efficientnet;
        // std::cout<<"g_index : "<<net->g_index<<" , "<<net->device->g_device<<std::endl;
        // nice(net->nice);
        float time,real;
        cudaEvent_t start, end,real_start;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventCreate(&real_start);

        int iter;

        cudaEventRecord(real_start);
        
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
            // std::cout<<"------------- efficient "<<net->index_n<<" ITER "<<iter<<"------------------\n";
            // net->input=inputs[efficientnet->device->g_index];
            // std::cout<<"Net index : "<<net->index_n<<" , input gid : "<<efficientnet->device->g_index<<std::endl;
            // std::cout<<"net index : "<<net->index_n<<"  /  current device : "<<(int)c10::cuda::current_device()<<std::endl; 
            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[efficientnet->index_n]);
                cond_i[efficientnet->index_n] = 1;
                net->index = i;
                net->all_api = 0;
                float cpy_mem = 0.0; 
                float q_pred = 0.0;
                at::Tensor prev_identity;
                // multi_GPU 시 loss 비교 후 device[] 선택
                // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                if(!(iter == 0 && i == 0)){
                    if(i != 0){
                        vv.clear();
                        if(i!=1 && net->layers[i].name != "avgpool"){
                            prev_identity = net->identity;
                            cpy_mem += net->layers[i-1].l_identity;
                        }
                        //if(net->layers[i].name == "Residual" ){
                            for(int j=0;j<net->layers[net->index].from_idx.size();j++){
                                // std::cout<<(net->index) + net->layers[(net->index)].from_idx[j]<<std::endl;
                                cpy_mem += net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].l_mem;
                                vv.push_back(std::make_pair((net->index) + net->layers[(net->index)].from_idx[j],net->layers[(net->index) + net->layers[(net->index)].from_idx[j]].output));
                            }
                        //}else{  //나중에 합치자.
                            //cpy_mem += net->layers[(net->index-1)].l_mem;
                            // std::cout<<cpy_mem<<std::endl;
                            //vv.push_back(std::make_pair((net->index-1),net->layers[(net->index-1)].output));
                            // std::cout<<"h÷ere?222\n";
                        //}
                    } 
                    int L_idx; //= get_lowest_load_idx();
                    int selec_q;// = select_queue(net,net->g_index,L_idx,cpy_mem);
                    q_pred = net->q_mean;
                    // std::cout<< "q_pred : "<<q_pred<<"\n\n";
                    //std::cout<<"Predict "<<net->index_n<<"  index : "<<net->index<<" , name : "<<net->layers[net->index].name<<" , change : "<<net->change_gid<<std::endl;
                    if(net->change_gid){    //GPU index 가 바뀌었을 경우 to
                        net->change_gid = false;
                        efficientnet = (*vec_efficient)[selec_q];
                        efficientnet->g_index = selec_q;

                        net=efficientnet;
                        net->index = i;

                        if(i!=0){
                            if(i!=1 && net->layers[i].name != "avgpool"){
                                net->identity = prev_identity.to(net->device->g_device);
                            }
                            for(int j=0;j<vv.size();j++){
                                // std::cout<<" vv first : "<<vv[j].first<<std::endl;
                                net->layers[vv[j].first].output = vv[j].second.to(net->device->g_device);   //-1 이 결국 input 
                            }
                            net->input.clear();
                            net->input.push_back(net->layers[i-1].output);
                        }else{
                             net->input.clear();
                            net->input = inputs3[net->device->g_index];
                        }
                       
                        // std::cout<<"weight : "<<net->weight<<std::endl;
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
                
                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_efficientnet,&th);

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
            net->input=inputs3[efficientnet->g_index];
    
            cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
            cudaEventRecord(end);
            // if(net->warming == true){
            //     pthread_mutex_lock(&mutex_g[net->device->g_index]);
                
            //     net->device->n_net -= 1;

            //     pthread_mutex_unlock(&mutex_g[net->device->g_index]);
            // }
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, start, end);
            cudaEventElapsedTime(&real, real_start, end);

            #if RECORD
                if(efficientnet->warming == true){
                    // fprintf((fp_res),"EfficientNet,%d,%d,%d,%d,%lf,%lf\n",efficientnet->index_n,iter,net->nice,net->weight,time,real);
                }   
            #endif
            
            std::cout << "\n*****"<<efficientnet->name<<" "<<efficientnet->index_n<<" result  "<<time<<"ms ***** \n";
            // std::cout << (efficientnet->layers[net->last].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
        }
    }
}

void forward_efficientnet(th_arg *th){
    	{
		at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[(th->arg->index_s)%(n_streamPerPool)]);
		pthread_mutex_lock(&mutex_t[th->arg->index_n]);

        #if L_RECORD
            cudaEvent_t l_start, l_end;
            float l_time;
            cudaEventCreate(&l_start);
            cudaEventCreate(&l_end);
            cudaEventRecord(l_start);
        #endif

		#if NVTX
            char str[30];
            sprintf(str, "Efficient layer - %d", th->arg->index);
            nvtxRangeId_t id1 = nvtxRangeStartA(str);
        #endif
		Net *net = th->arg;
        int k = net->index;

        for(k;k<=net->cur_round_last;k++){
            std::vector<torch::jit::IValue> inputs = net->input;
            at::Tensor out;
            // std::cout<<"*In JOB "<<net->index_n<<"  index : "<<k<<" , name : "<<net->layers[k].name<<" , tensor device : "<<inputs[0].toTensor().device()<<"*\n";

            if(net->layers[k].name == "avgpool"){
                net->identity = inputs[0].toTensor();
                // out = net->layers[k].layer.forward(inputs).toTensor();

                // if(k+1 < net->layers.size()){
                //     net->layers[k].output = out;
                //     k++;
                //     inputs.clear();
                //     inputs.push_back(out);
                // }
            }
            if(net->layers[k].name == "Residual"){
                int add_index = k + net->layers[k].from_idx[0];
                out = net->layers[add_index].output;
                for(int i=1;i<net->layers[k].from_idx.size();i++){
                    int add_index = k + net->layers[k].from_idx[i];
                    out += net->layers[add_index].output;
                }
            }
            else if(k==net->flatten){	//flatten
                out = inputs[0].toTensor().view({net->layers[k-1].output.size(0), -1});
                inputs.clear();
                inputs.push_back(out);
                out = net->layers[k].layer.forward(inputs).toTensor();
            }
            else{
                out = net->layers[k].layer.forward(inputs).toTensor();
                if(net->layers[k].name == "ConvSiLU"){
                    out = F::silu(out);
                }
                else if(net->layers[k].name == "ConvSigmoid"){
                    out = torch::sigmoid(out);
                    out = net->identity * out;
                }
            }
            net->layers[k].output = out;    //필요?
            net->input.clear();
            net->input.push_back(net->layers[k].output);
        }k--;
		

		#if L_SYNC
            cudaStreamSynchronize(th->arg->device->g_stream[(th->arg->index_s)%(n_streamPerPool)]); // 나중에 지워야함
        #endif

		// net->layers[k].output = out;

		#if NVTX
            nvtxRangeEnd(id1);
        #endif

        #if L_RECORD
			cudaEventRecord(l_end);
			cudaEventSynchronize(l_end);
			cudaEventElapsedTime(&l_time, l_start, l_end);
            net->layers[net->index].l_time = l_time;
		#endif
		// cudaEventRecord(end);
		// cudaEventSynchronize(end);
		// cudaEventElapsedTime(&l_time, start, end);
		//fprintf((net->fp),"%d,%lf\n",net->index,l_time/1000);
		net->index = k;
		cond_i[net->index_n]=0;
        pthread_cond_signal(&cond_t[net->index_n]);
		pthread_mutex_unlock(&mutex_t[net->index_n]);		
	}
}