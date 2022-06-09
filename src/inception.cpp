#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <thread>
#include <unistd.h>
#include "inception.h"

/*

event_idx : branch_num in inception (for recording event)
input_idx : the index of the input from the current layer
skip : Number of layer modules in one branch (How many more signals do thread have to send)
branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)

*/

namespace F = torch::nn::functional;
using namespace std;

void get_submodule_inception(torch::jit::script::Module module, Net &net){
    Layer t_layer;    
    Dummy temp;
    for(auto children : module.named_children()){
        if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){ //InceptionA
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.input_idx = PREV_IDX_7;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);    
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_7, PREV_IDX_5, PREV_IDX_2, CURRENT_IDX};
                }
                if(branch.name == "branch1x1"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_2, NEXT_IDX_5, NEXT_IDX_7};
                }
                else if(branch.name == "branch5x5_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_2;
                }
                else if(branch.name == "branch5x5_2"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_2, NEXT_IDX_3, NEXT_IDX_5};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = PREV_IDX_4;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_5, PREV_IDX_3, NEXT_IDX_2};
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                }
                t_layer.name = "A_" + branch.name;
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_8, PREV_IDX_6, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6a"){   //InceptionB
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch3x3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_3, NEXT_IDX_4};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_3, NEXT_IDX_1};
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "B_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = PREV_IDX_5;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "max_pool2d";
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {PREV_IDX_4, PREV_IDX_1, CURRENT_IDX};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_5, PREV_IDX_2, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){ //InceptionC
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = PREV_IDX_10;
                    t_layer.layer = temp;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_10, PREV_IDX_7, PREV_IDX_2, CURRENT_IDX};
                }
                else if(branch.name == "branch1x1"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_3, NEXT_IDX_8, NEXT_IDX_10};
                }
                else if(branch.name == "branch7x7_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch7x7_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_3, NEXT_IDX_5, NEXT_IDX_7};
                }
                else if(branch.name == "branch7x7dbl_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = PREV_IDX_5;
                    t_layer.skip = SKIP_IDX_5;
                }
                else if(branch.name == "branch7x7dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_8, PREV_IDX_5, NEXT_IDX_2};
                }
                else{
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.input_idx = CURRENT_IDX;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "C_" + branch.name;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.from_idx = {PREV_IDX_11, PREV_IDX_8, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7a"){   //InceptionD
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                t_layer.skip = SKIP_IDX_0;
                if(branch.name == "branch7x7x3_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = PREV_IDX_3;
                    t_layer.skip = SKIP_IDX_4;
                }
                else {
                    t_layer.input_idx = CURRENT_IDX;
                    if(branch.name == "branch3x3_1"){
                        t_layer.skip = SKIP_IDX_2;
                        t_layer.event_idx = ++event_idx;
                    }
                    else if(branch.name == "branch7x7x3_4"){
                        t_layer.branch_idx = {PREV_IDX_4, NEXT_IDX_1};
                    }
                    else if(branch.name == "branch3x3_2"){
                        t_layer.branch_idx = {NEXT_IDX_4, NEXT_IDX_5};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "D_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch7x7x3_4"){
                    t_layer.input_idx = PREV_IDX_7;
                    t_layer.layer = temp;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "max_pool2d";
                    t_layer.branch_idx = {PREV_IDX_5, PREV_IDX_1, CURRENT_IDX};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.from_idx = {PREV_IDX_6, PREV_IDX_2, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.skip = SKIP_IDX_0;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){    //InceptionE
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                t_layer.skip = SKIP_IDX_0;
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = PREV_IDX_11;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
	                t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);
                    t_layer.branch_idx = {PREV_IDX_11, PREV_IDX_7, PREV_IDX_2, CURRENT_IDX}; 
                    t_layer.input_idx = CURRENT_IDX;
                }
                else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    t_layer.input_idx = PREV_IDX_2;
                    if(branch.name == "branch3x3_1"){
	                    t_layer.skip = SKIP_IDX_4;
                        t_layer.event_idx = ++event_idx;
                    }
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.event_idx = ++event_idx;
	                t_layer.skip = SKIP_IDX_5;
                    t_layer.input_idx = PREV_IDX_6;
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    if(branch.name == "branch1x1"){
                        t_layer.skip = SKIP_IDX_1;
                        t_layer.event_idx = ++event_idx;
                        t_layer.branch_idx = {NEXT_IDX_4, NEXT_IDX_9, NEXT_IDX_11};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "E_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    if(branch.name == "branch3x3dbl_3b") t_layer.branch_idx = {PREV_IDX_9, PREV_IDX_5, NEXT_IDX_2}; 
                    else t_layer.branch_idx = {PREV_IDX_4, NEXT_IDX_5, NEXT_IDX_7}; 
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.from_idx = {PREV_IDX_2, PREV_IDX_1};
                    t_layer.layer = temp;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.exe_success = false;
                    t_layer.name = "concat";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.skip = SKIP_IDX_0;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_12, PREV_IDX_8, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.event_idx =INIT_EVENT_IDX;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "dropout"){
            continue;
        }
        else if(children.name != "AuxLogits")
        {   
            t_layer.input_idx = CURRENT_IDX;
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.layer = children.value;
            t_layer.skip = SKIP_IDX_0;
            t_layer.name = children.name;
            t_layer.exe_success = false;
            net.layers.push_back(t_layer);   
        }
    }
}


void *predict_inception_warming(Net *inception){
	{
		at::cuda::CUDAGuard guard(inception->device->g_device);
        Net *net = inception;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        // for(int iter=0;iter<ITERATION;iter++){
            int i;
            float l_sum = 0.0;
            float real_sum = 0.0;
            int round = 0;
            th_arg th;
            net->input=inputs2[inception->device->g_index];
            net->cur_round_last = 0;

            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[inception->index_n]);
                cond_i[inception->index_n] = 1;
                net->index = i;
                inception->layers[i].exe_success = false;
                net->all_api = 0;
                //multi_GPU 시 loss 비교 후 device[] 선택

                // net->device->total_weight += net->weight;
                pthread_mutex_lock(&mutex_g[net->device->g_index]);

                gpu_list[net->device->g_index].load += net->weight;
                net->layers[net->index].l_load = gpu_list[net->device->g_index].load; //왼쪽이 net->load 여도되지않나..?
                
                pthread_mutex_unlock(&mutex_g[net->device->g_index]);
                
                // net->timeslice = cal_timeslice(gpu_list[net->device->g_index].load,net->weight);
                // std::cout<<"ts : "<<net->timeslice<<std::endl;
                // std::cout<<"net : "<<net->index_n<<" , total_weight : "<<inception->device->total_weight<<" , my weight : "<<inception->weight<<" , timeslice : "<<net->timeslice<<std::endl<<std::endl;

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

                thpool_add_work(thpool[net->device->g_index],(void(*)(void *))forward_inception,&th);
                
                while (cond_i[inception->index_n] == 1)
                {
                    pthread_cond_wait(&cond_t[inception->index_n], &mutex_t[inception->index_n]);
                }

                l_sum = 0.0;
                real_sum = 0.0;
                round += 1;
                i = net->index;
                net->input.clear();
                net->input.push_back(net->layers[i].output);
                pthread_mutex_unlock(&mutex_t[inception->index_n]);
            }
        // }
        cudaStreamSynchronize(net->device->g_stream[(inception->index_s)%(n_streamPerPool)]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        
        std::cout << "\n*****"<<inception->name<<" "<<inception->index_n<<" result  "<<time<<"ms ***** \n";
        // std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void *predict_inception(std::vector<Net*> *vec_inception){
    {
        at::cuda::CUDAGuard guard(gpu_list[(*vec_inception)[gpu_idx[0]]->g_index].g_device);
        Net *inception = (*vec_inception)[(*vec_inception)[gpu_idx[0]]->g_index];
        Net *net = inception;
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
            // std::cout<<"------------- alex "<<net->index_n<<" ITER "<<iter<<"------------------\n";
            // net->input=inputs[inception->device->g_index];
            // std::cout<<"Net index : "<<net->index_n<<" , input gid : "<<inception->device->g_index<<std::endl;
            // std::cout<<"net index : "<<net->index_n<<"  /  current device : "<<(int)c10::cuda::current_device()<<std::endl; 
            for (i=0;i<net->layers.size();i++){
                pthread_mutex_lock(&mutex_t[inception->index_n]);
                cond_i[inception->index_n] = 1;
                net->index = i;
                net->all_api = 0;
                net->all_mem = 0.0;
                float cpy_mem = 0.0; 
                float q_pred = 0.0;
                at::Tensor prev_out;
                //at::Tensor prev_identity;
                // multi_GPU 시 loss 비교 후 device[] 선택
                // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                if(!(iter==0 && i ==0)){
                    // std::cout<< "Iter "<<iter<<" , round "<<round<<" i "<<i<<std::endl;
                    if(i!=0){
                        vv.clear();
                        if(i<=8 || i>121){   //i<=8   ||    i>121
                            
                            cpy_mem += net->layers[net->layers[i].l_prev].l_mem;
                            //prev_identity = net->identity;
                            vv.push_back(std::make_pair(((net->index)-1) , net->layers[(net->index)-1].output));

                        }
                        else if(i>8 && i<=15){       // 8 < i <= 15
                            for(int j=1;j<9;j++){
                                cpy_mem += net->layers[(net->index)-j].l_mem;
                                vv.push_back(std::make_pair(((net->index)-j) , net->layers[(net->index)-j].output));
                            }
                        }else{               //other
                            for(int j=1;j<13;j++){
                                cpy_mem += net->layers[(net->index)-j].l_mem;
                                vv.push_back(std::make_pair(((net->index)-j) , net->layers[(net->index)-j].output));
                            }
                            // std::cout<<"cpy : "<<cpy_mem<<std::endl;
                        }
                        net->all_mem = cpy_mem;
                            // std::cout<<"cpy : "<<cpy_mem<<std::endl;
                    }
                    int L_idx; //= get_lowest_load_idx();
                    int selec_q;// = select_queue(net,net->g_index,L_idx,cpy_mem);
                    q_pred = net->q_mean;
                    // std::cout<< "q_pred : "<<q_pred<<"\n\n";
                    //std::cout<<"Predict "<<net->index_n<<"  index : "<<net->index<<" , name : "<<net->layers[net->index].name<<" , change : "<<net->change_gid<<std::endl;
                    if(net->change_gid){    //GPU index 가 바뀌었을 경우 to
                        net->change_gid = false;
                        //at::Tensor prev_out= net->input[0].toTensor();
                        inception = (*vec_inception)[selec_q];
                        inception->g_index = selec_q;
                        // prev_out = prev_out.to(inception->device->g_device);
                        net=inception;
                        net->index = i;
                        
                        if(i!=0){
                            for(int j=0;j<vv.size();j++){
                                net->layers[vv[j].first].output = vv[j].second.to(net->device->g_device);   //-1 이 결국 input 
                                // std::cout<<"net index "<<net->index_n<<" copy vv : "<<vv[j].first<<"";
                            }
                            // std::cout<<net->index_n <<"copy "<<vv.size()<<" from "<< vv[0].first<<" to "<<vv[vv.size()-1].first<<std::endl;
                            net->input.clear();
                            net->input.push_back(net->layers[i-1].output);
                        }else{
                            net->input.clear();
                            net->input=inputs2[net->g_index];
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
                
                thpool_add_work(thpool[net->g_index],(void(*)(void *))forward_inception,&th);

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
                        
                                // std::cout<<"\n******* NET : "<<net->index_n<<" round : "<<round<<" weight : "<<net->weight<<" index : "<<i<<" my api : "<<net->all_api<< " q api :"<<net->q_all_api<<" load : "<<net->layers[i].l_load<<" ts : "<<net->timeslice<<" Q pred : "<<q_pred<<" Q time : "<<net->layers[i].q_time<<" l_sum : "<<l_sum<<" runtime :"<<net->layers[i].l_time<<std::endl;
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
                //int tmp = i;
                i = net->index;
                pthread_mutex_unlock(&mutex_t[net->index_n]);
            }
            net->index=0;
            round = 0;
            net->input=inputs2[net->g_index]; //필요?
    
            // cudaStreamSynchronize(net->device->g_stream[net->index_s%(n_streamPerPool)]);
            cudaStreamSynchronize(streams[net->g_index][net->index_s%(n_streamPerPool)]);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&time, start, end);
            cudaEventElapsedTime(&real, real_start, end);

            #if RECORD
                if(inception->warming == true){
                    // fprintf((fp_res),"Inception,%d,%d,%d,%d,%lf,%lf\n",inception->index_n,iter,net->nice,net->weight,time,real);
                }   
            #endif
            
            std::cout << "\n*****"<<inception->name<<" "<<inception->index_n<<" result  "<<time<<"ms ***** \n";
            //std::cout << (inception->layers[net->last].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
        }
    }
}

void forward_inception(th_arg *th){
	{
		at::cuda::CUDAStreamGuard guard(th->arg->device->g_stream[(th->arg->index_s)%(n_streamPerPool)]);
        pthread_mutex_lock(&mutex_t[th->arg->index_n]);
        // nice(th->arg->nice);

        #if L_RECORD
            cudaEvent_t l_start, l_end;
            float l_time;
            cudaEventCreate(&l_start);
            cudaEventCreate(&l_end);
            cudaEventRecord(l_start);
        #endif

        #if NVTX
            char str[30];
            sprintf(str, "Inception layer - %d", th->arg->index);
            nvtxRangeId_t id1 = nvtxRangeStartA(str);
        #endif

        Net *net = th->arg;
        int k = net->index;
        // int n_all = net->n_all;
        for(k;k<=net->cur_round_last;k++){
            std::vector<torch::jit::IValue> inputs;
            if(net->layers[k].input_idx != 0){
                inputs.push_back(net->layers[k + net->layers[k].input_idx].output);
            }
            else {
                // std::cout<<"here 0\n\n";
                inputs = net->input;
            }      
            at::Tensor out;
            if(net->layers[k].name == "concat"){
                std::vector<at::Tensor> cat_input;
                for(int i=0;i<net->layers[k].from_idx.size();i++){
                    cat_input.push_back(net->layers[k + net->layers[k].from_idx[i]].output);
                }
                out = torch::cat(cat_input, 1);
            }
            else if(net->layers[k].name == "avg_pool2d"){
                out = inputs[0].toTensor();            
                out = F::avg_pool2d(out,F::AvgPool2dFuncOptions(3).stride(1).padding(1));
            }
            else if(net->layers[k].name == "max_pool2d"){
                out = inputs[0].toTensor();            
                out = F::max_pool2d(out,F::MaxPool2dFuncOptions(3).stride(2));
            }
                        // else{
                        //     out = net->layers[k].layer.forward(inputs).toTensor();
                        // }
                        // net->layers[k].output = out;
                    // }
                    // k--;
                    // int record = net->layers[k].event_idx;
                    // cudaEventRecord(net->record[record], streams[th->arg->stream_id[(net->layers[k].event_idx)%4]]);
                // }
            //}
            // else if(net->layers[k].name == "concat"){  //brach out
            //     std::vector<at::Tensor> cat_input;
            //     for(int i=0;i<net->layers[k].from_idx.size();i++){
            //         cat_input.push_back(net->layers[k + net->layers[k].from_idx[i]].output);
            //     }
            //     out = torch::cat(cat_input, 1);
            // }
            else if(k == net->flatten){
                out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
                inputs.clear();
                inputs.push_back(out);
                out = net->layers[k].layer.forward(inputs).toTensor();
            }
            else{
                out = net->layers[k].layer.forward(inputs).toTensor();
            }
            net->layers[k].output = out;
            net->input.clear();
            net->input.push_back(net->layers[k].output);
        }k--;
        #if L_SYNC
            cudaStreamSynchronize(th->arg->device->g_stream[net->index_s%(n_streamPerPool)]); // 나중에 지워야함
        #endif
        // if(net->layers[k].event_idx >= 0){
        //     cudaEventSynchronize(net->record[net->layers[k].event_idx]);
        //     net->layers[k].output = out;
        //     net->layers[k].exe_success = true;
        // }


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
    
    cond_i[th->arg->index_n]=0;
    pthread_cond_signal(&cond_t[th->arg->index_n]);
    pthread_mutex_unlock(&mutex_t[th->arg->index_n]);		
    }
}

// #include "inception.h"

// /*

// event_idx : branch_num in inception (for recording event)
// input_idx : the index of the input from the current layer
// skip : Number of layer modules in one branch (How many more signals do thread have to send)
// branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)

// */

// namespace F = torch::nn::functional;
// using namespace std;

// void get_submodule_inception(torch::jit::script::Module module, Net &net){
//     Layer t_layer;    
//     Dummy temp;
//     for(auto children : module.named_children()){
//         if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){ //InceptionA
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch_pool"){
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.input_idx = -7;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "avg_pool2d";
//                     t_layer.skip = 2;
//                     net.layers.push_back(t_layer);    
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-7, -5, -2, 0};
//                 }
//                 if(branch.name == "branch1x1"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {2, 5, 7};
//                 }
//                 else if(branch.name == "branch5x5_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 2;
//                 }
//                 else if(branch.name == "branch5x5_2"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-2, 3, 5};
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.input_idx = -4;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-5, -3, 2};
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                 }
//                 t_layer.name = "A_" + branch.name;
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 net.layers.push_back(t_layer);
//             }
//             t_layer.event_idx = -1;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-8,-6,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_6a"){   //InceptionB
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch3x3"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {3, 4};
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-3, 1};
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "B_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch3x3dbl_3"){
//                     t_layer.input_idx = -5;
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "max_pool2d";
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {-4, -1, 0};
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.event_idx = -1;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-5,-2, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){ //InceptionC
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 if(branch.name == "branch_pool"){
//                     t_layer.input_idx = -10;
//                     t_layer.layer = temp;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.exe_success = false;
//                     t_layer.name = "avg_pool2d";
//                     t_layer.skip = 2;
//                     net.layers.push_back(t_layer);
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-10,-7,-2, 0};
//                 }
//                 else if(branch.name == "branch1x1"){
//                     t_layer.input_idx = 0;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 1;
//                     t_layer.branch_idx = {3,8,10};
//                 }
//                 else if(branch.name == "branch7x7_1"){
//                     t_layer.input_idx = -2;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.skip = 3;
//                 }
//                 else if(branch.name == "branch7x7_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-3,5,7};
//                 }
//                 else if(branch.name == "branch7x7dbl_1"){
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.input_idx = -5;
//                     t_layer.skip = 5;
//                 }
//                 else if(branch.name == "branch7x7dbl_3"){
//                     t_layer.input_idx = 0;
//                     t_layer.skip = 0;
//                     t_layer.branch_idx = {-8,-5,2};
//                 }
//                 else{
//                     t_layer.skip = 0;
//                     t_layer.input_idx = 0;
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "C_" + branch.name;
//                 net.layers.push_back(t_layer);
//             }
//             t_layer.event_idx = -1;
//             t_layer.from_idx = {-11,-8,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.name = "concat";
//             t_layer.skip = 0;
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_7a"){   //InceptionD
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 t_layer.skip = 0;
//                 if(branch.name == "branch7x7x3_1"){
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.input_idx = -3;
//                     t_layer.skip = 4;
//                 }
//                 else {
//                     t_layer.input_idx = 0;
//                     if(branch.name == "branch3x3_1"){
//                         t_layer.skip = 2;
//                         t_layer.event_idx = ++event_idx;
//                     }
//                     else if(branch.name == "branch7x7x3_4"){
//                         t_layer.branch_idx = {-4, 1};
//                     }
//                     else if(branch.name == "branch3x3_2"){
//                         t_layer.branch_idx = {4, 5};
//                     }
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "D_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch7x7x3_4"){
//                     t_layer.input_idx = -7;
//                     t_layer.layer = temp;
//                     t_layer.skip = 1;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.exe_success = false;
//                     t_layer.name = "max_pool2d";
//                     t_layer.branch_idx = {-5, -1, 0};
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.event_idx = -1;
//             t_layer.from_idx = {-6,-2, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.skip = 0;
//             t_layer.name = "concat";
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){    //InceptionE
//             int event_idx = -1;
//             for(auto branch : children.value.named_children()){
//                 t_layer.skip = 0;
//                 if(branch.name == "branch_pool"){
//                     t_layer.input_idx = -11;
//                     t_layer.layer = temp;
//                     t_layer.exe_success = false;
//                     t_layer.event_idx = ++event_idx;
//                     t_layer.name = "avg_pool2d";
// 	                t_layer.skip = 2;
//                     net.layers.push_back(t_layer);
//                     t_layer.branch_idx = {-11, -7, -2, 0}; 
//                     t_layer.input_idx = 0;
//                 }
//                 else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
//                     t_layer.input_idx = -2;
//                     if(branch.name == "branch3x3_1"){
// 	                    t_layer.skip = 4;
//                         t_layer.event_idx = ++event_idx;
//                     }
//                 }
//                 else if(branch.name == "branch3x3dbl_1"){
//                     t_layer.event_idx = ++event_idx;
// 	                t_layer.skip = 5;
//                     t_layer.input_idx = -6;
//                 }
//                 else{
//                     t_layer.input_idx = 0;
//                     if(branch.name == "branch1x1"){
//                         t_layer.skip = 1;
//                         t_layer.event_idx = ++event_idx;
//                         t_layer.branch_idx = {4, 9, 11};
//                     }
//                 }
//                 t_layer.layer = branch.value;
//                 t_layer.exe_success = false;
//                 t_layer.name = "E_" + branch.name;
//                 net.layers.push_back(t_layer);
//                 if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
//                     if(branch.name == "branch3x3dbl_3b") t_layer.branch_idx = {-9, -5, 2}; 
//                     else t_layer.branch_idx = {-4, 5, 7}; 
//                     t_layer.input_idx = 0;
//                     t_layer.from_idx = {-2, -1};
//                     t_layer.layer = temp;
//                     t_layer.skip = 0;
//                     t_layer.exe_success = false;
//                     t_layer.name = "concat";
//                     net.layers.push_back(t_layer);
//                 }
//             }
//             t_layer.skip = 0;
//             t_layer.input_idx = 0;
//             t_layer.from_idx = {-12,-8,-3, -1};
//             t_layer.layer = temp;
//             t_layer.exe_success = false;
//             t_layer.event_idx =-1;
//             t_layer.name = "concat";
//             net.layers.push_back(t_layer);
//             continue;
//         }
//         else if(children.name != "AuxLogits")
//         {   
//             t_layer.input_idx = 0;
//             t_layer.event_idx = -1;
//             t_layer.layer = children.value;
//             t_layer.skip = 0;
//             t_layer.name = children.name;
//             t_layer.exe_success = false;
//             net.layers.push_back(t_layer);   
//         }
//     }
// }


// void *predict_inception(Net *inception){
// 	int i;
//     float time;
//     cudaEvent_t start, end;
//     cudaEventCreate(&start);
//     cudaEventCreate(&end);
//     cudaEventRecord(start);

// 	for(i=0;i<inception->layers.size();i++){
// 		pthread_mutex_lock(&mutex_t[inception->index_n]);
// 		cond_i[inception->index_n] = 1;
// 		inception->layers[i].exe_success = false;

// 		netlayer nl;
// 		nl.net = inception;
// 		nl.net->index = i;

// 		th_arg th;
// 		th.arg = &nl;

// 		thpool_add_work(thpool,(void(*)(void *))forward_inception,&th);
		
//         while (cond_i[inception->index_n] == 1)
//     	{
//            	pthread_cond_wait(&cond_t[inception->index_n], &mutex_t[inception->index_n]);
//     	}
//         i = nl.net->index;
// 		inception->input.clear();
// 		inception->input.push_back(inception->layers[i].output);
// 		pthread_mutex_unlock(&mutex_t[inception->index_n]);
// 	}
//     cudaStreamSynchronize(streams[inception->index_s%(n_streamPerPool)]);
//     cudaEventRecord(end);
//     cudaEventSynchronize(end);
//     cudaEventElapsedTime(&time, start, end);
// 	std::cout << "\n*****"<<inception->name<<" result  "<<time/1000<<"s ***** \n";
// 	std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
// 	}

// void forward_inception(th_arg *th){
// 	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
// 	netlayer *nl = th->arg;
// 	int k = nl->net->index;
//     int n_all = nl->net->n_all;
//     std::vector<torch::jit::IValue> inputs;
//     //std::vector<int> stream_id = {(nl->net->index_n%(n_streamPerPool-n_Branch)), n_streamPerPool-1, n_streamPerPool-2,n_streamPerPool-3};
//     std::vector<int> stream_id = {(nl->net->index_s)%n_streamPerPool, abs(nl->net->index_b)%n_streamPerPool, abs((nl->net->index_b)-1)%n_streamPerPool, abs((nl->net->index_b)-2)%n_streamPerPool};
//     if(nl->net->layers[k].input_idx != 0){
//         inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
//     }
//     else {
//         inputs = nl->net->input;
//     }
//     if(nl->net->layers[k + nl->net->layers[k].skip].skip > 0){ // +1 why? predict for loop 
//         nl->net->index = k + nl->net->layers[k].skip - 1;
//         cond_i[nl->net->index_n]=0;
// 		pthread_cond_signal(&cond_t[nl->net->index_n]);
// 	}
// 	pthread_mutex_unlock(&mutex_t[nl->net->index_n]); 
// 	at::Tensor out;
//     {
//         at::cuda::CUDAStreamGuard guard(streams[stream_id[0]]);
//         if(k == nl->net->flatten){
//             out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
//             inputs.clear();
//             inputs.push_back(out);
//             out = nl->net->layers[k].layer.forward(inputs).toTensor();
//         }
//         else if(nl->net->layers[k].skip > 0){   //branch
//             {
//                 at::cuda::CUDAStreamGuard guard(streams[stream_id[(nl->net->layers[k].event_idx)%4]]); //event_idx == branch_num
//                 out = inputs[0].toTensor();
//                 int T = nl->net->layers[k].skip;
//                 for(int t=0;t<T;k++,t++){
//                     if(nl->net->layers[k].input_idx != 0){
//                         inputs.clear();
//                         inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
//                     }
//                     else {
//                         inputs.clear();
//                         inputs.push_back(out);
//                     } 
                    
//                     if(nl->net->layers[k].name == "concat"){
//                         std::vector<at::Tensor> cat_input;
//                         for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
//                             cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
//                         }
//                         out = torch::cat(cat_input, 1);
//                     }
//                     else if(nl->net->layers[k].name == "avg_pool2d"){
//                         out = F::avg_pool2d(out,F::AvgPool2dFuncOptions(3).stride(1).padding(1));
//                     }
//                     else if(nl->net->layers[k].name == "max_pool2d"){
//                         out = F::max_pool2d(out,F::MaxPool2dFuncOptions(3).stride(2));
//                     }
//                     else{
//                         out = nl->net->layers[k].layer.forward(inputs).toTensor();
//                     }
//                     nl->net->layers[k].output = out;
//                 }
//                 k--;
//                 int record = nl->net->layers[k].event_idx;
//                 cudaEventRecord(nl->net->record[record], streams[stream_id[(nl->net->layers[k].event_idx)%4]]);
//             }
//         }
//         else if(nl->net->layers[k].name == "concat"){  //brach out
//             std::vector<at::Tensor> cat_input;
//             for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
//                 cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
//             }
//             out = torch::cat(cat_input, 1);
//         }
//         else{
//             out = nl->net->layers[k].layer.forward(inputs).toTensor();
//         }
//     }
//     if(nl->net->layers[k].event_idx >= 0){
// 		cudaEventSynchronize(nl->net->record[nl->net->layers[k].event_idx]);
// 		nl->net->layers[k].output = out;
// 		nl->net->layers[k].exe_success = true;
// 	}
//     nl->net->layers[k].output = out;

//     pthread_mutex_lock(&mutex_t[nl->net->index_n]);

//     if(nl->net->layers[k].exe_success == false){
//         cond_i[nl->net->index_n]=0;
//         pthread_cond_signal(&cond_t[nl->net->index_n]);
//     }
//     else{
//        for(int i=0;i<nl->net->layers[k].branch_idx.size();i++){
//            if(nl->net->layers[k + nl->net->layers[k].branch_idx[i]].exe_success == false){
//                pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
//                return;
//            }
//        }
//        for(int i=0;i<nl->net->layers[k].branch_idx.size();i++){ //complete
//            nl->net->layers[k + nl->net->layers[k].branch_idx[i]].exe_success = false;
//        }
//        nl->net->layers[k].exe_success = false;
//        nl->net->index = k + nl->net->layers[k].branch_idx.back(); // last layer index of branch
//        cond_i[nl->net->index_n]=0;
//        pthread_cond_signal(&cond_t[nl->net->index_n]);
//     }
// 	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
// }

