#ifndef BALANCECHECK_H
#define BALANCECHECK_H

#include "test.h"

void* check_loadbalancing();
void cal_kernels_enqueue(Net *net);
void cal_kernels_dequeue(Net *net,int index);
float find_time_value(int api_num);
void select_queue(Net *net, int my_gid, int L_gid, float mem_size);
int get_lowest_load_idx(Net *net);

#endif
