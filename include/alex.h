#ifndef ALEX_H
#define ALEX_H

#include "net.h"
#include "test.h"
#include "thpool.h"
#include "balancecheck.h"

void get_submodule_alexnet(torch::jit::script::Module module, Net &child);
void *predict_alexnet_warming(Net *input);
void *predict_alexnet(std::vector<Net*> *vec_res);
void forward_alexnet(th_arg *th);
#endif

