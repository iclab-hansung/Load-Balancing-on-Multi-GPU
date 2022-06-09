#ifndef RESNET18_H
#define RESNET18_H

#include "test.h"
#include "balancecheck.h"

void get_submodule_resnet(torch::jit::script::Module module, Net &net);
void *predict_resnet(std::vector<Net*> *vec_res);
void *predict_resnet_warming(Net *input);
void forward_resnet(th_arg *th);

#endif
