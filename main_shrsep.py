import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import time
import os
import shutil

import copy
import datetime

from layers import ConvNet
import network_operators
import utils



# hyperparameters
mutation_time_limit_hours = 23

n_models = 8  # number of child models per generation

n_mutations = 5  # number of mutations applied per generation

budgets = 5  # budget for training all models combined (in epochs)

n_epochs_between = 10  # epochs for warm restart of learning rate

epoch_final = 200  # epochs for final training
lr_final = 0.025
n_experiments = 8
max_n_params = 20*10**6
expfolder = "./results_shrsep/"
shutil.rmtree('./results_shrsep', ignore_errors=True)
os.makedirs(expfolder)


# data
trainloader, validloader, testloader = utils.prepare_data(batch_size=128, valid_frac=0.1)

# one batch which will use for many computations
for batch_idx, (inputs, targets) in enumerate(trainloader):
    data, target = Variable(inputs.cuda()), Variable(targets.cuda())
    batch = data
    batch_y = target
    break


# basic data structure
layer0 = {'type': 'input', 'params': {'shape': (32,32,3)}, 'input': [-1],'id': 0}
layer1 = {'type': 'conv', 'params': {'channels': 64, 'ks1': 3, 'ks2': 3, "in_channels": 3}, 'input': [0], 'id': 1}
layer1_1 = {'type': 'batchnorm', 'params': {"in_channels": 64}, 'input': [1], 'id': 2}
layer1_2 = {'type': 'activation', 'params': {}, 'input': [2], 'id': 3}
layer4 = {'type': 'pool', 'params': {'pooltype': 'max', 'poolsize': 2}, 'input': [3],'id': 10}
layer5 = {'type': 'conv', 'params': {'channels': 128, 'ks1': 3, 'ks2': 3, "in_channels": 64}, 'input': [10], 'id': 11}
layer5_1 = {'type': 'batchnorm', 'params': {"in_channels": 128}, 'input': [11], 'id': 12}
layer5_2 = {'type': 'activation', 'params': {}, 'input' : [12], 'id': 13}
layer8 = {'type': 'pool', 'params': {'pooltype': 'max', 'poolsize': 2}, 'input': [13],'id': 20}
layer9 = {'type': 'conv', 'params': {'channels': 256, 'ks1': 3, 'ks2': 3, "in_channels": 128}, 'input': [20], 'id': 21}
layer9_1 = {'type': 'batchnorm', 'params': {"in_channels": 256}, 'input': [21], 'id': 22}
layer9_2 = {'type': 'activation', 'params': {}, 'input' : [22], 'id': 23}

layer11 = {'type': 'dense', 'params': {'units': 10, "in_channels": 256, "in_size": 8}, 'input': [23], 'id': 27}


lr_vanilla = 0.01
opt_algo = {'name': optim.SGD, 'lr': lr_vanilla, 'momentum': 0.9, 'weight_decay': 0.0005, 'alpha': 1.0}
sch_algo = {'name': optim.lr_scheduler.CosineAnnealingLR, 'T_max': 5, 'eta_min': 0, 'last_epoch': -1}
comp = {'optimizer': opt_algo, 'scheduler': sch_algo, 'loss': nn.CrossEntropyLoss, 'metrics': ['accuracy']}


model_descriptor = {}

model_descriptor['layers'] = [layer0, layer1, layer1_1, layer1_2,
                              layer4, layer5, layer5_1, layer5_2,
                              layer8, layer9, layer9_1, layer9_2, layer11]


model_descriptor['compile']= comp

# create a new basic model
mod = ConvNet(model_descriptor)
mod.cuda()

vanilla_model = {'pytorch_model': mod, 'model_descriptor': model_descriptor, 'topo_ordering': mod.topo_ordering}


# train initially our vanilla model and save
vanilla_model['pytorch_model'].fit_vanilla(trainloader, epochs=20)

# save vanilla model weights
torch.save(vanilla_model['pytorch_model'].state_dict(), expfolder + "vanilla_model")



def SpecialChild(n_models, n_mutations, n_epochs_total, initial_model, savepath, folder_out):
    """
    generate and train children, update best model

    n_models = number of child models
    n_mutations = number of mutations/network operators to be applied per model_descriptor
    n_epochs_total = number of epochs for training in total
    initial model = current best model_descriptor
    savepath = where to save stuff
    folder_out = where to save the general files for one run
    """

    n_epochs_each = int(n_epochs_total)

    print('Train all models for', int(n_epochs_each), 'epochs.')

    init_weights_path = savepath + 'ini_weights'
    torch.save(initial_model['pytorch_model'].state_dict(), init_weights_path)

    performance = np.zeros(shape=(n_models,))
    descriptors = []

    for model_idx in range(0, n_models):
        print('\nmodel idx ' + str(model_idx))

        # save some data
        time_overall_s = time.time()

        pytorch_model = ConvNet(initial_model['model_descriptor'])
        pytorch_model.cuda()
        pytorch_model.load_state_dict(torch.load(init_weights_path), strict=False)

        model = {'pytorch_model': pytorch_model,
                 'model_descriptor': copy.deepcopy(initial_model['model_descriptor']),
                 'topo_ordering': pytorch_model.topo_ordering}
        descriptors.append(model['model_descriptor'])

        mutations_applied = []
        # overall , mutations, training
        times = [0, 0, 0]

        # apply operators
        for i in range(0, n_mutations):

            time_mut_s = time.time()

            # we don't mutate the first child!
            if model_idx != 0:

                mutations_probs = np.array([1, 1, 1, 1, 1, 1])
                [model, mutation_type, params] = network_operators.MutateNetwork(model, batch,
                                                                                 mutation_probs=mutations_probs)
                mutations_applied.append(mutation_type)
                time_mut_e = time.time()

                times[1] = times[1] + (time_mut_e - time_mut_s)

                pytorch_total_params = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)

                if pytorch_total_params > max_n_params:
                    break

        # train
        time_train_s = time.time()

        # initial short training of the children
        model['pytorch_model'].fit(trainloader, epochs=n_epochs_each)

        time_train_e = time.time()
        times[2] = times[2] + (time_train_e - time_train_s)

        performance[model_idx] = model['pytorch_model'].evaluate(validloader)

        pytorch_total_params_child = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)
        torch.save(model['pytorch_model'].state_dict(), savepath + 'model_' + str(model_idx))
        with open(folder_out + "performance.txt", "a+") as f_out:
            f_out.write('child ' + str(model_idx) + ' performance ' +str(performance[model_idx])+' num params '+str(pytorch_total_params_child) +'\n')

        descriptors[model_idx] = copy.deepcopy(model['model_descriptor'])

        time_overall_e = time.time()

        times[0] = times[0] + (time_overall_e - time_overall_s)

        np.savetxt(savepath + 'model_' + str(model_idx) + '_times', times)
        descriptor_file = open(savepath + 'model_' + str(model_idx) + '_model_descriptor.txt', 'w')

        for layer in model['model_descriptor']['layers']:
            layer_str = str(layer)
            descriptor_file.write(layer_str + "\n")
        descriptor_file.close()

        # delete the model (attempt to clean the memory)
        del model['pytorch_model']
        del model
        torch.cuda.empty_cache()

    # continue SH steps
    sorted_children = np.argsort(performance)

    n_children = len(sorted_children)
    n_epochs_train_children = n_epochs_each

    while n_children > 1:

        # pick the best halve of the children
        best_children = sorted_children[(n_children // 2):]
        # increase the training budget for them
        n_epochs_train_children = n_epochs_train_children * 2

        print("\nbest_children", best_children)
        print("n_epochs_train_children", n_epochs_train_children)

        for child in best_children:
            print("child ", child)

            # load the child parameters
            pytorch_model = ConvNet(descriptors[child])
            pytorch_model.cuda()
            pytorch_model.load_state_dict(torch.load(savepath + 'model_' + str(child)), strict=False)
            model = {'pytorch_model': pytorch_model,
                     'model_descriptor': copy.deepcopy(descriptors[child]),
                     'topo_ordering': pytorch_model.topo_ordering}

            # train a child
            model['pytorch_model'].fit(trainloader, epochs=n_epochs_train_children)

            # evaluate a child
            performance[child] = model['pytorch_model'].evaluate(validloader)
            pytorch_total_params_child = sum(p.numel() for p in model['pytorch_model'].parameters() if p.requires_grad)
            with open(folder_out + "performance.txt", "a+") as f_out:
                f_out.write('child ' + str(child) + ' performance ' +str(performance[child])+' num params '+str(pytorch_total_params_child) +'\n')

            # update a child model
            torch.save(model['pytorch_model'].state_dict(), savepath + 'model_' + str(child))

            # delete the model (attempt to clean the memory)
            del model['pytorch_model']
            del model
            torch.cuda.empty_cache()

        print("\nperformance ", performance)
        temp_children_array = np.argsort(performance)
        sorted_children = []

        for i, t in enumerate(temp_children_array):
            if t in best_children:
                sorted_children.append(t)

        print("sorted_children ", sorted_children)
        n_children = len(sorted_children)

    print("it should be the winner", sorted_children[0])
    print("it should be the best performance", performance[sorted_children[0]])

    # load the best child
    the_best_child = sorted_children[0]

    pytorch_model = ConvNet(descriptors[the_best_child])
    pytorch_model.cuda()
    pytorch_model.load_state_dict(torch.load(savepath + 'model_' + str(the_best_child)), strict=False)
    model = {'pytorch_model': pytorch_model,
             'model_descriptor': copy.deepcopy(descriptors[the_best_child]),
             'topo_ordering': pytorch_model.topo_ordering}

    with open(folder_out + "performance.txt", "a+") as f_out:
        f_out.write("****************************\n")

    return model, performance[sorted_children[0]]


# main part

for outeriter_idx in range(0, n_experiments):

    # the start point of the run
    start_run = datetime.datetime.now()
    # create folder for this best model

    folder_out = expfolder + 'run_' + str(outeriter_idx) + '/'
    os.mkdir(folder_out)

    # load vanilla model
    initial_model = vanilla_model
    # load the vanilla model parameters
    initial_model['pytorch_model'].load_state_dict(torch.load(expfolder + "vanilla_model"), strict=False)

    # the counter for steps in one particular run
    sh_idx = 0
    while True:

        # create a folder for all models in this iteration
        savepath = folder_out + str(sh_idx) + '/'
        os.mkdir(savepath)

        st = time.time()
        initial_model, perf = SpecialChild(n_models, n_mutations,
                                           budgets, initial_model, savepath, folder_out)
        end = time.time()
        print("\n\n" + 20 * "*")
        print("Performance before final train for run " + str(outeriter_idx) + " model " + str(
            sh_idx) + " performance:" + str(perf))
        print(20 * "*" + "\n\n")

        # check the number of params
        pytorch_total_params = sum(p.numel() for p in initial_model['pytorch_model'].parameters() if p.requires_grad)

        # even we reach the limit of parameters
        if pytorch_total_params > max_n_params:
            break

        # or we reach the limit of mutation duration
        if datetime.datetime.now() > (start_run + datetime.timedelta(hours=mutation_time_limit_hours)):
            break
        sh_idx += 1

    print('final training')
    # load training data without validation part before final training
    trainloader_final, _, testloader_final = utils.prepare_data(valid_frac=0.0)

    # change lr for the final training for some reasons
    initial_model['pytorch_model'].optimizer.param_groups[0]['initial_lr'] = lr_final
    initial_model['pytorch_model'].optimizer.param_groups[0]['lr'] = lr_final

    # train the model
    initial_model['pytorch_model'].fit(trainloader_final, epochs=epoch_final)

    # evaluate the performance
    performance = initial_model['pytorch_model'].evaluate(testloader_final)
    final_num_params = sum(p.numel() for p in initial_model['pytorch_model'].parameters() if p.requires_grad)

    # save everything
    with open(folder_out + "performance.txt", "a+") as f_out:
        f_out.write('final perf ' + str(performance) + ' final number of params ' + str(final_num_params))

    torch.save(initial_model['pytorch_model'].state_dict(), folder_out + 'best_model')
    descriptor_file = open(folder_out + 'best_model_descriptor.txt', 'w')
    for layer in initial_model['model_descriptor']['layers']:
        layer_str = str(layer)
        descriptor_file.write(layer_str + "\n")
    descriptor_file.close()


