import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils


import copy


# class for "Merge layers (by concatenation)" network operator
class Concatenate(torch.nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()
        
    def forward(self, input1, input2):
        return torch.cat((input1, input2), dim=1)


# class for "Merge layers (by convex combining them)" network operator
class ConcatenateConvex(torch.nn.Module):
    def __init__(self):
        super(ConcatenateConvex, self).__init__()
        self._lambda = nn.Parameter(torch.zeros(1,1).cuda())
        
    def forward(self, input1, input2):
        output = (torch.ones(1,1).cuda() - self._lambda) * input1 + self._lambda * input2
        return output


# class for split upt connection
class AddLayer(torch.nn.Module):
    def __init__(self):
        super(AddLayer, self).__init__()

    def forward(self, input1, input2):
        output = input1 + input2
        return output


# class for depthwise separable convolutions
class depthwise_separable_conv(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size, padding):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# the main class creating the model using model_description
class ConvNet(torch.nn.Module):
    def __init__(self, model_descriptor, dropout=False):
        super(ConvNet, self).__init__()

        self.model_descriptor = copy.deepcopy(model_descriptor)
        self.topo_ordering = []
        self.dropout = dropout
        self.alpha = model_descriptor['compile']['optimizer']['alpha']

        self.layerdic = {}

        self.init_layers()

        # optimizer params
        m = self.model_descriptor['compile']['optimizer']['momentum']
        lr = self.model_descriptor['compile']['optimizer']['lr']
        self.optimizer = self.model_descriptor['compile']['optimizer']['name'](self.parameters(), lr=lr, momentum=m)

        # scheduler params (usually they are rewritten at fit* methods)
        T_max = self.model_descriptor['compile']['scheduler']['T_max']
        eta_min = self.model_descriptor['compile']['scheduler']['eta_min']
        last_epoch = self.model_descriptor['compile']['scheduler']['last_epoch']
        self.scheduler = self.model_descriptor['compile']['scheduler']['name'](self.optimizer, T_max=T_max,
                                                                              eta_min=eta_min, last_epoch=last_epoch)

        self.loss = self.model_descriptor['compile']['loss']()

    def init_layers(self):

        model_descriptor_layers_copy = copy.deepcopy(self.model_descriptor['layers'])
        layers2process = [layer for layer in model_descriptor_layers_copy if layer['type'] == 'input']

        while len(layers2process) > 0:

            layer_added = True

            # get next layer to process
            layer = layers2process.pop(0)
            layer['id'] = str(layer['id'])
            layer['input'][0] = str(layer['input'][0])

            if layer['type'] == 'input':
                pass

            elif layer['type'] == 'conv':
                # we emulate padding "same" behavior
                padding1 = (int(layer['params']['ks1']) - 1) // 2
                padding2 = (int(layer['params']['ks2']) - 1) // 2

                self.add_module(str(layer['id']), nn.Conv2d(layer['params']['in_channels'],
                                                            layer['params']['channels'],
                                                            kernel_size=(
                                                                layer['params']['ks1'],
                                                                layer['params']['ks2']),
                                                            padding=(padding1, padding2)))
            elif layer['type'] == 'sep':
                padding1 = (int(layer['params']['ks1']) - 1) // 2
                padding2 = (int(layer['params']['ks2']) - 1) // 2

                self.add_module(str(layer['id']), depthwise_separable_conv(layer['params']['in_channels'],
                                                                layer['params']['channels'],
                                                                kernel_size=(
                                                                    layer['params']['ks1'],
                                                                    layer['params']['ks2']),
                                                                padding=(padding1, padding2)))

            elif layer['type'] == 'batchnorm':
                self.add_module(str(layer['id']), nn.BatchNorm2d(num_features=layer['params']['in_channels']))

            elif layer['type'] == 'activation':
                self.add_module(str(layer['id']), nn.ReLU())

            elif layer['type'] == 'pool':

                if layer['params']['pooltype'] == 'avg':
                    self.conv.add_module(str(layer['id']), nn.AvgPool2d(kernel_size=layer['params']['poolsize']))

                elif layer['params']['pooltype'] == 'max':
                    self.add_module(str(layer['id']), nn.MaxPool2d(kernel_size=layer['params']['poolsize']))

            elif layer['type'] == 'dense':
                # before applying the dense layer, we need to average input
                # this layer is fixed and has the constant name
                self.add_module("agp", nn.AvgPool2d(layer['params']['in_size']))

                # we split layers in two places because we can't use Linear layer directly to convolutional layer
                # we should reshape it before (done at a forward-function)
                self.add_module(str(layer['id']),
                                nn.Linear(layer['params']['in_channels'], layer['params']['units']))

                self.output_id = layer['id']
                self.last_channels = layer['params']['in_channels']

            elif layer['type'] == 'merge':

                # for all merge layers there are two input layers required
                # they must be already at modules before we add merge layer
                if (str(layer['input'][0]) in self.topo_ordering) and (
                        str(layer['input'][1]) in self.topo_ordering) and (str(layer['id']) not in self.topo_ordering):

                    if layer['params']['mergetype'] == 'concat':
                        self.add_module(str(layer['id']), Concatenate())

                    elif layer['params']['mergetype'] == 'convcomb':
                        self.add_module(str(layer['id']), ConcatenateConvex())

                    elif layer['params']['mergetype'] == 'add':
                        self.add_module(str(layer['id']), AddLayer())

                else:
                    # wait until both input layers are added (we can check topo_ordering)
                    layer_added = False

            if layer_added:
                # append all layers whose input node = current layer
                layers2process.extend(
                    [subsequent_layers for subsequent_layers in copy.deepcopy(self.model_descriptor['layers']) if
                     int(layer['id']) in subsequent_layers['input']])

                self.topo_ordering.append(str(layer['id']))

    def forward(self, x):
        model_descriptor_layers_copy = copy.deepcopy(self.model_descriptor['layers'])
        layers2process = [inputlayer for inputlayer in model_descriptor_layers_copy if inputlayer['type'] == 'input']

        # this is our main dictionary which allows to receive the output from any layer
        self.layerdic = {}

        while len(layers2process) > 0:

            # get next layer to process
            layer = layers2process.pop(0)
            layer['id'] = str(layer['id'])
            layer['input'][0] = str(layer['input'][0])

            layer_added = True

            if layer['type'] == 'input':
                self.layerdic[layer['id']] = x

            elif layer['type'] == 'dense':
                agp_result = self._modules["agp"](self.layerdic[layer['input'][0]])

                flat_result = agp_result.view(-1, self.last_channels)

                dense_result = self._modules[str(layer['id'])](flat_result)
                self.layerdic[layer['id']] = dense_result

            elif layer['type'] == 'merge':
                # for all merge layers there are two input layers required
                # they must be already at layerdic before we can receive the output from the merge layer
                if (str(layer['input'][0]) in self.layerdic.keys()) and (str(layer['input'][1]) in self.layerdic.keys()) \
                        and (str(layer['id']) not in self.layerdic.keys()):

                    self.layerdic[layer['id']] = self._modules[str(layer['id'])](
                        self.layerdic[str(layer['input'][0])], self.layerdic[str(layer['input'][1])])

                else:
                    # wait until both input layers are added (we can check topo_ordering)
                    layer_added = False

            else:  # for the rest type of layers just call forward function (=call())
                self.layerdic[layer['id']] = self._modules[str(layer['id'])](self.layerdic[layer['input'][0]])

            if layer_added:
                # append all layers whose input node = current layer
                layers2process.extend(
                    [subsequent_layers for subsequent_layers in copy.deepcopy(self.model_descriptor['layers']) if
                     int(layer['id']) in subsequent_layers['input']])

                # last layer so far
                self.output_id = layer['id']

        output = self.layerdic[self.output_id]
        return output

    def train(self, x_val, y_val):
        x = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False)

        # MIX UP PREPARE
        x, y_a, y_b, lam = utils.mixup_data(x, y, self.alpha)

        x = Variable(x, requires_grad=False)
        y_a = Variable(y_a, requires_grad=False)
        y_b = Variable(y_b, requires_grad=False)

        ###################

        self.optimizer.zero_grad()

        output = self.forward(x)

        # MIXUP CRITERION
        loss_mixup = lam * self.loss(output, y_a) + (1 - lam) * self.loss(output, y_b)

        # weight decay
        L2_decay_sum = 0        
        for name, param in self.named_parameters():
            if 'weight' in name:
                name_id = str(name.split('.')[0]) 

                layer_name = copy.deepcopy(self._modules[name_id].__class__.__name__)
                if layer_name == 'Conv2d' or layer_name == 'Linear' or layer_name == "depthwise_separable_conv":
                    L2_decay_sum += 0.0005 * torch.norm(param.view(-1),2)

        # total loss
        loss_loc = loss_mixup + L2_decay_sum

        loss_loc.backward(retain_graph=True)

        self.optimizer.step()

        return output, loss_loc.data

    def fit(self, trainloader, epochs=1, scheduler=optim.lr_scheduler.CosineAnnealingLR):
        # fit with a scheduler which will be created at this function
        loss_arr = []

        # initialize scheduler
        n_minibatches = len(trainloader)
        sch_epochs = epochs * n_minibatches - 1
        self.scheduler = scheduler(self.optimizer, T_max=sch_epochs, eta_min=0, last_epoch=-1)

        for i in range(epochs):
            print("epoch = ", i)
            loss_arr = []
            for batch_idx, (inputs, targets) in enumerate(trainloader):

                self.scheduler.step()
                data, target = Variable(inputs.cuda()), Variable(targets.cuda())

                outputs, loss = self.train(data, target)

                loss_arr.append(loss)

                if batch_idx % 50 == 0:
                    print(batch_idx, loss)
        return loss_arr

    def fit_vanilla(self, trainloader, epochs=1):
        # fit without any scheduler
        loss_arr = []

        for i in range(epochs):

            print("epoch = ", i)
            loss_arr = []
            for batch_idx, (inputs, targets) in enumerate(trainloader): 

                data, target = Variable(inputs.cuda()), Variable(targets.cuda())

                outputs, loss = self.train(data, target)

                loss_arr.append(loss)

                if batch_idx % 50 == 0:
                    print(batch_idx, loss)
        return loss_arr

    def fit_with_sch(self, trainloader, epochs=1):
        # fit with a scheduler which was initialized somewhere else
        # (like SpecialChild function in main_cy.py)
        loss_arr = []

        for i in range(epochs):

            print("epoch = ", i)
            loss_arr = []
            for batch_idx, (inputs, targets) in enumerate(trainloader):

                self.scheduler.step()

                data, target = Variable(inputs.cuda()), Variable(targets.cuda())

                outputs, loss = self.train(data, target)

                loss_arr.append(loss)

                if batch_idx % 50 == 0:
                    print(batch_idx, loss)
        return loss_arr

    def predict(self, x_val):
        outputs = self.forward(x_val)
        temp_cpu_output = outputs.cpu()
        return temp_cpu_output.data.numpy().argmax(axis=1)

    def evaluate(self, testloader):
        acc = 0
        set_size = 0
        for batch_idx_test, (inputs_test, targets_test) in enumerate(testloader):
            data_test, target_test = Variable(inputs_test.cuda()), Variable(targets_test.cuda())

            y_hat = self.predict(data_test)
            y_hat_np = np.array(y_hat)
            target_test_cpu = target_test.cpu()
            target_test_np = target_test_cpu.numpy()

            acc += np.sum(y_hat_np == target_test_np)
            set_size = set_size + len(inputs_test)

        acc = 100 * acc / set_size
        print("Acc = " + str(acc) + "%")
        return acc
