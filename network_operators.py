import numpy as np
import copy

import torch
import torch.nn as nn
from random import randint
import math

from layers import ConvNet
import utils


def MutateNetwork(model, batch, id_mutation=1, mutation_probs='unif', inheritance=True):
    # apply network operators = mutations

    """ possible operators:
    # network morphisms
    1) Merge layers (by convex combining them)
    2) Merge layers (by concatenation)
    3) Insert Conv Layer Block
    4) Alter (increase ) number of channels in conv layer
    5) replace one conv layer by two parallel branches (split upt connection)
    6) Insert SepConv Layer Block
    """

    # hyperparameters
    n_mutations = 6  # number of operators/mutations
    allowed_kernel_sizes = [3, 5]  # allowed kernel sizes
    allowed_factors_for_altering_channels = [2, 4]
    max_n_channels = 600  # limit for one layer
    allowed_factors_downsampling = [1, 2, 4]  # allowed downsampling factors for connecting layers

    #################################

    pytorch_model = model['pytorch_model']
    model_descriptor = model['model_descriptor']
    topo_ordering = model['topo_ordering']

    # sample mutation to apply
    # default is uniform
    if isinstance(mutation_probs, str) and mutation_probs == 'unif':
        mutation_probs = np.ones((1, n_mutations))[0]
        mutation_probs[2] = mutation_probs[2] * 2

    mutation_list = np.arange(1, n_mutations + 1)

    valid_mutation_found = False

    while not valid_mutation_found:

        if np.sum(mutation_probs) == 0:
            print('invalid distribution')
            return [model, -1, {'empyty': []}]

        # normalize distribution
        mutation_probs = mutation_probs / np.sum(mutation_probs)

        # sample mutation type
        mutation_type = np.random.choice(mutation_list, 1, p=mutation_probs)[0]

        ###############################################

        print('Mutation to be applied: ' + str(mutation_type))

        # Merge layers (by convex combining them)
        if mutation_type == 1:

            layers = model_descriptor['layers']

            matching_output_shapes = GetMatchingOutputShapes(pytorch_model, batch, layers)

            existing_connections = [layer['input'] for layer in model_descriptor['layers'] if layer['type'] == 'merge']

            possible_connections = []
            for (id_1, id_2) in matching_output_shapes:

                if not [int(id_1), int(id_2)] in existing_connections and not [int(id_2),
                                                                               int(id_1)] in existing_connections:
                    possible_connections.append([id_1, id_2])

            n_possible_connections = len(possible_connections)

            if n_possible_connections > 0:  # can apply mutation

                # choose random connection
                new_connection = randint(0, n_possible_connections - 1)

                [id_1, id_2] = possible_connections[new_connection]

                layer1_id = topo_ordering[min(topo_ordering.index(id_1), topo_ordering.index(id_2))]
                layer2_id = topo_ordering[max(topo_ordering.index(id_1), topo_ordering.index(id_2))]

                print('Merging layers', layer1_id, 'and', layer2_id, 'by convex combination.')

                new_model = MergeLayersAdd(layer1_id, layer2_id, model)

                params = {'layer1': layer1_id, 'layer2': layer2_id}

                valid_mutation_found = True

            else:  # mutation can not be applied to model, force to do other mutation

                print('Mutation not applicable. Choosing another mutation...')
                mutation_probs[mutation_type - 1] = 0

        # Merge layers (by concatenation)
        elif mutation_type == 2:

            layers = model_descriptor['layers']

            preliminary_possible_connections = []

            for factor in allowed_factors_downsampling:
                preliminary_possible_connections.extend(
                    GetMatchingOutputShapesByFactor(pytorch_model, factor, batch, layers))

            possible_connections = []

            for possible_connection in preliminary_possible_connections:
                [layer1_temp_id, layer2_temp_id, factor] = possible_connection

                # maybe swap layers depending on topological ordering
                layer1_id = topo_ordering[min(topo_ordering.index(layer1_temp_id), topo_ordering.index(layer2_temp_id))]
                layer2_id = topo_ordering[max(topo_ordering.index(layer1_temp_id), topo_ordering.index(layer2_temp_id))]

                # if the next layer is Conv
                if utils.GetSubsequentLayers(int(layer2_id), model_descriptor)[-1]:
                    possible_connections.append([layer1_id, layer2_id, factor])

            n_possible_connections = len(possible_connections)

            if n_possible_connections > 0:  # can apply mutation

                # choose random connection
                new_connection = randint(0, n_possible_connections - 1)

                [layer1_id, layer2_id, factor] = possible_connections[new_connection]

                print('Merging layers', layer1_id, 'and', layer2_id, 'by concatenation. Factor ', str(factor))
                new_model = MergeLayersConcatWithDS(layer1_id, layer2_id, factor, model, batch)

                params = {'layer1': layer1_id, 'layer2': layer2_id}

                valid_mutation_found = True
            else:  # mutation can not be applied to model, force to do other mutation

                print('Mutation not applicable. Choosing another mutation...')
                mutation_probs[mutation_type - 1] = 0

        # Insert Conv Layer
        elif mutation_type == 3:

            predecessor_layer_type_allowed = {"AddLayer", 'ReLU', 'MaxPool2d', "AvgPool2d", "ConcatenateConvex", "Concatenate"}

            allowed_predecessor = []
            for i, layer_type in enumerate(pytorch_model._modules):
                class_name = pytorch_model._modules[layer_type].__class__.__name__

                if class_name in predecessor_layer_type_allowed and layer_type != "agp":
                    allowed_predecessor.append(layer_type)

            # randomly chosen predecessor
            r = randint(0, len(allowed_predecessor) - 1)
            predecessor_id = int(allowed_predecessor[r])

            # choose one of its subsequent layers as successor
            subsequentlayers = utils.GetSubsequentLayers(predecessor_id, model_descriptor)[0]

            successor_id = subsequentlayers[randint(0, len(subsequentlayers) - 1)]['id']

            # sample kernel size
            kernel_size = allowed_kernel_sizes[randint(0, len(allowed_kernel_sizes) - 1)]

            print('Inserting conv layer between layer', predecessor_id, 'and', successor_id, 'with kernel size',
                  kernel_size, '.')

            new_model, layers2burnin = InsertConvolution(predecessor_id, successor_id, model, kernel_size,
                                                         id_mutation, batch)
            if layers2burnin:
                params = {'layer1': predecessor_id, 'layer2': successor_id, 'kernel_size': kernel_size,
                          'layers2burnin': layers2burnin}

            else:

                params = {'layer1': predecessor_id, 'layer2': successor_id, 'kernel_size': kernel_size}

            # always possible
            valid_mutation_found = True

        # Alter (increase) number of channels in conv layer
        elif mutation_type == 4:

            # if altering number of channels, do this by a factor of ...

            convlayers = [layer for layer in model_descriptor['layers'] if (layer['type'] == 'conv' or layer['type'] == 'sep' )]
            mult_factor = allowed_factors_for_altering_channels[
                randint(0, len(allowed_factors_for_altering_channels) - 1)]

            for i, convlayer in enumerate(copy.deepcopy(convlayers)):
                # print(i)
                bn_layer = [layer for layer in model_descriptor['layers'] if layer['input'] == [convlayer['id']]][0]
                acti_layer = [layer for layer in model_descriptor['layers'] if layer['input'] == [bn_layer['id']]][
                    0]

                # if the next layer after activation is NOT convolution (or dense) 
                # or future layer size will be more han allowed maximum
                # delete this layer from candidates array

                if (not utils.GetSubsequentLayers(acti_layer['id'], model_descriptor)[-1]) or (
                        convlayer['params']['channels'] * mult_factor > max_n_channels):

                    convlayers.remove(convlayer)

            if convlayers:  # not empty

                # sample one of them
                layer2alter = convlayers[randint(0, len(convlayers) - 1)]
                layer2alter_id = layer2alter['id']

                # sample new number of channels

                new_n_channels = mult_factor * layer2alter['params']['channels']

                print('Altering number of channels of layer', layer2alter['id'], 'from',
                      layer2alter['params']['channels'], 'channels to ', new_n_channels, 'channels.')

                new_model = AlterNChannels(layer2alter_id, new_n_channels, model, id_mutation)
                
                params = {'layer': layer2alter_id, 'channels': new_n_channels}
                valid_mutation_found = True
            else:

                mutation_probs[mutation_type - 1] = 0

        # replace one conv layer by two parallel branches (split upt connection)
        elif mutation_type == 5:

            # get all conv layers
            convlayers = [layer for layer in model_descriptor['layers'] if (layer['type'] == 'conv' or layer['type'] == 'sep')]

            if convlayers:  # not empty

                # sample one of them
                layer2split = convlayers[randint(0, len(convlayers) - 1)]
                layer2split_id = layer2split['id']

                print('Splitting up conv layer', layer2split_id, '.')

                new_model = SplitConnection(layer2split_id, model, batch, id_mutation)

                params = {'layer': layer2split_id}

                valid_mutation_found = True

            else:

                mutation_probs[mutation_type - 1] = 0

        # Insert Separable Conv Layer
        elif mutation_type == 6:

            predecessor_layer_type_allowed = {"AddLayer", 'ReLU', 'MaxPool2d', "AvgPool2d", "ConcatenateConvex", "Concatenate"}

            allowed_predecessor = []
            for i, layer_type in enumerate(pytorch_model._modules):
                class_name = pytorch_model._modules[layer_type].__class__.__name__

                if class_name in predecessor_layer_type_allowed and layer_type != "agp":
                    allowed_predecessor.append(layer_type)

            # randomly chosen predecessor
            r = randint(0, len(allowed_predecessor) - 1)

            predecessor_id = int(allowed_predecessor[r])

            # choose one of its subsequent layers as successor
            subsequentlayers = utils.GetSubsequentLayers(predecessor_id, model_descriptor)[0]

            successor_id = subsequentlayers[randint(0, len(subsequentlayers) - 1)]['id']

            # sample kernel size
            kernel_size = allowed_kernel_sizes[randint(0, len(allowed_kernel_sizes) - 1)]

            print('Inserting conv layer between layer', predecessor_id, 'and', successor_id, 'with kernel size',
                  kernel_size, '.')

            new_model, layers2burnin = InsertSepConvolution(predecessor_id, successor_id, model, kernel_size,
                                                         id_mutation, batch)
            if layers2burnin:
                params = {'layer1': predecessor_id, 'layer2': successor_id, 'kernel_size': kernel_size,
                          'layers2burnin': layers2burnin}

            else:

                params = {'layer1': predecessor_id, 'layer2': successor_id, 'kernel_size': kernel_size}

            # always possible
            valid_mutation_found = True

    print("\nMutation was successful!\n\n")
    return [new_model, mutation_type, params]


def GetMatchingOutputShapes(model, batch, layers):
    # utils for building models
    # return all pairs of layers that have same output shep wrt spatial dim only
    # !!!! returns index of layer, not its ID !!!!!!!!

    allowed_layer_types4connection = {'ReLU', 'MaxPool2d'}

    model.forward(batch)

    MatchingOS = []

    for layer_idx1 in layers:
        for layer_idx2 in layers:

            id1 = str(layer_idx1["id"])
            id2 = str(layer_idx2["id"])
            if id1 != "0" and id2 != "0" and id1 != id2:

                if (model._modules[id1].__class__.__name__ in allowed_layer_types4connection) \
                        and (model._modules[id2].__class__.__name__ in allowed_layer_types4connection):
                    # the size of layers should correnponds with factor
                    if model.layerdic[id1].size()[1] == model.layerdic[id2].size()[1]:
                        if model.layerdic[id1].size()[2] == model.layerdic[id2].size()[2]:
                            if model.layerdic[id1].size()[3] == model.layerdic[id2].size()[3]:
                                MatchingOS.append([id1, id2])

    return MatchingOS


def MergeLayersAdd(layer1_id, layer2_id, old_model, inheritance=True):
    # adds a connection from  layer1_id to  layer2_id,
    # i.e. insert a layer ConcatenateConvex(layer1_id,layer2_id) after layer 2

    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])

    subsequentlayers, _, _ = utils.GetSubsequentLayers(int(layer2_id), new_model_descriptor, no_merge_layers=False)

    new_id = utils.GetUnusedID(new_model_descriptor)

    merge_layer = {'type': 'merge',
                   'params': {'mergetype': 'convcomb'},
                   'id': new_id,
                   'input': [int(layer2_id), int(layer1_id)]}

    new_model_descriptor['layers'].append(merge_layer)

    utils.ReplaceInput(subsequentlayers, layer2_id, new_id)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()

    if inheritance:
        new_pytorch_model = utils.InheritWeights(old_model['pytorch_model'], new_pytorch_model)

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model


def MergeLayersConcatWithDS(layer1_id, layer2_id, downsampling_factor, old_model, batch, id_mutation=1, inheritance=True):
    # adds a concat connection from  layer1_id to  layer2_id,
    # i.e. insert a layer Concatenate(layer1_id,layer2_id) after layer 2
    # id_mutation bool, if true: new_model(x) == old_model(x) for all x

    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])
    old_pytorch_model = old_model['pytorch_model']

    [subsequentlayers, _, _] = utils.GetSubsequentLayers(int(layer2_id), new_model_descriptor)

    new_id = utils.GetUnusedID(new_model_descriptor)
    new_id_subseq = new_id + 1

    if downsampling_factor != 1:  # pooling layer needed

        new_id_pool = new_id_subseq + 1

        # check which layer is smaller
        old_pytorch_model.forward(batch)
        if old_pytorch_model.layerdic[str(layer1_id)].size()[2] > old_pytorch_model.layerdic[str(layer2_id)].size()[2]:

            pool_layer = {'type': 'pool',
                          'params': {'poolsize': downsampling_factor,
                                     'pooltype': 'max'},
                          'id': new_id_pool,
                          'input': [int(layer1_id)]}
            new_model_descriptor['layers'].append(pool_layer)

            merge_layer = {'type': 'merge',
                           'params': {'mergetype': 'concat'},
                           'id': new_id,
                           'input': [int(layer2_id), int(new_id_pool)]}
            new_model_descriptor['layers'].append(merge_layer)
        else:
            pool_layer = {'type': 'pool',
                          'params': {'poolsize': downsampling_factor,
                                     'pooltype': 'max'},
                          'id': new_id_pool,
                          'input': [int(layer2_id)]}
            new_model_descriptor['layers'].append(pool_layer)

            merge_layer = {'type': 'merge',
                           'params': {'mergetype': 'concat'},
                           'id': new_id,
                           'input': [int(layer1_id), int(new_id_pool)]}
            new_model_descriptor['layers'].append(merge_layer)

    else:

        merge_layer = {'type': 'merge',
                       'params': {'mergetype': 'concat'},
                       'id': new_id,
                       'input': [int(layer2_id), int(layer1_id)]}

        new_model_descriptor['layers'].append(merge_layer)

    old_id_subseq = subsequentlayers[0]['id']
    subsequentlayers[0]['id'] = new_id_subseq

    utils.ReplaceInput(subsequentlayers, int(layer2_id), new_id)

    # update input for subsequent layers of subsequent conv layer
    subsubsequentlayers, _, _ = utils.GetSubsequentLayers(old_id_subseq, new_model_descriptor)

    # comment it
    utils.ReplaceInput(subsubsequentlayers, old_id_subseq, new_id_subseq)

    # replace in_channels for subsequent layer
    # we need the number of channels from this layers layer2_id, layer1_id
    # can use forward of old model to calculate the shape
    old_pytorch_model.forward(batch)
    parent_1_channels = old_pytorch_model.layerdic[str(layer1_id)].shape[1]
    parent_2_channels = old_pytorch_model.layerdic[str(layer2_id)].shape[1]

    subsequentlayers[0]['params']['in_channels'] = int(parent_1_channels) + int(parent_2_channels)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()

    if inheritance:

        new_pytorch_model = utils.InheritWeights(old_pytorch_model, new_pytorch_model)

    if id_mutation:
        try:    
            new_pytorch_model.forward(batch)
        except:
            print("Problem with sizes MergeLayersConcatWithDS")
            return old_model

        new_weights = copy.deepcopy(new_pytorch_model._modules[str(new_id_subseq)].weight)
        old_weights = copy.deepcopy(old_pytorch_model._modules[str(old_id_subseq)].weight)

        old_bias = copy.deepcopy(old_pytorch_model._modules[str(old_id_subseq)].bias)

        new_weights[:, 0:old_weights.shape[1], :, :] = old_weights
        new_weights[:, old_weights.shape[1]:, :, :] = torch.from_numpy(
            np.zeros(shape=new_weights[:, old_weights.shape[1]:, :, :].shape))

        # save
        state_dict = {"weight": nn.Parameter(new_weights.cuda()),
                      "bias": nn.Parameter(old_bias.cuda())}
        new_pytorch_model._modules[str(new_id_subseq)].load_state_dict(state_dict)

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model


def InsertConvolution(predecessor_id, successor_id, old_model, kernel_size=3, id_mutation=1, batch=None,
                      inheritance=True):

    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])
    old_pytorch_model = old_model['pytorch_model']

    successor = [layer for layer in new_model_descriptor['layers'] if str(layer['id']) == str(successor_id)][0]

    new_id_conv = utils.GetUnusedID(new_model_descriptor)
    new_id_bn = new_id_conv + 1
    new_id_acti = new_id_bn + 1

    old_pytorch_model.forward(batch)
    channels = old_pytorch_model.layerdic[str(predecessor_id)].size()[1]

    new_layer_conv = {'type': 'conv',
                      'params': {'channels': channels,
                                 'ks1': kernel_size,
                                 'ks2': kernel_size,
                                 "in_channels": channels},
                      'input': [predecessor_id],
                      'id': new_id_conv}

    new_layer_bn = {'type': 'batchnorm',
                    'params': {"in_channels": channels},
                    'input': [new_id_conv],
                    'id': new_id_bn}

    new_layer_acti = {'type': 'activation',
                      'params': {},
                      'input': [new_id_bn],
                      'id': new_id_acti}

    utils.ReplaceInput([successor], predecessor_id, new_id_acti)

    new_model_descriptor['layers'].append(new_layer_conv)
    new_model_descriptor['layers'].append(new_layer_bn)
    new_model_descriptor['layers'].append(new_layer_acti)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()
    new_pytorch_model._modules[str(new_id_bn)].momentum = 1.0
    new_pytorch_model._modules[str(new_id_bn)].eps = 0.0

    # test
    new_pytorch_model.forward(batch)


    if inheritance:
        new_pytorch_model = utils.InheritWeights(old_pytorch_model, new_pytorch_model)

        print("inheritance done")

    if id_mutation:
        # get weights fur id-convolution

        IDConv = conv2d_identity(channels, kernel_size)

        bias_shape = new_pytorch_model._modules[str(new_id_conv)].weight[1].size()[0]

        # save
        state_dict = {"weight": torch.from_numpy(IDConv).cuda(),
                      "bias": torch.from_numpy(np.zeros(shape=bias_shape)).cuda()}
        new_pytorch_model._modules[str(new_id_conv)].load_state_dict(state_dict)

        new_pytorch_model.forward(batch)

        predecessor_output_batch = new_pytorch_model.layerdic[str(new_id_conv)][0]
        predecessor_output_batch_cpu = predecessor_output_batch.cpu()
        predecessor_output_batch_data = predecessor_output_batch_cpu.data.numpy()

        batch_mean = np.mean(predecessor_output_batch_data, axis=(1, 2))

        batch_var = np.var(predecessor_output_batch_data, axis=(1, 2))

        eps = new_pytorch_model._modules[str(new_id_bn)].eps

        #gamma_ini = np.sqrt(batch_var + eps)
        beta_ini = batch_mean

        rm_copy = copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].running_mean)
        rv_copy = copy.deepcopy(np.sqrt(new_pytorch_model._modules[str(new_id_bn)].running_var + eps))

        # save
        state_dict = {"weight": nn.Parameter(rv_copy.cuda()),
                      "bias": nn.Parameter(rm_copy.cuda()),
                      "running_var": torch.from_numpy(batch_var).cuda(),
                      "running_mean": torch.from_numpy(beta_ini).cuda()}
        new_pytorch_model._modules[str(new_id_bn)].load_state_dict(state_dict)

        layers2burnin = []

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model, layers2burnin


def AlterNChannels(layer2alter_id, new_n_channels, old_model, id_mutation=1, inheritance=True):
    # alter number of channels in conv layer with id layer2alter_id

    # copy old model
    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])
    old_pytorch_model = old_model['pytorch_model']

    # get layer where altering number of channels and also subsequent layers (as input of subsequent layer is changed)
    layer2alter_conv = [layer for layer in new_model_descriptor['layers'] if layer['id'] == layer2alter_id][0]
    layer2alter_bn = [layer for layer in new_model_descriptor['layers'] if layer['input'] == [layer2alter_id]][0]
    layer2alter_acti = [layer for layer in new_model_descriptor['layers'] if layer['input'] == [layer2alter_bn['id']]][
        0]

    subsequentlayer2alter = [layer for layer in new_model_descriptor['layers'] if
                             layer2alter_acti['id'] in layer['input']]

    layer_type = layer2alter_conv['type']

    # check some constraints
    assert ((layer2alter_conv['type'] == 'conv') or (layer2alter_conv['type'] == 'sep')), 'Error: Layer hast to be conv or sepconv layer.'
    assert layer2alter_conv['params']['channels'] < new_n_channels, 'Error: Can only increase number of channels.'
    assert len(subsequentlayer2alter) == 1, 'Error, more than one outgoing connection not allowed'
    assert ((subsequentlayer2alter[0]['type'] == 'conv') or (
            subsequentlayer2alter[0]['type'] == 'dense')), 'Error, subsequent layer has to be conv or dense layer'

    # make necessary changes to new descriptor
    layer2alter_conv['params']['channels'] = new_n_channels

    # for new architecture
    layer2alter_bn['params']['in_channels'] = new_n_channels

    old_id_conv = layer2alter_conv['id']
    old_id_bn = layer2alter_bn['id']
    old_id_sub = subsequentlayer2alter[0]['id']

    new_id_conv = utils.GetUnusedID(new_model_descriptor)

    new_id_bn = new_id_conv + 1
    new_id_acti = new_id_conv + 2
    new_id_sub = new_id_conv + 3
    layer2alter_conv['id'] = new_id_conv

    layer2alter_bn['id'] = new_id_bn
    layer2alter_bn['input'] = [new_id_conv]

    layer2alter_acti['id'] = new_id_acti
    layer2alter_acti['input'] = [new_id_bn]

    subsequentlayer2alter[0]['input'] = [new_id_acti]
    subsequentlayer2alter[0]['id'] = new_id_sub

    subsubsequentlayers = [layer for layer in new_model_descriptor['layers'] if old_id_sub in layer['input']]

    # for new architecture
    for layer in subsequentlayer2alter:
        layer['params']['in_channels'] = new_n_channels

    utils.ReplaceInput(subsubsequentlayers, old_id_sub, new_id_sub)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()

    if inheritance:
        new_pytorch_model = utils.InheritWeights(old_model['pytorch_model'], new_pytorch_model)

    if id_mutation:

        # modify weights of changed layers
        if layer_type == 'conv':

            # conv layer where number of channels have been changed

            new_weights_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].weight)
            new_bias_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].bias)

            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].weight)
            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].bias)

            # recalculate
            new_weights_conv[0:old_weights_conv.shape[0], :, :, :] = nn.Parameter(old_weights_conv)
            new_bias_conv[0:old_bias_conv.shape[0]] = nn.Parameter(old_bias_conv)

            # save
            state_dict = {"weight": new_weights_conv.cuda(),
                          "bias": new_bias_conv.cuda()}
            new_pytorch_model._modules[str(new_id_conv)].load_state_dict(state_dict)

        elif layer_type == 'sep':
            print("altering sep conv layer'schannel")
            # depthwise
            #new_weights_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].depthwise.weight)
            #new_bias_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].depthwise.bias)

            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].depthwise.weight)
            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].depthwise.bias)

            # save
            state_dict = {"weight": nn.Parameter(old_weights_conv).cuda(),
                          "bias": nn.Parameter(old_bias_conv).cuda()}
            new_pytorch_model._modules[str(new_id_conv)].depthwise.load_state_dict(state_dict)

            ############################
            # pointwise

            new_weights_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].pointwise.weight)
            new_bias_conv = copy.deepcopy(new_pytorch_model._modules[str(new_id_conv)].pointwise.bias)

            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].pointwise.weight)
            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].pointwise.bias)

            # recalculate
            new_weights_conv[0:old_weights_conv.shape[0], :, :, :] = nn.Parameter(old_weights_conv)
            new_bias_conv[0:old_bias_conv.shape[0]] = nn.Parameter(old_bias_conv)

            # save
            state_dict = {"weight": new_weights_conv.cuda(),
                          "bias": new_bias_conv.cuda()}
            new_pytorch_model._modules[str(new_id_conv)].pointwise.load_state_dict(state_dict)

        # copy old weights for BN layer
        new_weights_bn = []
        new_weights_bn.append(copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].weight))
        new_weights_bn.append(copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].bias))
        new_weights_bn.append(copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].running_mean))
        new_weights_bn.append(copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].running_var))

        old_weights_bn = []
        old_weights_bn.append(copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].weight))
        old_weights_bn.append(copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].bias))
        old_weights_bn.append(copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].running_mean))
        old_weights_bn.append(copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].running_var))

        for weight_idx, weight in enumerate(new_weights_bn):

            if weight_idx < 2:
                new_weights_bn[weight_idx][0:old_weights_bn[weight_idx].shape[0]] = nn.Parameter(
                    old_weights_bn[weight_idx])
            else:
                new_weights_bn[weight_idx][0:old_weights_bn[weight_idx].shape[0]] = old_weights_bn[weight_idx]

        # save
        state_dict = {"weight": new_weights_bn[0].cuda(),
                      "bias": new_weights_bn[1].cuda(),
                      "running_mean": new_weights_bn[2].cuda(),
                      "running_var": new_weights_bn[3].cuda()}
        new_pytorch_model._modules[str(new_id_bn)].load_state_dict(state_dict)

        # we can't check this part because we always have ONLY conv-layers as subsequentlayer
        # probably delete this part for dense
        if subsequentlayer2alter[0]['type'] == 'dense':

            # subsequent layer, fill up with zeros
            new_weights_sub = new_pytorch_model._modules[str(new_id_sub)].weight
            new_bias_sub = new_pytorch_model._modules[str(new_id_sub)].bias

            old_weights_sub = old_pytorch_model._modules[str(old_id_sub)].weight
            old_bias_sub = old_pytorch_model._modules[str(old_id_sub)].bias

            new_weights_sub[0:old_weights_sub.shape[0], :] = nn.Parameter(copy.deepcopy(old_weights_sub))
            new_bias_sub = nn.Parameter(copy.deepcopy(old_bias_sub))

            # save
            state_dict = {"weight": new_weights_sub,
                          "bias": new_bias_sub}
            new_pytorch_model._modules[str(new_id_sub)].load_state_dict(state_dict)
        else:

            new_weights_sub = copy.deepcopy(new_pytorch_model._modules[str(new_id_sub)].weight)
            #new_bias_sub = copy.deepcopy(new_pytorch_model._modules[str(new_id_sub)].bias)

            old_weights_sub = copy.deepcopy(old_pytorch_model._modules[str(old_id_sub)].weight)
            old_bias_sub = copy.deepcopy(old_pytorch_model._modules[str(old_id_sub)].bias)

            # copy old weights
            new_weights_sub[:, 0:old_weights_sub.shape[1], :, :] = old_weights_sub

            # fill up new channels with 0's
            new_weights_sub[:, old_weights_sub.shape[1]:, :, :] = torch.from_numpy(
                np.zeros(shape=new_weights_sub[:, old_weights_sub.shape[1]:, :, :].shape))

            new_bias_sub = copy.deepcopy(old_bias_sub)

            # save
            state_dict = {"weight": nn.Parameter(new_weights_sub.cuda()),
                          "bias": nn.Parameter(new_bias_sub.cuda())}
            new_pytorch_model._modules[str(new_id_sub)].load_state_dict(state_dict)

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model


def GetMatchingOutputShapesByFactor(model, factor, batch, layers):
    # utils for building models
    # return all pairs of layers that have same output shep wrt spatial dim only
    # !!!! returns index of layer, not its ID !!!!!!!!

    allowed_layer_types4connection = {'ReLU', 'MaxPool2d'}
    n_layers = len(model._modules)

    model.forward(batch)

    MatchingOS = []

    for layer_idx1 in layers:
        for layer_idx2 in layers:

            id1 = str(layer_idx1["id"])
            id2 = str(layer_idx2["id"])
            if id1 != "0" and id2 != "0" and id1 != id2:

                if (model._modules[id1].__class__.__name__ in allowed_layer_types4connection) \
                        and model._modules[id2].__class__.__name__ in allowed_layer_types4connection:
                    # the size of layers should corresponds with factor
                    if model.layerdic[id1].size()[2] == factor * model.layerdic[id2].size()[2]:
                        MatchingOS.append([id1, id2, factor])

    return MatchingOS


def conv2d_identity(channels, kernel_size):
    # returns convolution which is ID mapping with ID.shape=(channels,channels,kernel_size,kernel_size)
    # so that Conv(ID,G) = G for arbitrary G

    IDconv = np.zeros(shape=(channels, channels, kernel_size, kernel_size))

    non_zero_kernel_idx = math.floor(kernel_size / 2)

    for idx in range(0, channels):
        IDconv[idx, idx, non_zero_kernel_idx, non_zero_kernel_idx] = 1

    return IDconv


def conv2d_identity_sep(channels, kernel_size):
    # returns depthwise convolution which is ID mapping with ID.shape=(channels, 1, kernel_size,kernel_size)
    # so that Conv(ID,G) = G for arbitrary G

    IDconv = np.zeros(shape=(channels, 1, kernel_size, kernel_size))

    non_zero_kernel_idx = math.floor(kernel_size / 2)

    for idx in range(0, channels):
        IDconv[idx, 0, non_zero_kernel_idx, non_zero_kernel_idx] = 1

    return IDconv


def SplitConnection(layer2split_id, old_model, batch, id_mutation=1, split_factor=0.3, inheritance=True):

    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])

    old_pytorch_model = old_model['pytorch_model']

    # get BN and activation layer belonging to conv layer

    layer2split_bn = [layer for layer in new_model_descriptor['layers'] if layer['input'] == [layer2split_id]][0]

    layer2split_acti = [layer for layer in new_model_descriptor['layers'] if layer['input'] == [layer2split_bn['id']]][
        0]

    subsequentlayers = [layer for layer in new_model_descriptor['layers'] if layer2split_acti['id'] in layer['input']]

    layer2split = [layer for layer in new_model_descriptor['layers'] if layer['id'] == layer2split_id][0]

    old_id_conv = layer2split_id
    old_id_bn = layer2split_bn['id']
    old_id_acti = layer2split_acti['id']


    old_bn_layer = [layer for layer in new_model_descriptor['layers'] if layer['id'] == old_id_bn][0]

    assert ((layer2split['type'] == 'conv') or (layer2split['type'] == 'sep')), 'Error: Layer hast to be conv or sep layer.'

    layer_type = layer2split['type']

    # 1st branch
    new_id_conv1 = utils.GetUnusedID(new_model_descriptor)
    new_id_bn1 = new_id_conv1 + 1
    new_id_acti1 = new_id_conv1 + 2

    # 2nd branch
    new_id_conv2 = new_id_conv1 + 3
    new_id_bn2 = new_id_conv1 + 4
    new_id_acti2 = new_id_conv1 + 5

    # sum up split
    new_id_add = new_id_conv1 + 6
    layer2split['id'] = new_id_conv1
    layer2split_bn['id'] = new_id_bn1
    layer2split_acti['id'] = new_id_acti1

    layer2split_bn['input'] = [new_id_conv1]
    layer2split_acti['input'] = [new_id_bn1]

    new_conv_layer = {'type': layer2split['type'],
                      'params': copy.deepcopy(layer2split['params']),
                      'id': new_id_conv2,
                      'input': copy.deepcopy(layer2split['input'])}

    new_bn_layer = {'type': 'batchnorm',
                    'params': copy.deepcopy(old_bn_layer['params']),
                    'id': new_id_bn2,
                    'input': [new_id_conv2]}

    new_acti_layer = {'type': 'activation',
                      'params': {},
                      'id': new_id_acti2,
                      'input': [new_id_bn2]}

    new_merge_layer = {'type': 'merge',
                       'params': {'mergetype': 'add'},
                       'id': new_id_add,
                       'input': [int(new_id_acti1), int(new_id_acti2)]}

    utils.ReplaceInput(subsequentlayers, old_id_acti, new_id_add)

    new_model_descriptor['layers'].append(new_conv_layer)
    new_model_descriptor['layers'].append(new_bn_layer)
    new_model_descriptor['layers'].append(new_acti_layer)
    new_model_descriptor['layers'].append(new_merge_layer)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()

    if inheritance:
        new_pytorch_model = utils.InheritWeights(old_model['pytorch_model'], new_pytorch_model)

    if id_mutation:
        new_pytorch_model.forward(batch)

        if layer_type == 'conv':
            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].weight)
            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].bias)

            # new_id_conv1
            state_dict = {"weight": nn.Parameter((split_factor * old_weights_conv).cuda()),
                          "bias": nn.Parameter((split_factor * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv1)].load_state_dict(state_dict)

            # new_id_conv2
            state_dict = {"weight": nn.Parameter(((1 - split_factor) * old_weights_conv).cuda()),
                          "bias": nn.Parameter(((1 -split_factor) * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv2)].load_state_dict(state_dict)

        elif layer_type == 'sep':

            # DEPTHWISE IS SPLITTING UP
            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].depthwise.weight)
            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].depthwise.bias)

            # new_id_conv1
            state_dict = {"weight": nn.Parameter((split_factor * old_weights_conv).cuda()),
                          "bias": nn.Parameter((split_factor * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv1)].depthwise.load_state_dict(state_dict)

            # new_id_conv2
            state_dict = {"weight": nn.Parameter(((1 - split_factor) * old_weights_conv).cuda()),
                          "bias": nn.Parameter(((1 -split_factor) * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv2)].depthwise.load_state_dict(state_dict)

            # POINTWISE IS SPLITTING UP
            old_weights_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].pointwise.weight)

            old_bias_conv = copy.deepcopy(old_pytorch_model._modules[str(old_id_conv)].pointwise.bias)

            # new_id_conv1
            state_dict = {"weight": nn.Parameter((split_factor * old_weights_conv).cuda()),
                          "bias": nn.Parameter((split_factor * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv1)].pointwise.load_state_dict(state_dict)

            # new_id_conv2
            state_dict = {"weight": nn.Parameter(((1 - split_factor) * old_weights_conv).cuda()),
                          "bias": nn.Parameter(((1 -split_factor) * old_bias_conv).cuda())}

            new_pytorch_model._modules[str(new_id_conv2)].pointwise.load_state_dict(state_dict)


        # old_id_bn
        old_weights_bn = copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].weight)
        old_bias_bn = copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].bias)
        old_mean_bn = copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].running_mean)
        old_var_bn = copy.deepcopy(old_pytorch_model._modules[str(old_id_bn)].running_var)

        # new_id_bn1
        state_dict = {"weight": nn.Parameter((split_factor * old_weights_bn).cuda()),
                      "bias": nn.Parameter((split_factor * old_bias_bn).cuda()),
                      "running_var": nn.Parameter((split_factor * old_var_bn).cuda()),
                      "running_mean": nn.Parameter((split_factor * old_mean_bn).cuda())}

        new_pytorch_model._modules[str(new_id_bn1)].load_state_dict(state_dict)

        # new_id_bn2
        state_dict = {"weight": nn.Parameter(((1 - split_factor) * old_weights_bn).cuda()),
                      "bias": nn.Parameter(((1 - split_factor) * old_bias_bn).cuda()),
                      "running_var": nn.Parameter(((1 - split_factor) * old_var_bn).cuda()),
                      "running_mean": nn.Parameter(((1 - split_factor) * old_mean_bn).cuda())}

        new_pytorch_model._modules[str(new_id_bn2)].load_state_dict(state_dict)

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model


def InsertSepConvolution(predecessor_id, successor_id, old_model, kernel_size=3, id_mutation=1, batch=None,
                      inheritance=True):

    new_model_descriptor = copy.deepcopy(old_model['model_descriptor'])
    old_pytorch_model = old_model['pytorch_model']

    successor = [layer for layer in new_model_descriptor['layers'] if str(layer['id']) == str(successor_id)][0]

    new_id_conv = utils.GetUnusedID(new_model_descriptor)
    new_id_bn = new_id_conv + 1
    new_id_acti = new_id_bn + 1

    old_pytorch_model.forward(batch)
    channels = old_pytorch_model.layerdic[str(predecessor_id)].size()[1]

    new_layer_conv = {'type': 'sep',
                      'params': {'channels': channels,
                                 'ks1': kernel_size,
                                 'ks2': kernel_size,
                                 "in_channels": channels},
                      'input': [predecessor_id],
                      'id': new_id_conv}

    new_layer_bn = {'type': 'batchnorm',
                    'params': {"in_channels": channels},
                    'input': [new_id_conv],
                    'id': new_id_bn}

    new_layer_acti = {'type': 'activation',
                      'params': {},
                      'input': [new_id_bn],
                      'id': new_id_acti}

    utils.ReplaceInput([successor], predecessor_id, new_id_acti)

    new_model_descriptor['layers'].append(new_layer_conv)
    new_model_descriptor['layers'].append(new_layer_bn)
    new_model_descriptor['layers'].append(new_layer_acti)

    new_pytorch_model = ConvNet(new_model_descriptor)
    new_pytorch_model.cuda()
    new_pytorch_model._modules[str(new_id_bn)].momentum = 1.0
    new_pytorch_model._modules[str(new_id_bn)].eps = 0.0

    # test
    new_pytorch_model.forward(batch)

    if inheritance:
        new_pytorch_model = utils.InheritWeights(old_pytorch_model, new_pytorch_model)
        print("inheritance done")

    if id_mutation:
        # get weights fur id-convolution

        # PREPARING WEIGHTS FOR DEPTHWISE
        IDConv = conv2d_identity_sep(channels, kernel_size)

        # weight[0] - weights, weight[1] - bias
        bias_shape = new_pytorch_model._modules[str(new_id_conv)].depthwise.weight[1].size()[0]

        # save
        state_dict = {"weight": torch.from_numpy(IDConv).cuda(),
                      "bias": torch.from_numpy(np.zeros(shape=bias_shape)).cuda()}
        new_pytorch_model._modules[str(new_id_conv)].depthwise.load_state_dict(state_dict)

        #PREPARING WEIGHTS FOR POINTWISE
        IDConv = conv2d_identity(channels, 1)

        # weight[0] - weights, weight[1] - bias
        bias_shape = new_pytorch_model._modules[str(new_id_conv)].pointwise.weight[1].size()[0]

        # save
        state_dict = {"weight": torch.from_numpy(IDConv).cuda(),
                      "bias": torch.from_numpy(np.zeros(shape=bias_shape)).cuda()}
        new_pytorch_model._modules[str(new_id_conv)].pointwise.load_state_dict(state_dict)

        # receive outputs for all layers in order to calculate parameters for BN
        new_pytorch_model.forward(batch)

        predecessor_output_batch = new_pytorch_model.layerdic[str(new_id_conv)][0]
        predecessor_output_batch_cpu = predecessor_output_batch.cpu()
        predecessor_output_batch_data = predecessor_output_batch_cpu.data.numpy()

        batch_mean = np.mean(predecessor_output_batch_data, axis=(1, 2))

        batch_var = np.var(predecessor_output_batch_data, axis=(1, 2))

        eps = new_pytorch_model._modules[str(new_id_bn)].eps

        #gamma_ini = np.sqrt(batch_var + eps)
        beta_ini = batch_mean

        rm_copy = copy.deepcopy(new_pytorch_model._modules[str(new_id_bn)].running_mean)
        rv_copy = copy.deepcopy(np.sqrt(new_pytorch_model._modules[str(new_id_bn)].running_var + eps))

        # save
        state_dict = {"weight": nn.Parameter(rv_copy.cuda()),
                      "bias": nn.Parameter(rm_copy.cuda()),
                      "running_var": torch.from_numpy(batch_var).cuda(),
                      "running_mean": torch.from_numpy(beta_ini).cuda()}
        new_pytorch_model._modules[str(new_id_bn)].load_state_dict(state_dict)

        layers2burnin = []

    new_model = {'pytorch_model': new_pytorch_model,
                 'model_descriptor': new_model_descriptor,
                 'topo_ordering': new_pytorch_model.topo_ordering}

    return new_model, layers2burnin


