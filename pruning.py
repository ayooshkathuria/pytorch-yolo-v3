import torch
import torch.nn as nn
from darknet import Darknet
import copy
import os


def load_pruned_info(path='output/pruned_channels.txt'):
    pruned_list = []
    with open(path, 'r') as file:
        content = file.readlines()

    layer_info = {}
    for line in content:
        line = line.strip()
        if not line == '-' * 50:
            k, v = line.split("=")
            k = k.strip()
            v = v.strip()
            layer_info[k] = v
        else:
            pruned_list.append(layer_info)
            layer_info = {}

    return pruned_list


def save_pruned_filters(layer_index_in_module_list, bn, path='output/pruned_channels.txt'):
    """
    this function saves the indices of pruned filters of each layer to a txt
    :param path: the path
    :return: None
    """
    with open(path, 'a+') as file:
        file.write("layer = {}\n".format(layer_index_in_module_list))
        file.write("batch_norm = {}\n".format(bn))
        file.write("pruned_filters = ")
        file.write(', '.join([str(x) for x in removed_index]) + "\n")
        file.write("filters_after_pruning = " + blocks[layer_index_in_module_list + 1]['filters'] + "\n")
        file.write("-" * 50 + "\n")


def prune_conv(layer_index_in_module_list, bn=True, being_routed=False):
    """
    prunes conv wit batch_norm
    :param layer_index_in_module_list:
    :param filter_list:
    :param model:
    :return:
    """
    route_to_where = []
    if being_routed:
        for router in routers:
            if layer_index_in_module_list in router['routed']:
                route_to_where.append(router['routeto'])
    layer = net_struct[layer_index_in_module_list][0]
    list_to_tensor = torch.LongTensor([x for x in range(layer.out_channels) if x not in removed_index])

    # copy the weight of conv and batch_norm
    if bn == False:
        layerbias = copy.deepcopy(layer.bias.data)
        pruned_bias = torch.index_select(layerbias, 0, list_to_tensor)

    layerweight = copy.deepcopy(layer.weight.data)
    pruned_weight = torch.index_select(layerweight, 0, list_to_tensor)

    net_struct[layer_index_in_module_list][0] = nn.Conv2d(in_channels=layer.in_channels,
                                                          out_channels=layer.out_channels - len(removed_index),
                                                          kernel_size=layer.kernel_size, stride=layer.stride,
                                                          bias=not bn,
                                                          padding=layer.padding)
    net_struct[layer_index_in_module_list][0].weight.data.copy_(pruned_weight)

    if bn:
        batch_norm = net_struct[layer_index_in_module_list][1]
        bn_weight = copy.deepcopy(batch_norm.weight.data)
        bn_bias = copy.deepcopy(batch_norm.bias.data)
        bn_rm = copy.deepcopy(batch_norm.running_mean.data)
        bn_var = copy.deepcopy(batch_norm.running_var.data)

        # prune the bn weight
        p_bn_weight = torch.index_select(bn_weight, 0, list_to_tensor)
        p_bn_bias = torch.index_select(bn_bias, 0, list_to_tensor)
        p_bn_rm = torch.index_select(bn_rm, 0, list_to_tensor)
        p_bn_var = torch.index_select(bn_var, 0, list_to_tensor)

        net_struct[layer_index_in_module_list][1] = nn.BatchNorm2d(list_to_tensor.size()[0])
        net_struct[layer_index_in_module_list][1].bias.data.copy_(p_bn_bias)
        net_struct[layer_index_in_module_list][1].weight.data.copy_(p_bn_weight)
        net_struct[layer_index_in_module_list][1].running_mean.copy_(p_bn_rm)
        net_struct[layer_index_in_module_list][1].running_var.copy_(p_bn_var)

    else:
        net_struct[layer_index_in_module_list][0].bias.data.copy_(pruned_bias)

    #################################################################
    """
    after the pruning of weight and model, now let's modify the cfg to change the network structure,
    and then write to a new cfg file
    """
    ######################

    # Wait, it seems that there's still other work to do
    # Don't be hurry to write the cfg
    # Since we pruned a convolutional layer, its output channels has reduced.
    # We also need to prune the input channels of the next convolutional layer
    # However, there are several layers do not have any parameters to prune, we just skip them in the outer loop.
    # They are [maxpool], [upsample], [yolo] and [route]
    # Though [route] layer do not have weights, it's a little tricky to process it. We will deal with it elsewhere.
    # Also [yolo] layer means detection and forward propagation ends here, so be careful with it.

    ############################################################
    # Now we prune the inputs of next layer.
    next_conv_index = layer_index_in_module_list + 1
    prune_flag = True
    while blocks[next_conv_index + 1]['type'] != 'conv' and blocks[next_conv_index + 1]['type'] != 'convolutional':
        if blocks[next_conv_index + 1]['type'] == 'yolo' or blocks[next_conv_index + 1]['type'] == 'route':
            prune_flag = False
            break
        next_conv_index += 1

    if prune_flag:
        next_layer = net_struct[next_conv_index][0]
        print("I am the next layer to be pruned: " + str(next_layer))

        has_bn = blocks[next_conv_index + 1].get('batch_normalize', '0')
        if int(has_bn) == 1:
            has_bn = True
        else:
            has_bn = False
            next_bias = copy.deepcopy(next_layer.bias.data)

        next_weight = copy.deepcopy(next_layer.weight.data)
        pruned_next_weight = torch.index_select(next_weight, 1, list_to_tensor)
        net_struct[next_conv_index][0] = nn.Conv2d(in_channels=next_layer.in_channels - len(removed_index),
                                                   out_channels=next_layer.out_channels,
                                                   kernel_size=next_layer.kernel_size, bias=not has_bn,
                                                   stride=next_layer.stride, padding=next_layer.padding)
        net_struct[next_conv_index][0].weight.data.copy_(pruned_next_weight)
        if not has_bn:
            net_struct[next_conv_index][0].bias.data.copy_(next_bias)
    else:
        print("Can not prune layer: {}".format(next_conv_index))

    if being_routed:
        pruned_info = load_pruned_info()
        for router in routers:
            routed_list = router['routed']
            routeto = router['routeto']

            if layer_index_in_module_list in routed_list:

                if len(routed_list) == 1:
                    route_to_layer = net_struct[routeto][0]
                    print("Hey I am the route to layer: " + str(route_to_layer))
                    route_to_bn = blocks[routeto + 1].get('batch_normalize', '0')
                    if int(route_to_bn) == 0:
                        route_to_bias = copy.deepcopy(route_to_layer.bias.data)
                        route_to_bn = False
                    elif int(route_to_bn) == 1:
                        route_to_bn = True

                    route_to_weight = copy.deepcopy(route_to_layer.weight.data)
                    pruned_route_to_weight = torch.index_select(route_to_weight, 1, list_to_tensor)
                    net_struct[routeto][0] = nn.Conv2d(in_channels=route_to_layer.in_channels - len(removed_index),
                                                       out_channels=route_to_layer.out_channels,
                                                       kernel_size=route_to_layer.kernel_size, bias=not route_to_bn,
                                                       stride=route_to_layer.stride, padding=route_to_layer.padding)
                    net_struct[routeto][0].weight.data.copy_(pruned_route_to_weight)
                    if not route_to_bn:
                        net_struct[routeto][0].bias.data.copy_(route_to_bias)

                elif len(routed_list) == 2:

                    if layer_index_in_module_list == routed_list[0]:
                        cat_filters_num = int(blocks[routed_list[0] + 1]['filters']) + int(
                            blocks[routed_list[1] + 1]['filters'])
                        route_to_list_tensor = torch.LongTensor(
                            [x for x in range(cat_filters_num) if x not in removed_index])

                        to_layer = net_struct[routeto][0]
                        to_layer_weight = copy.deepcopy(to_layer.weight.data)
                        pruned_to_layer_weight = torch.index_select(to_layer_weight, 1, route_to_list_tensor)

                        has_to_bn = blocks[routeto + 1].get('batch_normalize', '0')
                        if int(has_to_bn) == 0:
                            has_to_bn = False
                            to_layer_bias = copy.deepcopy(to_layer.bias.data)
                        elif int(has_to_bn) == 1:
                            has_to_bn = True

                        net_struct[routeto][0] = nn.Conv2d(in_channels=to_layer.in_channels - len(removed_index),
                                                           out_channels=to_layer.out_channels,
                                                           kernel_size=to_layer.kernel_size, bias=not has_to_bn,
                                                           stride=to_layer.stride, padding=to_layer.padding)
                        net_struct[routeto][0].weight.data.copy_(pruned_to_layer_weight)

                        if not has_to_bn:
                            net_struct[routeto][0].bias.data.copy_(to_layer_bias)

                    elif layer_index_in_module_list == routed_list[1]:
                        first_layer_pruned_filters = []
                        for pruned_layer in pruned_info:
                            pruned_layer_index = int(pruned_layer['layer'])
                            if routed_list[0] == pruned_layer_index:
                                pruned_filters = pruned_layer['pruned_filters'].split(',')
                                pruned_filters = [int(x.strip()) for x in pruned_filters]
                                first_layer_pruned_filters.extend(pruned_filters)
                                break
                        if len(first_layer_pruned_filters) != 0:
                            filters_num_before_pruning = int(pruned_layer['filters_num_before_pruning'])
                        else:
                            filters_num_before_pruning = int(blocks[routed_list[0] + 1]['filters'])

                        start_index = filters_num_before_pruning - len(first_layer_pruned_filters)
                        cat_two_layer_channels = int(blocks[routed_list[0] + 1]['filters']) + int(
                            blocks[routed_list[1] + 1]['filters'])
                        second_remove_index = [(x + start_index) for x in removed_index]
                        to_prune_list_tensor = torch.LongTensor(
                            [x for x in range(cat_two_layer_channels) if x not in second_remove_index])

                        dest_layer = net_struct[routeto][0]
                        dest_weight = copy.deepcopy(dest_layer.weight.data)
                        pruned_dest_weight = torch.index_select(dest_weight, 1, to_prune_list_tensor)

                        dest_bn = blocks[routeto + 1].get('batch_normalize', '0')
                        if int(dest_bn) == 1:
                            dest_bn = True
                        elif int(dest_bn) == 0:
                            dest_bn = False
                            dest_bias = copy.deepcopy(dest_layer.bias.data)
                        print(pruned_dest_weight.size())
                        print(net_struct[routeto][0])
                        print(removed_num)
                        print(len(removed_index))
                        net_struct[routeto][0] = nn.Conv2d(in_channels=dest_layer.in_channels - len(removed_index),
                                                           out_channels=dest_layer.out_channels,
                                                           kernel_size=dest_layer.kernel_size, bias=not dest_bn,
                                                           stride=dest_layer.stride, padding=dest_layer.padding)
                        print(net_struct[routeto][0])
                        print(pruned_dest_weight.size())
                        net_struct[routeto][0].weight.data.copy_(pruned_dest_weight)

                        if not dest_bn:
                            net_struct[routeto][0].bias.data.copy_(dest_bias)
    model.blocks[layer_index_in_module_list + 1]['filters'] = str(
        int(blocks[layer_index_in_module_list + 1]['filters']) - removed_num)


if __name__ == '__main__':

    which_layer_to_prune = 21

    file_list = os.listdir('output')
    file_name = [x for x in file_list if x.endswith('cfg')]
    iters = []
    for file in file_name:
        name, _ = file.split('.')
        iters.append(int(name[10:]))

    read_index = max(iters)

    print("Restoring process from iteration {}".format(read_index))
    model = Darknet('output/pruned_cfg{}.cfg'.format(read_index))
    model.load_weights('output/pruned_weights{}.weights'.format(read_index))
    # fm1 = torch.zeros((5, 3, 3))
    # fm2 = torch.ones((3, 3, 3))
    # print(torch.cat((fm1, fm2), 0))
    # for el in model.blocks:
    #     print(el)
    # model.save_pruned_cfg()
    # model.save_weights()
    # a=np.array([[[1,2,3],[4,-5,6]],[[-7.9,8,9],[-10.77,11,12]],[[13,14,15],[16,17,18]]])
    # b=torch.from_numpy(a)
    # c=torch.Tensor.sum(b)
    # print("-----------------------\n",c)
    # b = model.module_list[0][1].bias.data
    # print(b.size())
    # print(b)
    # # index=torch.LongTensor([0])
    # # c=torch.index_select(b,0,index)
    # # print(c)

    # model.save_weights('output.weights')

    # a list that stores all the squential moudles of the network
    net_struct = model.module_list
    # print(net_struct[0])
    # net_struct[0][0] = nn.Linear(2, 3)
    # print(model.module_list[0])
    # print(net_struct[0])
    # a list of dict that generated from cfg, store all the layers of the network
    blocks = model.blocks

    ##############################################################
    # search for all [route] layers
    routers = []
    layers_being_routed = []
    for i, ablock in enumerate(blocks):
        if ablock['type'] == "route":
            router = {}
            router['index'] = i - 1
            routed_index = []
            for idex in ablock['layers']:
                idex = int(idex)
                if idex > 0:
                    real_idex = idex
                    while blocks[real_idex + 1]['type'] != 'conv' and blocks[real_idex + 1]['type'] != 'convolutional':
                        real_idex -= 1
                    routed_index.append(real_idex)
                    layers_being_routed.append(real_idex)

                else:
                    real_idex = i - 1 + idex
                    while blocks[real_idex + 1]['type'] != 'conv' and blocks[real_idex + 1]['type'] != 'convolutional':
                        real_idex -= 1
                    routed_index.append(real_idex)
                    layers_being_routed.append(real_idex)
            router['routed'] = routed_index
            router['routeto'] = i
            routers.append(router)
    ##############################################################

    #
    ############################
    # now we prune the first conv layer according to the abs sum of filters
    # how to choose factor m???? well  let's try out if 20% will work
    removed_rate = 0.2
    for i in range(which_layer_to_prune + 1, which_layer_to_prune + 2):
        layercfg = blocks[i]

        # get current layer to prune its filters
        layer = net_struct[i - 1]

        # find next conv layer to prune its channels

        if (layercfg['type'] == "conv" or layercfg['type'] == 'convolutional') and blocks[i + 1]['type'] != 'yolo':
            print("Now pruning the layer {} Current layer type: {}".format(i - 1, blocks[i]['type']))
            removed_index = []
            filters_num = int(layercfg['filters'])
            removed_num = round(filters_num * removed_rate)
            has_bn = layercfg.get('batch_normalize', '0')
            if int(has_bn) == 1:
                has_bn = True
            else:
                has_bn = False

            conv = layer[0]
            convweight = conv.weight.data

            # how I get the smallest m% filters? how to get their indices?
            abs_sum_list = []
            for j in range(filters_num):
                filter_abs_sum = torch.sum(torch.abs(convweight.data[j]))
                abs_sum_list.append(filter_abs_sum.item())

            # then select the smallest, but how
            # do iterations over and over? well that's the simplest way
            sorted_sum_list = sorted(abs_sum_list)
            for k in range(removed_num):
                removed_index.append(abs_sum_list.index(sorted_sum_list[k]))

            # do the pruning
            prune_conv(i - 1, has_bn, i - 1 in layers_being_routed)
            # save the pruned filters
            save_pruned_filters(i - 1, has_bn)
            # save the config
            model.save_pruned_cfg('output/pruned_cfg{}.cfg'.format(i))
            # save the weight
            model.save_weights('output/pruned_weights{}.weights'.format(i))

        else:
            if (blocks[i]['type'] == 'conv' or blocks[i]['type'] == 'convolutional'):
                print("Skip layer {} because {} layer behind it. Current layer type: {}".format(i - 1,
                                                                                                blocks[i + 1]['type'],
                                                                                                blocks[i]['type']))
            else:
                print(
                    "Skip layer {} Current Layer type: {}".format(i - 1, blocks[i]['type']))
