#!/usr/bin/env python
"""
This is the python version of the script tools/expand_net
which expands templates layers to generate a network proto_file

Usage:
    expand_net net_proto_file_in net_proto_file_out
"""
import os
from google.protobuf import text_format
from caffe.proto import caffe_pb2

def join_path(cwd, path):
    result_path = os.path.normpath(os.path.join(cwd, path))
    result_path = result_path.replace(os.sep, '/')
    if result_path.startswith('/'):
        result_path = result_path[1:]
    return result_path

def expand_template(net, cwd=''):
    expanded_net = caffe_pb2.NetParameter()
    expanded_net.CopyFrom(net)
    expanded_net.ClearField('layers')
    tmp_layer_counter = 0
    for layer in net.layers:
        if not layer.HasField('name'):
            layer.name = 'temp_layer_%d' % tmp_layer_counter
            tmp_layer_counter += 1
        layer.name = join_path(cwd, layer.name)
        
        if layer.type == layer.TEMPLATE:
            if layer.HasField('template_param'):
                # load template and fill it with values
                with open(layer.template_param.source) as f:
                    layer_template = f.read()
                for var in layer.template_param.variable:
                    layer_template = layer_template.replace('${%s}' % var.name, 
                                                            var.value)
                # TODO: Keep missing value as its default
                
                # generate real network from the specialized template network
                sub_net = caffe_pb2.NetParameter()
                text_format.Merge(layer_template, sub_net)
                # TODO: Handle recursive definition
                net = expand_template(sub_net, layer.name)
                expanded_net.layers.MergeFrom(net.layers)
                layer_template = None
            else :
                raise Exception
        else:
            # change relative names into absolute ones
            for i, b in enumerate(layer.bottom):
                layer.bottom[i] = join_path(cwd, b)
            for i, t in enumerate(layer.top):
                layer.top[i] = join_path(cwd, t)
            
            expanded_net.layers.add().CopyFrom(layer)
            
    return expanded_net

def main(argv):
    if len(argv) != 3:
        print 'Usage: %s expand_net net_proto_file_in net_proto_file_out' % \
                os.path.basename(sys.argv[0])
    else:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(sys.argv[1]).read(), net)
        net = expand_template(net)
        print 'Expanding net to %s' % sys.argv[2]
        with open(sys.argv[2], 'w') as net_file:
            net_file.write(text_format.MessageToString(net))


if __name__ == '__main__':
    import sys
    main(sys.argv)

