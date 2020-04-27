import tensorflow as tf
import math
import numpy as np

def AC_Block(input_tensor, name=None, 
             filters=None, kernel_size=None, strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform',
             momentum=0.99, epsilon=0.001, beta_initializer='zeros', gamma_initializer='ones',
             moving_mean_initializer='zeros', moving_variance_initializer='ones'):
    # this function takes in a tensor and outputs another tensor to be used in keras functional API
    if name == None:
        raise Exception('Please set the name of this convolution layer')
    if filters == None:
        raise Exception('Please set the number of filters this convolution layer')
    if kernel_size == None:
        raise Exception('Please set the kernel size of this convolution layer')
    if not isinstance(kernel_size, int):
        raise Exception('kernel size must be an integer')
        
    square_conv1_out = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=strides,
                                              padding=padding, kernel_initializer=kernel_initializer, use_bias=False, name=name+'_sqr')(input_tensor)
    ver_conv1_out = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,1), strides=strides,
                                       padding=padding, kernel_initializer=kernel_initializer,use_bias=False, name=name+'_ver')(input_tensor)
    hor_conv1_out = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,3), strides=strides,
                                       padding=padding, kernel_initializer=kernel_initializer,use_bias=False, name=name+'_hor')(input_tensor)

    square_conv1_out = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer = beta_initializer,
                                                          gamma_initializer = gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                                          moving_variance_initializer = moving_variance_initializer, name = name+'_sqr_bn')(square_conv1_out)
    ver_conv1_out = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer = beta_initializer,
                                                       gamma_initializer = gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                                       moving_variance_initializer = moving_variance_initializer, name = name+'_ver_bn')(ver_conv1_out)
    hor_conv1_out = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, beta_initializer = beta_initializer,
                                                       gamma_initializer = gamma_initializer, moving_mean_initializer=moving_mean_initializer,
                                                       moving_variance_initializer = moving_variance_initializer, name = name+'_hor_bn')(hor_conv1_out)

    added = tf.keras.layers.Add(name = name+'_add')([square_conv1_out, ver_conv1_out, hor_conv1_out])
    
    return added

def kernel_fusion(kernels, gammas, betas, means, var):
    # lists of parameters in the order of [square, vertical, horizontal]
    kernel_size = kernels[0].shape[0]
    
    square = (gammas[0] / np.sqrt(np.add(var[0], 1e-5))) * kernels[0]
    ver = (gammas[1] / np.sqrt(np.add(var[1], 1e-5))) * kernels[1]
    hor = (gammas[2] / np.sqrt(np.add(var[2], 1e-5))) * kernels[2]

    b = 0
    for i in range(3):
        b += -((means[i] * gammas[i]) / np.sqrt(np.add(var[i], 1e-5))) + betas[i]

    square[kernel_size//2-1:kernel_size//2+2, kernel_size//2, :, :] += np.squeeze(ver)
    square[kernel_size//2, kernel_size//2-1:kernel_size//2+2, :, :] += np.squeeze(hor)
    
    return square, b

def deploy(ckpt_file, AC_Block_names):
    # takes in a .h5 model file and a list of the names of AC Blocks used in that model 
    # perform kernel fusion and save a model in deploy mode

    if ckpt_file == None:
        raise Exception('Please pass in the trained model')
    if AC_Block_names == None:
        raise Exception('Please pass in the names of AC Blocks used')

    trained_model = tf.keras.models.load_model(ckpt_file)
    load_fused_model = False
    
    for conv_name in AC_Block_names:
        if load_fused_model == True:
            trained_model = tf.keras.models.load_model('fused_model.h5')
        sqr_weights = trained_model.get_layer(conv_name+'_sqr').get_weights()[0]
        ver_weights = trained_model.get_layer(conv_name+'_ver').get_weights()[0]
        hor_weights = trained_model.get_layer(conv_name+'_hor').get_weights()[0]
        sqr_bn_weights = trained_model.get_layer(conv_name+'_sqr_bn').get_weights()
        ver_bn_weights = trained_model.get_layer(conv_name+'_ver_bn').get_weights()
        hor_bn_weights = trained_model.get_layer(conv_name+'_hor_bn').get_weights()
        kernels = [sqr_weights, ver_weights, hor_weights]

        gammas = [sqr_bn_weights[0], ver_bn_weights[0], hor_bn_weights[0]]
        betas = [sqr_bn_weights[1], ver_bn_weights[1], hor_bn_weights[1]]
        means = [sqr_bn_weights[2], ver_bn_weights[2], hor_bn_weights[2]]
        var = [sqr_bn_weights[3], ver_bn_weights[3], hor_bn_weights[3]]

        fused_square, b = kernel_fusion(kernels, gammas, betas, means, var)
        
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        for layer in trained_model.layers:
            for node in layer.outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)
        network_dict['new_output_tensor_of'].update({trained_model.layers[0].name: trained_model.input})
        new_model_input = trained_model.input
        conv_input = trained_model.get_layer(conv_name+'_sqr').input
        conv_output = tf.keras.layers.Conv2D(filters=fused_square.shape[3], 
                                             kernel_size=(fused_square.shape[0], fused_square.shape[0]), 
                                             padding='same', use_bias=False, 
                                             weights = [fused_square], name = conv_name+'_fused')(conv_input)
        bias = np.zeros((conv_output.shape[1],conv_output.shape[2],conv_output.shape[3]))
        for channel in range(conv_output.shape[3]):
            for i in range(conv_output.shape[2]):
                for j in range(conv_output.shape[1]):
                    bias[j][i][channel] = b[channel]
        bias = tf.keras.backend.constant(bias)
        bias = tf.keras.backend.reshape(bias, [-1, conv_output.shape[1],conv_output.shape[2],conv_output.shape[3]])
        conv_output = tf.keras.layers.Add()([conv_output, bias])
        network_dict['new_output_tensor_of'].update({conv_name+'_add': conv_output})
        
        layer_names = [layer.name for layer in trained_model.layers]
        layer_idx = layer_names.index(conv_name+'_add')
        for layer in trained_model.layers[layer_idx+1:]:
            layer_input = []
            for layer_aux in network_dict['input_layers_of'][layer_names[layer_idx+1]]:
                layer_input.append(network_dict['new_output_tensor_of'][layer_aux])
            if len(layer_input) == 1:
                layer_input = layer_input[0]
            x = layer(layer_input)
            network_dict['new_output_tensor_of'].update({layer.name: x})
            layer_idx += 1
            
        trained_model = tf.keras.Model(new_model_input, x, name='fused_model')
        trained_model.save('fused_model.h5') 
        load_fused_model = True