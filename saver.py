import yaml
import units
import all2all
import logging as log
import zlib

import base64
import struct

import numpy as np
from numpy.core.fromnumeric import shape

#TODO(EBulychev): python float -> float32, write to file matrix like base64-array

activation_functions_dict = ['tanh', 'softmax', 'step_function', 'linear_combination', 'log-sigmoid_function']

neural_network_types = ['Feedforward neural network', 'Radial basis function (RBF) network', 'Kohonen self-organizing network',
                        'Learning Vector Quantization', 'Fully recurrent network', 'Hopfield network', 'Boltzmann machine',
                        'Simple recurrent networks', 'Echo state network', 'Long short term memory network', 'Bi-directional RNN',
                        'Hierarchical RNN', 'Stochastic neural networks', 'Committee of machines', 'Associative neural network (ASNN)',
                        'Holographic associative memory', 'Instantaneously trained networks', 'Spiking neural networks',
                        'Dynamic neural networks', 'Cascading neural networks', 'Neuro-fuzzy networks', 'Compositional pattern-producing networks',
                        'One-shot associative memory']



class NeuralNetworkLayer(object):   
     
    def __init__(self, layer_number, height_for_init, width_for_init, act_func):
        self.layer_number = layer_number
        self.activation_function = act_func
        self.activation_function_descr = activation_functions_dict[act_func]
        self.weight = [[ ] ]
        
    def set_activation_function(self,act_func):
            self.activation_function = act_func
            self.activation_function_descr = activation_functions_dict[act_func]
                
    def set_weight(self,weight_vector):
        
        self.height = len(weight_vector)//len(weight_vector[0])
        self.width = len(weight_vector[0])
        log.info('weight column: %d' % self.width)
        log.info('weight row: %d' % self.height)

        self.weight = np.empty(dtype=np.float32, shape=[self.width * self.height])        
        for x in range(self.width):
            for y in range(self.height):
                self.weight[x * self.height + y] = np.float32(weight_vector[y][x])
                      
        compressed_matrix = zlib.compress(self.weight, 9)
        log.info("Compressed weight to %d bytes" % compressed_matrix.__sizeof__())        
        self.weight = base64.b64encode(compressed_matrix)
        log.info("weight base64 encoded to %d bytes" % self.weight.__sizeof__())
                
    def set_bias(self,bias_vector):
#         self.bias = [  "%.7f" % float(bias_vector[y]) for y in range(len(bias_vector))]
#         self.bias = []
        self.bias = np.empty(dtype=np.float32, shape=[len(bias_vector)])
        for x in range(len(bias_vector)):
            self.bias[x] = (np.float32(bias_vector[x]))
        
        compressed_matrix = zlib.compress(self.bias, 9)
        log.info("Compressed bias to %d bytes" % compressed_matrix.__sizeof__())
        self.bias = base64.b64encode(compressed_matrix)
        log.info("bias base64 encoded to %d bytes" % self.bias.__sizeof__())
#         s = struct.Struct('%sf' % len(self.bias))
#         packed_data = s.pack(*self.bias )
#         print('Lenght P:', len(self.bias) )
#         self.bias = base64.b64encode(packed_data) 
#         
#         self.bias = np.empty(dtype=np.float32, shape=[len(bias_vector)])

class NeuralNetwork(object):
       
    def __init__(self, network_type, number_of_layers, width_for_init, height_for_init, act_func):
        self.service_info = 'yaml_test' 
        self.layers_number = number_of_layers
        self.layers = []
        self.neural_network_type = network_type
        self.neural_network_type_desc = neural_network_types[network_type]
        
        for i in range(self.layers_number):
            #ToDo: read here information about every layer
            NNlayer = NeuralNetworkLayer(i, height_for_init, width_for_init, act_func)
            self.layers.append(NNlayer)
        
        print('NeuralNetwork is successful initialized')
            
    
    def print_to_yaml(self, yaml_name):
        stream = open(yaml_name, 'w')    
        yaml.dump(self, stream)    
#print(yaml.dump(self))
        stream.close()
#         yaml.dump(NeuralNetwork(self.layers_number,
#                                 self.neural_network_type,
#                                 self.layers[0].height,
#                                 self.layers[0].width,
#                                 self.layers[0].activation_function), stream )
#         print(yaml.dump(NeuralNetwork(self.layers_number,
#                                 self.neural_network_type,
#                                 self.layers[0].height,
#                                 self.layers[0].width,
#                                 self.layers[0].activation_function)))
        
class SaverUnit(units.Unit):
    """SaverUnit to save All2All weight matrix and bias after signal from decision unit

    Should be assigned before initialize():        

    Updates after run():        

    Creates within initialize():
        stream to file 

    Attributes:
        forward: list of All2All units.
    """
    def __init__(self, forward, unpickling=0): 
        super(SaverUnit, self).__init__(unpickling=unpickling)       
        self.forward = forward

    def initialize(self):
        pass

    def run(self):
        network = NeuralNetwork(0, # Type of neural network. How can it be understand?
                                len(self.forward), # Number of layers
                                self.forward[0].weights.v.shape[0], # Width for weight matrix initialization 
                                self.forward[0].weights.v.shape[1], # Height for weight matrix initialization
                                0) # Activation function  
        for i in range(len(self.forward)):
            network.layers[i].set_weight(self.forward[i].weights)
            network.layers[i].set_bias(self.forward[i].bias)
            if type(self.forward[i]) == all2all.All2AllSoftmax :
                network.layers[i].set_activation_function(1)
            elif type(self.forward[i]) == all2all.All2AllTanh :
                network.layers[i].set_activation_function(0)
                    
        yaml_name = 'default.yaml'
        network.print_to_yaml(yaml_name)  
        
        # Compress
        yaml_compress_name = 'default_compress.yaml'
        str_object1 = open(yaml_name, 'rb').read()
        str_object2 = zlib.compress(str_object1, 9)
        f = open(yaml_compress_name, 'wb')
        f.write(str_object2)
        f.close()
        
        