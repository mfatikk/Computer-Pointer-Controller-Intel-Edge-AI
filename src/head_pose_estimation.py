'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class HeadPoseEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''

        self.device = device
        self.model_name = model_name
        self.extensions = extensions
        self.network = None
        self.plugin = None
        self.exec_net = None
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.output_shape = None


        self.network = IENetwork(model=self.model_structure, weights=self.model_weights)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def load_model(self):

        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        self.plugin = IECore()

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            if not self.extensions==None:
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("Unsupported layers found.")
                    exit(1)



        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        preprocessed_img = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:preprocessed_img})
        final_output = self.preprocess_output(outputs)
        return final_output


    def preprocess_input(self, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_img = np.transpose(np.expand_dims(preprocessed_img,axis=0), (0,3,1,2))
        return preprocessed_img

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        outs = []
        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])

        return outs

