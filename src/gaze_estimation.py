'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import math


class GazeEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
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
            
        
    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_preprocessed_img, re_preprocessed_img = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_preprocessed_img, 'right_eye_image':re_preprocessed_img})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)
        
        return new_mouse_coord, gaze_vector
         
        
        
    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        
        le_preprocessed_img = cv2.resize(left_eye, (60, 60))
        le_preprocessed_img = le_preprocessed_img.transpose((2, 0, 1))
        le_preprocessed_img = le_preprocessed_img.reshape(1, *le_preprocessed_img.shape)
        
        re_preprocessed_img = cv2.resize(right_eye, (60, 60))
        re_preprocessed_img = re_preprocessed_img.transpose((2, 0, 1))
        re_preprocessed_img = re_preprocessed_img.reshape(1, *re_preprocessed_img.shape)


        return le_preprocessed_img, re_preprocessed_img
    
    def preprocess_output(self, outputs,hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        gaze_vector = outputs[self.output_names][0]

        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        x_value = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        y_value = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue

        return (x_value,y_value), gaze_vector