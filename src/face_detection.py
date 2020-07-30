'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import cv2
import numpy as np

class FaceDetectionModel:
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
        


    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        preprocessed_img = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:preprocessed_img})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0] 
        height=image.shape[0]
        width=image.shape[1]
        
        coords=[int(coords[0]*width),int(coords[1]*height),int(coords[2]*width),int(coords[3]*height)]
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face, coords

    def preprocess_input(self, image):
        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_img = preprocessed_img.transpose((2,0,1))
        preprocessed_img = preprocessed_img.reshape(1, *preprocessed_img.shape)

        return preprocessed_img

    def preprocess_output(self, outputs, prob_threshold):

        area = []
        coords = []
        for id, label, confidence, x_min, y_min, x_max, y_max in outputs[self.output_names][0][0]:
            if confidence > prob_threshold:
                width = x_max - x_min
                height = y_max - y_min
                area.append(width * height)
                coords.append([x_min, y_min, x_max, y_max])

                
        return coords
