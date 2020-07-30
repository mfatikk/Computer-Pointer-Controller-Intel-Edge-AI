
import os
import sys
import cv2
import time
import numpy as np

from argparse import ArgumentParser
import logging as log

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    '''
    Parse command line arguments.

    :return: command line arguments
    '''
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to Intel Pre Trained Face Detection model .xml file.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to Intel Pre Trained Facial Landmark Detection model .xml file.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to Intel Pre Trained Head Pose Estimation model .xml file.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to Intel Pre Trained Gaze Estimation model .xml file.")
                        
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
                        
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="targeted custom layers (CPU).")
                             
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.5,
                        help="Probability threshold for detection fitering.")
                        
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+', default=[],
                        help="Specify the flags from f, fl, hp, g to Visualize different model output on frame"
                             "f: Face Detection Model,       fl: Facial Landmark Detection Model"
                             "hp: Head Pose Estimation Model, g: Gaze Estimation Model")
    
    return parser



def main():

    args = build_argparser().parse_args()
    inputFilePath = args.input
    inputFeeder = None
    
    if args.input =="CAM":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(args.input):
            log.info("Unable to find specified video file")
            sys.exit(1)
        inputFeeder = InputFeeder("video",args.input)

    
    modelPathDict = {'FaceDetectionModel':args.face_detection_model, 'FacialLandmarksDetectionModel':args.facial_landmark_model, 
    'GazeEstimationModel':args.gaze_estimation_model, 'HeadPoseEstimationModel':args.head_pose_model}
    

    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            log.info("Unable to find specified "+fileNameKey+" xml file")
            sys.exit(1)
    
    fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')


    start_time_1= time.time()
    inputFeeder.load_data()
    
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    total_model_load_time= (time.time()-start_time_1)
    print("Model Load Time: {:.3f}".format(total_model_load_time))
    
    frame_count = 0
    start_time = time.time()

    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(450,450)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            log.info("Unable to detect the face.")
            if key==27:
                break
            continue
        

        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)

        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
            break
    log.info("VideoStream has ended.")
    cv2.destroyAllWindows()
    inputFeeder.close()

    total_time = time.time() - start_time
    total_inference_time=total_time
    fps=frame_count/total_inference_time
    print("Inference Time: {:.3f}".format(total_inference_time))
    print("FPS: {}".format(fps))
   

if __name__ == '__main__':
    main() 
 
