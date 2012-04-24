import cPickle
import sys
from PIL import Image
import numpy as np
import cv
import cv2
import tools
import logging
import parameter


def safe_data(data):
    """pickle data object"""

    logging.debug('Start saving Data into %s (Pickle)' % parameter.pickle_filename)    
    convert_for_pickle(data) # convert data before pickle

    try:
        with open (parameter.pickle_filename, 'wb') as trained_data:
            cPickle.dump(data, trained_data)
        logging.info('Data successfully stored in actual folder as: %s (persistence.py)' % parameter.pickle_filename)

    except TypeError, msg:
        logging.error('Error in saving data (persistence.py). Error message: %s' % msg)
        sys.exit()

    logging.debug('Data is converted into initial format for further processing (persistence.py)')    
    convert_after_pickle(data) # reconvert data, because the actual data object is converted and could not be used, when it will used later in the program
    

def load_data():
    """unpickle data object"""
    
    try:
        with open (parameter.pickle_filename, 'rb') as trained_data:
            data = cPickle.load(trained_data)
        logging.info('Data successfully loaded from %s (persistence.py)' % parameter.pickle_filename)
    except IOError, msg:
        logging.error('No trained data found. Check file %s (persistence.py). Error message: %s' %(parameter.pickle_filename, msg))
        sys.exit()
    except TypeError, msg:
        logging.error('Error in loading data (persistence.py). Error message: %s' %msg)
        sys.exit()
        
    convert_after_pickle(data)
    return data


def convert_for_pickle(data):
    """ converts data for pickle into necessary format:
    data.image: from cv2.cv.iplimage to numpy.ndarray
    data.facedata.face: from cv2.cv.cvmat to numpy.ndarray
    data.facedata.keypoint: from type Keypoint to string"""

    for i in range(len(data)):
        logging.debug('Start convert file %s' % data[i].filename)
        #convert image to numpy array if exists
        try:
            if hasattr(data[i], 'image'):
                if type(data[i].image) == cv2.cv.iplimage:
                    logging.debug('Start convert image from format iplimage to numpy-array (persistence.py)')
                    data[i].image = tools.cv2array(data[i].image)
        except AttributeError, msg:
            logging.error('Error in Converting Image. Maybe there is no image in file %s (persistence.py). Error message: ' % (data[i].filename, msg))
            
        #convert face if exists
        try:
            if data[i].facedata: # check if this image has faces with corresponding data
                for j in range(len(data[i].facedata)):                  
                    #convert faces to numpy arrays (ich brauch fuer Matching die faces als ndarray, daher werden diese beim unpickeln nicht mehr zurueck konvertiert)
                    if type(data[i].facedata[j].face) == cv2.cv.cvmat:
                        logging.debug('Start convert face number %s from format cvmat to numpy-array (persistence.py)' % str(j+ 1))
                        data[i].facedata[j].face = np.array(data[i].facedata[j].face)
                        
        except AttributeError, msg:
            logging.info('Error in Converting Face. Maybe there is no face in file %s (persistence.py). Error message: %s' % (data[i].filename, msg))
            
        #convert keypoint to string
        try:
            if data[i].facedata: # check if this image has faces with corresponding data
                for j in range(len(data[i].facedata)):
                    if not data[i].facedata[j].keypoints:
                        logging.warn('No Keypoint in Nr %d from file %s (persistence.py)' % ((j+1), data[i].filename))
                    else:
                        logging.debug('Start converting keypoints for face %s (persistence.py)' % str(j+ 1))
                        for k in range(len(data[i].facedata[j].keypoints)):                                                  
                            data[i].facedata[j].keypoints[k] = keypoint2str(data[i].facedata[j].keypoints[k])
                     
        except AttributeError, msg:
            logging.info('Error in Converting Keypoints. Maybe there is no keypoint in file %s (persistence.py). Error message: %s' % (data[i].filename, msg))

    return data


def convert_after_pickle(data):    
    """ converts data back into original type if further processed:
    data.image: from  numpy.ndarray to cv2.cv.iplimage  
    data.facedata.face: not converted because face is needed as type numpy.ndarray 
    data.facedata.keypoint: from  string to typ Keypoint"""
        
    for i in range(len(data)):
        logging.debug('Start convert file %s ' % data[i].filename)
        #convert numpy array to iplimage
        try:
            if hasattr(data[i], 'image'):
                if type(data[i].image) == np.ndarray:
                    logging.debug('Start convert image from format numpy-array to iplimage(persistence.py)')
                    data[i].image = tools.array2cv(data[i].image)

        except AttributeError, msg:
            logging.debug('Error converting image: No image-data found or wrong format(persistence.py). Error message: %s' % msg)
            sys.exit()
            
        # convert the keypoints from string to keypoint-type
        if hasattr(data[i], 'facedata'):
            for j in range(len(data[i].facedata)):

                if hasattr(data[i].facedata[j], 'keypoints'):
                    logging.debug('Start convert keypoints for face Nr %s (persistence.py)' % str(j+ 1))
                    for k in range(len(data[i].facedata[j].keypoints)):
                        data[i].facedata[j].keypoints[k] = str2keypoint(data[i].facedata[j].keypoints[k])
                else:
                    logging.debug('No Keypoints found in file %s (persistence.py)' % data[i].filename)
    
        else:
            logging.debug('No facedata found in file %s (persistence.py)' % data[i].filename)

def keypoint2str(keypoint):
    """ convert OpenCV Keypoints into string to pickle the data"""
    keypoint_str = []
    keypoint_str.append(float(keypoint.pt[0]))
    keypoint_str.append(float(keypoint.pt[1]))
    keypoint_str.append(float(keypoint.size))
    keypoint_str.append(float(keypoint.angle))
    keypoint_str.append(float(keypoint.response))
    keypoint_str.append(float(keypoint.octave))
    keypoint_str.append(float(keypoint.class_id))    
    return keypoint_str

def str2keypoint(keypoint_str):
    """ convert OpenCV Keypoints from string into keypoint-type to process
    after unpickle the data"""
    keypoint = cv2.KeyPoint(float(keypoint_str[0]), float(keypoint_str[1]), float(keypoint_str[2]),
            float(keypoint_str[3]), float(keypoint_str[4]), int(keypoint_str[5]),
            int(keypoint_str[6]), )
    return keypoint
