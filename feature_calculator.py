import cv
import cv2
import tools
import logging
import parameter
import numpy as np
from PIL import Image
import sift

class feature_calculator:

    def detect_features(self, faces, image_filename):
        """ Detects Features in an list of faces and returns the images """

        logging.debug('start detect %s features for file %s (train.py)' %(parameter.description_method, image_filename))       
        keypoints = []
        descriptors = []

        # detect features and save 
        for i in range(len(faces)):
            if not parameter.face_enlargement == None:
                faces[i] = tools.enlarge_image(faces[i], parameter.face_enlargement)
                logging.debug('Cropped face from file %s has been enlarged with factor %.3f  (surf-detector.py)' % (image_filename, parameter.face_enlargement))

            face_numpy = np.asarray(faces[i]) # convert image for further processing

            # compute surf calculation    
            if parameter.description_method == 'surf':               
                surf = cv2.SURF(parameter.hessian_threshold, parameter.nOctaves, parameter.nOctaveLayers) # threshold, number of octaves, number of octave layers within each octave (http://opencv.itseez.com/modules/features2d/doc/feature_detection_and_description.html, http://www.mathworks.de/help/toolbox/vision/ref/detectsurffeatures.html)
                tmpkeypoints, tmpdescriptors = surf.detect(face_numpy, None, False) # extracting the SURF keys
                if len(tmpdescriptors) == 0:
                    logging.warn('No descriptors found for a face in file %s (surf-detector.py)' % (image_filename))
                else:
                    tmpdescriptors.shape = (-1, surf.descriptorSize()) # change the shape of the descriptor from 1-dim to 2-dim (notwendig, damit die Funktionen - match_bruteforce - bei der Suche funktionieren)
                    logging.info('%d Features found in file %s: face number %d (surf-detector.py)' % (len(tmpdescriptors), image_filename, (i+1)))

            # compute sift calculation    
            if parameter.description_method == 'sift':
                cv.SaveImage('tmp-sift.jpg', faces[i])
                sift.process_image('tmp-sift.jpg',"tmp.sift")
                
                l1,tmpdescriptors = sift.read_features_from_file("tmp.sift")
                tmpkeypoints = []
                if tmpdescriptors == None:
                    logging.warn('No descriptors found for a face in file %s (surf-detector.py)' % (image_filename))
                else:
                    for j in range(len(l1)):    
                        keypoint = cv2.KeyPoint(l1[j][0], l1[j][1], l1[j][2], l1[j][3])
                        tmpkeypoints.append(keypoint)
                    logging.info('%d Features found in file %s: face number %d (surf-detector.py)' % (len(tmpdescriptors), image_filename, (i+1)))

            keypoints.append(tmpkeypoints) # add keypoints do list even when none are found
            descriptors.append(tmpdescriptors) # add descriptors do list even when none are found
            
        return(keypoints, descriptors)
