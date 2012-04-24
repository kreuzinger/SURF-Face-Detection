import tools
from face_detector import face_detector
from feature_calculator import feature_calculator
from image_data import Image_Data
from image_data import Face_Data
import sys
import logging
import parameter


def get_face_data(image_filenames):
    """ Detects all faces and returns the cropped images"""

    logging.debug('Start method "get_face_data" (train.py)')
    logging.debug('%s images found (train.py)' % len(image_filenames))
    if len(image_filenames) == 0:
            logging.error('No image found in trainfolder (train.py)')
            sys.exit()
    
    fd = face_detector(); fc = feature_calculator() #create new instances for detector and calculator
    data = [] # container for image data
    
    for i in range(len(image_filenames)):
        data.append(Image_Data(image_filenames[i]));
        # get coordinates for faces and crop faces
        image, i_coordinates = fd.detect_faces(image_filenames[i])    # detect faces with coordinates

        if not i_coordinates: # if no face found, continue with next image
            logging.warn('No faces found in file %s (train.py)' % image_filenames[i])
            continue
       
        cropped_faces = fd.crop_face(image, i_coordinates, image_filenames[i]) # crop faces from images       
        # detect features with feature_detector
        (keypoints, descriptors) = fc.detect_features(cropped_faces, image_filenames[i]) 

        # save data including image-data
        if parameter.save_images == 1:
            data[i].add_image_data(i_coordinates, image) # save to data-container
        else: # save data without image-data
            if parameter.save_images != 0:
                logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (train.py)')
            data[i].add_image_data(image_filenames[i], i_coordinates) # save to data-container
 
        # save face-data
        face_data = [] # container for face-data
        for j in range(len(keypoints)):
            if len(keypoints[j]) > parameter.min_features: # only save data with minimal amount of features
                if parameter.save_images == 1: # only save face-image, when parameter says 1 (for performance)
                    face_data.append(Face_Data(keypoints[j], descriptors[j], (cropped_faces[j].rows, None) ,cropped_faces[j])) # create container for face data
                else: # save face-data without face-image
                    if parameter.save_images != 0:
                        logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (train.py)')
                    face_data.append(Face_Data(keypoints[j], descriptors[j],(cropped_faces[j].rows, None))) # create container for face data
                data[i].add_face_data(face_data) # add face data for every found face to corresponding image                    
            else:
                logging.warn('Face deleted in File %s (not enough features (%d), threshold: %d (train.py)' % (image_filenames[i], len(keypoints[j]), parameter.min_features))

    return data
