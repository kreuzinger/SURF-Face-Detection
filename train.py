import tools
from face_detector import face_detector
from surf_detector import surf_detector
from image_data import Image_Data
from image_data import Face_Data
import sys
import logging
import parameter



"""
ToDo:
- Konvertierung in Graustufenbild schon bei Detectfaces?
- Methode detect_features: Andere Moeglichkeit anstatt Zwischenstufe mit speichern des bildes als tmp.jpg, damit surf.detect funktioniert?
"""

def get_face_data(image_filenames):
    """ Detects all faces and returns the cropped images"""

    logging.debug('Start method "get_face_data" (train.py)')
    logging.debug('%s images found (train.py)' % len(image_filenames))
    if len(image_filenames) == 0:
            logging.error('No image found in trainfolder (train.py)')
            sys.exit()
    
    fd = face_detector(); sd = surf_detector() #create new instances for detectors
    data = [] # container for image data
    
    # get coordinates for faces and crop faces
    for i in range(len(image_filenames)):
        image, i_coordinates = fd.detect_faces(image_filenames[i])    # detect faces with coordinates

        # only save image-data, when forced to do so (for performance)
        if parameter.save_images == 1:
            data.append(Image_Data(image_filenames[i], i_coordinates, image)) # save to data-container
        else:
            if parameter.save_images != 0:
                logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (train.py)')
            data.append(Image_Data(image_filenames[i], i_coordinates)) # save to data-container

    
        if not data[i].coordinates:
            logging.warn('No faces found in file %s (train.py)' % data[i].filename)
            continue
        
        cropped_faces = fd.crop_face(image, data[i].coordinates, data[i].filename) # crop faces from images       
        (keypoints, descriptors) = sd.detect_features(cropped_faces, image_filenames[i]) # detect features with SURF detector

        ###################################################
        #print('type(cropped_faces):',type(cropped_faces))
        #print('Gesicht gefunden')
        #tools.show_images(cropped_faces)
        #################################################

        # only save faces, which have more than minimal number of features
        face_data = [] # container for image data
        for j in range(len(keypoints)):
            if len(keypoints[j]) > parameter.min_features: # only save data with minimal amount of features
                if parameter.save_images == 1: # only save image-data, when parameter says 1 (for performance)
                    face_data.append(Face_Data(keypoints[j], descriptors[j], (cropped_faces[j].rows, None ,cropped_faces[j]))) # create container for face data
                else:
                    if parameter.save_images != 0:
                        logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (train.py)')
                    face_data.append(Face_Data(keypoints[j], descriptors[j],(cropped_faces[j].rows, None))) # create container for face data
                data[i].add_face_data(face_data) # add face data for every found face to corresponding image                    
            else:
                logging.warn('Face deleted in File %s (not enough features (%d), threshold: %d (train.py)' % (image_filenames[i], len(keypoints[j]), parameter.min_features))


        ###################################################
        #print 'Dateiname %s:  es wurden %d Gesichter gefunden und angezeigt' % (data[i].filename, len(data[i].facedata.face))
        #tools.show_images(cropped_faces)
        ######################################################

    return data
