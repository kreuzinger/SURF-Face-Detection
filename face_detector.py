import cv
import tools
import logging
import parameter

class face_detector:

    def detect_faces(self, image_filename):
        """ Detects all faces and returns a list with
                images and corresponding coordinates"""

        logging.debug('Start method "detect_faces" for file %s (face-detector.py)' % image_filename)
        cascade = cv.Load(parameter.cascadefile) # load face cascade    
        image = cv.LoadImage(image_filename) # loads and converts image

        # detect and save coordinates of detected faces
        coordinates = cv.HaarDetectObjects(image, cascade, cv.CreateMemStorage(), parameter.scaleFactor, parameter.minNeighbors, parameter.flags, parameter.min_facesize)

        # Convert to greyscale - better results when converting AFTER facedetection with viola jones
        if image.channels == 3:
            logging.debug('Bild %s wird in Graustufenbild umgewandelt (face-detector.py)' % image_filename)
            grey_face = (cv.CreateImage((image.width,image.height), 8,1)) # Create grey-scale Image
            cv.CvtColor(image, grey_face, cv.CV_RGB2GRAY) # convert Image to Greyscale (necessary for SURF)
            image = grey_face
        
        logging.debug('%d faces successfully detected in file %s (face-detector.py)' % (len(coordinates), image_filename)) 
        return image, coordinates


    def crop_face(self, image, coordinates, image_filename):
        """ Crops all faces from a list of images and coordinates
        Returns a list with all faces"""

        logging.debug('Start method "crop_face" for file %s (face-detector.py)' % image_filename)
        cropped_faces = [] # list with all cropped faces (defined with ROI)
                 
        for i in range(len(coordinates)):
            rectangle = coordinates[i][0]
            cropped_faces.append(cv.GetSubRect(image, rectangle)) # save faces (with ROI) in new list

            #check face for max image size
            if cropped_faces[i].height > parameter.max_facesize[0] or cropped_faces[i].width > parameter.max_facesize[1]: #start resize
                (cropped_faces[i], downsize_factor) = tools.downsize_image(cropped_faces[i])
                logging.debug('Face in image %s has been downsized with factor %d (face-detector.py)' % (image_filename, downsize_factor))

        logging.debug('%d faces successfully cropped (face-detector.py)', len(cropped_faces))
        return cropped_faces # faces are defined with ROI
