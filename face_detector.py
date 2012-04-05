import cv
import cv2
import tools
import logging
import parameter

class face_detector:

    def __init__(self):
        """nicht notwendig?"""

    def detect_faces(self, image_filename):
        """ Detects all faces and returns a list with
                images and corresponding coordinates"""

        logging.debug('Start method "detect_faces" for file %s (face-detector.py)' % image_filename)
        cascade = cv.Load(parameter.cascadefile) # load face cascade    
        tmpimage = cv.LoadImageM(image_filename) # loads, converts and saves each image
        
        # enlarge image if size is smaller than parameter
        if parameter.min_imagesize != None:
            if tmpimage.height < parameter.min_imagesize[0] or tmpimage.width < parameter.min_imagesize[1]:
                tmpimage = tools.enlarge_image(tmpimage, parameter.min_imagesize[2]) # resize image with given factor to find more faces
                logging.debug('Image from file %s has been enlarged with factor %.3f (face-detector.py)' % (image_filename, parameter.min_imagesize[2]))

        image = cv.GetImage(tmpimage) # convert image to IplImage
        coordinates = cv.HaarDetectObjects(image, cascade, cv.CreateMemStorage(), parameter.scaleFactor, parameter.minNeighbors, parameter.flags, parameter.min_facesize) # detect and save coordinates of detected faces with standard parameters

        # Convert to greyscale - better results when converting AFTER facedetection with viola jones
        if image.channels == 3:
            logging.debug('Bild %s wird in Graustufenbild umgewandelt (face-detector.py)' % image_filename)
            grey_face = (cv.CreateImage((image.width,image.height), 8,1)) # Create grey-scale Image
            cv.CvtColor(image, grey_face, cv.CV_RGB2GRAY) # convert Image to Greyscale (necessary for SURF)
            image = grey_face

        #############################
        # zeichnet Rechteck - zur Kontrolle - kann am Ende entfernt werden
        #for ((x,y,w,h),n) in coordinates:
        #    cv.Rectangle(image, (x,y), (x+w,y+h), 255)
        #tools.show_images(image)
        #############################################
        
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
                logging.debug('Ein Gesicht in Bild %s wurde um den Faktor %d verkleinert (face-detector.py)' % (image_filename, downsize_factor))
            #print 'Groesse des Gesichtes in Datei %s: %d x %d (face_detector.py)' % (image_filename, cropped_faces[i].height, cropped_faces[i].width)

        #tools.show_images(cropped_faces)
        logging.debug('%d faces successfully cropped (face-detector.py)', len(cropped_faces))
        return cropped_faces # faces are defined with ROI
