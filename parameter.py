import logging

#Parameter for whole Programm
##################################
train = 1
pickle_data = 1
pickle_filename = 'C:/Users/Home/Eigene_Dokumente_-_nicht_synchronisiert/Studium_Technikum_Wien/Master_Thesis/Trainingsdatenbanken/Trainierte_Daten/test.pickle'

search = 1

save_images = 1                                     # for faster computing the saving of images and faces can be deactivated (0 = no, 1 = yes=
logger = logging.getLogger('myLogger');
logger.root.setLevel(logging.INFO)                  #DEBUG, INFO, WARN, ERROR, CRITICAL
"""     logging.debug('Debug') #Detailed information, typically of interest only when diagnosing problems.
        logging.info('info') #Confirmation that things are working as expected.
        logging.warn('warn') #An indication that something unexpected happened, or indicative of some problem in the near future (e.g. disk space low). The software is still working as expected
        logging.error('error') #Due to a more serious problem, the software has not been able to perform some function.
        logging.critical('critical') A serious error, indicating that the program itself may be unable to continue running."""


#PARAMETER FOR TRAIN
#######################
#Viola-Jones Detection
cascadefile = "haarcascade_frontalface_default.xml" # name of cascadefile for detecting the faces
scaleFactor = 1.1                                    # Parameter specifying how much the image size is reduced at each image scale. (standard value = 1.1 - http://opencv.itseez.com/modules/objdetect/doc/cascade_classification.html?highlight=cv.haar#cv.HaarDetectObjects)
minNeighbors = 3                                    # Parameter specifying how many neighbors each candiate rectangle should have to retain it. (standard value = 3 - http://opencv.itseez.com/modules/objdetect/doc/cascade_classification.html?highlight=cv.haar#cv.HaarDetectObjects)
flags = 0                                           # Mode of operation. Currently the only flag that may be specified is CV_HAAR_DO_CANNY_PRUNING (standard value = 0 - http://opencv.itseez.com/modules/objdetect/doc/cascade_classification.html?highlight=cv.haar#cv.HaarDetectObjects)
min_facesize = (150, 150)                           # Minimum possible object size. Objects smaller than that are ignored (height and width in pixel)

max_facesize = (800.0,800.0)                        # facesize will be reduced to that size for performance (hint: number must be decimal)

description_method = 'surf'
#SURF-Description
face_enlargement = 1.2                                # enlarge face BEFORE starting SURF (Example: 'None' for no enlargement, '1.5' for enlarge picture with 50 %)

min_features = 10                                   # only faces with this minimal amount of features will be analysed
hessian_threshold = 300                            # Threshold for hessian keypoint detector used in SURF
nOctaves = 3                                        # Number of pyramid octaves the keypoint detector will use
nOctaveLayers = 8                                   # Number of octave layers within each octave

#SIFT-Description


#PARAMETER FOR SEARCH
###########################
min_matches = 4                                    # Number of minimal matches for search. must be at least 4 matches. otherwise system crashes
number_topresults = 4                              # show number of topresults


#Matching
bruteforce_threshold = 0.6                           # fehlt noch: Threshold for Bruteforce Matching
