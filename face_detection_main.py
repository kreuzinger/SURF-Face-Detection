import tools
import train
import search
import cv2
import numpy as np
import persistence
import logging
import time
import parameter

help_message = '''Face Detection 

                NOCH AKTUALISIEREN:
                USAGE: face_detection_main.py [ <searchimage> ]
                Determine the configuration file parameter.py (mode, trainfolder, searchmode)
                '''

t0 = time.time()        # count execution-time for whole programm
localtime = time.asctime( time.localtime(time.time()) )
print "Local current time :", localtime
logging.info('Current Log-Level: %s' % logging.getLevelName(logging.getLogger('myLogger').root.getEffectiveLevel()))

if parameter.train == 1:
    #start train
    ##############################################
    print 'TRAINING STARTET'

    # foldername containing the training images
    trainfolder = "train"                               

    # make list with all images in the folder
    image_filenames = tools.get_images(trainfolder)

    # detect faces and features in images
    trained_data = train.get_face_data(image_filenames)
    
    if parameter.pickle_data == 1:
        # safe (pickle) data
        persistence.safe_data(trained_data)
        print time.time() - t0, "seconds" # print execution-time for Training
        
    print 'TRAINING ABGESCHLOSSEN'

    
    
    #optional evaluation
    ###############################################

    # show training-images and paint keypoints into the pictures
    # tools.show_keypoints(trained_data)

    # count and save features from trained data
    # tools.count_features(trained_data)




if parameter.search == 1:
    #start search
    ##############################################
    print 'SUCHE STARTET'

    # foldername containing the search images
    searchfolder = "search"                             

    try:
        print trained_data[0]
    except:
        trained_data = ""

    # get filenames in the training folder
    image_filenames = tools.get_images(searchfolder)

    # detect faces and features in images
    search_data = train.get_face_data(image_filenames) # get image-data for the seach-images

    # search, sort and save data (If no training-data is submitted, it should load and compare with the pickled training file)
    (search_data, match_data) = search.compute_search_sort(search_data, trained_data) 

    # show top matched results
    tools.topresults(match_data)

    print 'SUCHE ABGESCHLOSSEN'


print time.time() - t0, "seconds" # print execution-time for whole programm
 



