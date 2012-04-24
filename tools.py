import os
import cv
import cv2
import numpy as np
import logging
import csv
import operator
import parameter

def get_images(path):
    """ Returns a list of filenames for 
    all images in a directory. """
    image_filenames = []
    for f in os.listdir(path):
        if f.endswith('bmp') or f.endswith('jpg') or f.endswith('png'):
            image_filenames.append(os.path.join(path,f))
    return image_filenames

def show_images(images):
    """ Shows all images in a window"""
    if images == None:
        logging.error('Cannot Show Images (No image saved). Image-Type: %s (tools.py)' % str(type(images).__name__))
    elif type(images).__name__=='list':
        for i in range(len(images)):
            print type(images[i])
            if type(images[i]).__name__=='ndarray':
                tmpimage = []
                tmpimage[i] = array2cv(images[i])
                cv.ShowImage("Image", tmpimage[i])
                if cv.WaitKey() == 27:
                    cv.DestroyWindow("Image")
            else:
                cv.ShowImage("Image", images[i])
                if cv.WaitKey() == 27:
                    cv.DestroyWindow("Image")
    elif type(images).__name__=='cvmat':
        cv.ShowImage("Image", images)
        if cv.WaitKey() == 27:
            cv.DestroyWindow("Image")
    elif type(images).__name__=='iplimage':
        cv.ShowImage("Image", images)
        if cv.WaitKey() == 27:
            cv.DestroyWindow("Image")
    elif type(images).__name__=='ndarray':
        images = array2cv(images)
        cv.ShowImage("Image", images)
        if cv.WaitKey() == 27:
            cv.DestroyWindow("test")
    elif type(images).__name__=='str':
        logging.error('TypeError: Cannot Show Images (No image saved?). Image-Type: %s (tools.py)' % str(type(images).__name__))
    else:
        logging.error('TypeError: Cannot Show Images. Image-Type: %s (tools.py)' % str(type(images).__name__))


def show_keypoints(data):
    """ paint and show images with keypoints"""
    for i in range(len(data)):
        try:
            if data[i].facedata:                
                for j in range(len(data[i].facedata)):                                          
                    nbr_keypoints = len(data[i].facedata[j].keypoints)
                    print('%d Features found in file %s' %(nbr_keypoints, data[i].filename))
                    tmpImage = cv.CloneMat(data[i].facedata[j].face)
                    for k in range (nbr_keypoints):
                        if parameter.description_method == 'sift':
                            size = int(data[i].facedata[j].keypoints[k].size)
                        elif parameter.description_method == 'surf':
                            size = int(data[i].facedata[j].keypoints[k].size)/2
                        cv.Circle(tmpImage, (int(data[i].facedata[j].keypoints[k].pt[0]), int(data[i].facedata[j].keypoints[k].pt[1])), size, 255)
                    show_images(tmpImage)
        except:
            logging.error('Error showing keypoints - Maybe no Face in File %s (tools.py)' % data[i].filename)

            
def topresults(match_data):
    """ Show top matching results"""

    with open('match-statistics.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["Position", "Search-Image", "Train-Image",  "Matching-Value", "Nbr-Matches"])
        
        for i in range(len(match_data)):
            if hasattr(match_data[i], 'matcheddata'):
                nbr_images = len(match_data[i].matcheddata)
                matchvalues = {}
                if nbr_images == 0:
                    logging.warn('No MatchData available - Maybe no face in Search-File (tools.py)')
                else:  
                    # extract matchvalues into tmp dictionary
                    for j in range(nbr_images):
                        if not match_data[i].matcheddata[j].matchvalue:
                            logging.warn('No face found in file %s (tools.py)' % match_data[i].searchfilename)
                        else:   
                            matchvalues[j] = match_data[i].matcheddata[j].matchvalue

                    # sort dictionary            
                    sorted_matchvalues = sorted(matchvalues.iteritems(), key=operator.itemgetter(1), reverse=True)
              
                    nbr_matches = len(sorted_matchvalues)
                    for j in range(nbr_matches):     
                        if (j+1) <= int(parameter.number_topresults): # only show given number of topresults
                            if not match_data[i].matcheddata[j].matchvalue:
                                logging.warn('No face found in file %s (tools.py)' % match_data[i].filename)
                            else:
                                print('Nr. %d with MatchingValue %.5f (left %s, right %s, %d matches) (tools.py)' %((j+1), float(sorted_matchvalues[j][1]),match_data[i].matcheddata[sorted_matchvalues[j][0]].filename, match_data[i].searchfilename, match_data[i].matcheddata[sorted_matchvalues[j][0]].nbr_matches))

                                #write data to file
                                writer.writerow([j+1, match_data[i].searchfilename, match_data[i].matcheddata[sorted_matchvalues[j][0]].filename, float(sorted_matchvalues[j][1]), match_data[i].matcheddata[sorted_matchvalues[j][0]].nbr_matches])                    

                                # show images
                                if parameter.save_images == 1:
                                    show_images(match_data[i].matcheddata[sorted_matchvalues[j][0]].vis)
                        else:
                            print 'Only Top %s images are printed (tools.py)' % parameter.number_topresults
                            break
            else:
                logging.warn('No Match Data for Searchfile %s available' %match_data[i].searchfilename)
                writer.writerow(['', match_data[i].searchfilename, 'No Match-Data available'])                    


def count_features(data):
    """ prints and saves stats for the trained data"""

    with open('train-statistics.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["Image-Name", "Face-Nr", "number of Keypoints", "number of Descriptors", "rows", "cols"])
        for i in range(len(data)):
            try:
                if hasattr(data[i], 'facedata'):
                    for j in range(len(data[i].facedata)):
                        nbr_keypoints = len(data[i].facedata[j].keypoints)
                        nbr_descriptors = len(data[i].facedata[j].descriptors)                        
                        writer.writerow([data[i].filename, j+1, len(data[i].facedata[j].keypoints), len(data[i].facedata[j].descriptors), data[i].facedata[j].facesize[0], data[i].facedata[j].facesize[1]])                    
                else:
                    logging.info('No facedata available in File %s (Maybe no Face found or filtered?) (tools.py)' % data[i].filename)
                    writer.writerow([data[i].filename, "no data saved"])
            except AttributeError:
                logging.error('Error counting features for file %s (tools.py)' % data[i].filename)


def enlarge_image(image, factor):
    """ Enlarge the image to the given size
    Image must be of type cv.cvmat"""

    
    if type(image).__name__=='cvmat':
        new_image = cv.CreateMat(int(round(image.height * factor)), int(round(image.width * factor)), cv.GetElemType(image))
        cv.Resize(image, new_image)
        image = new_image
        logging.debug('Image has been enlarged with factor %.3f (face-detector.py)' % (factor))
        return image
    else:
        logging.error('Unkown Image Type (tools.py)')
        

def downsize_image(image):
    """ Resize the image to the given size
    Image must be of type cv.cvmat"""
    height_factor = float(image.height/parameter.max_facesize[0])
    width_factor = float(image.width/parameter.max_facesize[1])
    if height_factor > width_factor:        
        new_face = cv.CreateMat(image.height/height_factor, image.width/height_factor, cv.GetElemType(image))
        downsize_factor = height_factor
    else:
        new_face = cv.CreateMat(int(image.height/width_factor), int(image.width/width_factor), cv.GetElemType(image))
        downsize_factor = width_factor
    cv.Resize(image, new_face)
    return new_face, downsize_factor


def cv2array(im):
    """ convert from type cv into type array"""

    depth2dtype = { 
        cv.IPL_DEPTH_8U: 'uint8', 
        cv.IPL_DEPTH_8S: 'int8', 
        cv.IPL_DEPTH_16U: 'uint16', 
        cv.IPL_DEPTH_16S: 'int16', 
        cv.IPL_DEPTH_32S: 'int32', 
        cv.IPL_DEPTH_32F: 'float32', 
        cv.IPL_DEPTH_64F: 'float64', 
    } 

    arrdtype=im.depth 
    a = np.fromstring( 
        im.tostring(), 
        dtype=depth2dtype[im.depth], 
        count=im.width*im.height*im.nChannels) 
    a.shape = (im.height,im.width,im.nChannels) 
    return a 

def array2cv(a):
    """ convert from type array into type cv"""
    dtype2depth = { 
        'uint8':   cv.IPL_DEPTH_8U, 
        'int8':    cv.IPL_DEPTH_8S, 
        'uint16':  cv.IPL_DEPTH_16U, 
        'int16':   cv.IPL_DEPTH_16S, 
        'int32':   cv.IPL_DEPTH_32S, 
        'float32': cv.IPL_DEPTH_32F, 
        'float64': cv.IPL_DEPTH_64F, 
    } 
    try: 
        nChannels = a.shape[2] 
    except: 
        nChannels = 1 
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]), dtype2depth[str(a.dtype)], nChannels) 
    cv.SetData(cv_im, a.tostring(),a.dtype.itemsize*nChannels*a.shape[1]) 
    return cv_im


def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )
