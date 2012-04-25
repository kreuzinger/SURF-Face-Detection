import cv2
import numpy as np
import tools
import logging
import sys
from match_data import Matched_Data
import parameter

class descriptor_matcher:

    def match(self, trained_data, search_data):
        
        def match_and_draw(match_bruteforce, i, trained_data):
            logging.debug('Start method match_and_draw (descriptor_matcher.py)')
            m = match_bruteforce(desc1, desc2)

            if len(m) < parameter.min_matches: # must be at least 4 otherwise algorithm crashes
                vis = None
                matchvalue = None
                nbr_matches = None
                logging.debug('No Matching possible: less than %d matches with face %d in train-file %s (descriptor_matcher.py)' %(parameter.min_matches, (j+1), trained_data[i].filename))
                return vis, matchvalue, nbr_matches
            else:
                matched_p1 = np.array([kp1[k].pt for k, l in m])
                matched_p2 = np.array([kp2[l].pt for k, l in m])
     
                H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0) 
                logging.debug('%d / %d  inliers/matched (descriptor_matcher.py)' % (np.sum(status), len(status)))
                
                #Calculate the matching in a procent value
                matchvalue = float(len(status))/float(len(kp2))*1.0
                logging.debug('ratio number of matches and number of features in searchimage: %.5f (descriptor_matcher.py)' % matchvalue)
                
                # only draw-match, when images are available 
                if parameter.save_images == 1:
                    img1 = trained_data[i].facedata[j].face
                    # Comparision is only computed with the first found face in the searchfile
                    img2 = search_data.facedata[0].face
                    img2 = np.array(img2)                   
                    vis = self.draw_match(img1, img2, matched_p1, matched_p2, status, H)
                else:
                    if parameter.save_images != 0:
                        logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (descriptor_matcher.py)')
                    vis = None
                
                nbr_matches = len(status)
                return vis, matchvalue, nbr_matches

        matcheddata = []
        logger = logging.getLogger('myLogger')
        
        # Comparision is only computed with the first found face in the searchfile. If more than one faces are found, the second, third, ... face is not computed
        desc2 = search_data.facedata[0].descriptors
        kp2 = search_data.facedata[0].keypoints
        for i in range(len(trained_data)):
            if hasattr(trained_data[i], 'facedata'):
                logging.debug('Start match with trained file %s (descriptor_matcher.py)' % trained_data[i].filename)

                for j in range(len(trained_data[i].facedata)):                   
                    desc1 = trained_data[i].facedata[j].descriptors
                    kp1 = trained_data[i].facedata[j].keypoints
                    logging.debug('Face %d from train-image: %s - %d features, search-image: %s - %d features (descriptor_matcher.py)' % ((j+1),trained_data[i].filename, len(kp1), search_data.filename, len(kp2)))

                    (vis_brute, matchvalue, nbr_matches) = match_and_draw(self.match_bruteforce, i, trained_data)
                    if matchvalue == None:
                        logging.debug('No matchvalue for file %s. Maybe not enough matches? (descriptor_matcher.py)', trained_data[i].filename)
                    else:
                        # only append images when available
                        if parameter.save_images == 1: 
                            matcheddata.append(Matched_Data(trained_data[i].filename, matchvalue, nbr_matches, trained_data[i].facedata[j].face, vis_brute))
                        else:
                            if parameter.save_images != 0:
                                logging.critical('Invalid parameter "save-images", must be 0 or 1 - now changed to 0 (train.py)')
                            matcheddata.append(Matched_Data(trained_data[i].filename, matchvalue, nbr_matches))
            else:
                logging.debug('Match with file %s not possible - there are no face-data saved (descriptor_matcher.py)' % trained_data[i].filename)
        return matcheddata
          
    def match_bruteforce(self, desc1, desc2):
        logging.debug('Start Methode match_bruteforce (descriptor_matcher.py)')       
        res = []        
        for j in xrange(len(desc1)):
            dist = tools.anorm(desc2 - desc1[j])
            n1, n2 = dist.argsort()[:2]
            r = dist[n1] / dist[n2]
            if r < parameter.bruteforce_threshold: # only distances within threshold will be returned
                res.append((j, n1))
        return np.array(res) # position j and n1 is returned

    def draw_match(self, img1, img2, p1, p2, status = None, H = None):
        logging.debug('Start method draw_match (descriptor_matcher.py)')

        if type(img1) == cv2.cv.cvmat:
            logging.debug('Start convert image from format iplimage to numpy-array (persistence.py)')
            img1 = np.array(img1)
        if type(img2) == cv2.cv.cvmat:
            logging.debug('Start convert image from format iplimage to numpy-array (persistence.py)')
            img2 = np.array(img2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
            cv2.polylines(vis, [corners], True, (255, 255, 255))
        
        if status is None:
            status = np.ones(len(p1), np.bool_)
        green = (0, 255, 0)
        red = (0, 0, 255)
        for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
            col = [red, green][inlier]
            if inlier:
                cv2.line(vis, (x1, y1), (x2+w1, y2), col)
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2+w1, y2), 2, col, -1)
            else:
                r = 2
                thickness = 3
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
                cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
                cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
        return vis
