from descriptor_matcher import descriptor_matcher
import persistence
import logging
from match_data import Match_Data
from match_data import Matched_Data
import sys
import parameter



"""
ToDo:
Es wird immer nur mit seach_data.facedata.face[0] verglichen, also nur mit dem ersten gefundenen Gesicht im Suchbild
Diese Festlegung erfolgt in der Datei "descriptor_matcher": img2 = search_data.facedata.face[0]
Ich sollte hier oder im Descriptor Matcher auch eine Schleife ueber alle gefundenen Gesichter laufen lassen
"""

def compute_search_sort(search_data, trained_data):
    """ compare two datasets
    and return the matching data"""
    
    if not trained_data:

        trained_data = persistence.load_data()
        logging.info('Train-Data successfully loaded from %s' %parameter.pickle_filename)

            
    dm = descriptor_matcher()
    match_data =[]


    # start matching for the first search-image with all train-images
    for i in range(len(search_data)):
        match_data.append(Match_Data(search_data[i].filename)) # add searchfile data to object

        if hasattr(search_data[i], 'facedata'): # check if this image has faces with corresponding data
            logging.info('Start match with searchfile %s (search.py)' % search_data[i].filename)
            matcheddata = dm.match(trained_data, search_data[i]) # compute Matching
            match_data[i].add_matched_data(matcheddata)


        else:
            logging.error('Error in executing Descriptor-Matcher - Maybe no face in Search-File %s (search.py)' % search_data[i].filename)

    if not match_data:
        logging.critical('Error in executing Descriptor-Matcher - No match-data available (search.py)')
        sys.exit()
    else:
        return search_data, match_data








""" Diese Methode wird nicht mehr verwendet
def compute_search(search_data, trained_data):

    if not trained_data:
        try:
             trained_data = persistence.load_data('t_data.pickle')
        except: logging.error('No trained data found')
            
    dm = descriptor_matcher()
    matching_algorithm = 'bruteforce'

    # start matching for each search-image with all train-images
    nbr_search_data = len(search_data)
    for i in range(nbr_search_data):
        try:
            if search_data[i].facedata: # check if this image has faces with corresponding data
                logging.info('Starte Vergleich mit Datei %s' % search_data[i].filename)
                matcheddata = dm.match(matching_algorithm, trained_data, search_data[i]) # es wird derzeit nur mit search_data.facedata.face[0] verglichen
                       
        except:
            logging.error('Error in executing Descriptor-Matcher - Maybe no face in Search-File %s' % search_data[i].filename)
    return search_data
"""


