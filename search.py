from descriptor_matcher import descriptor_matcher
import logging
from match_data import Match_Data
from match_data import Matched_Data
import sys
import persistence

def compute_search_sort(search_data, trained_data):
    """ compare two datasets
    and return the matching data"""
    
    if not trained_data: # if no data is trained prior, then try to load the train data from pickled file
        trained_data = persistence.load_data()            
    dm = descriptor_matcher()
    match_data =[]

    # start matching for the first search-image with all train-images (only the first found face in search image is computed - could be changed here or in file descriptor_matcher.py) 
    for i in range(len(search_data)):
        match_data.append(Match_Data(search_data[i].filename)) # add searchfile data to object

        if hasattr(search_data[i], 'facedata'): # check if this image has faces with corresponding data
            logging.info('Start match with searchfile %s (search.py)' % search_data[i].filename)
            matcheddata = dm.match(trained_data, search_data[i]) # compute Matching
            match_data[i].add_matched_data(matcheddata) # add result to match_data container
        else:
            logging.error('Error in executing Descriptor-Matcher - Maybe no face in Search-File %s (search.py)' % search_data[i].filename)

    if not match_data:
        logging.critical('Error in executing Descriptor-Matcher - No match-data available (search.py)')
        sys.exit()
    else:
        return match_data
