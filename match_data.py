class Match_Data:
    """Data can be accessed the following way:
    'match_data[0].matched_data[0].matchvalue'
    for the Matching-Value of the first search-face with the first compared train-data"""
    
    def __init__(self, one_searchfilename, one_searchface = None):
        """contains the basic data of the search-image"""
        self.searchfilename = one_searchfilename
        self.searchface = one_searchface

    def add_matched_data(self,one_matcheddata):
        self.matcheddata = one_matcheddata

class Matched_Data:
    def __init__(self, one_filename, one_matchvalue, one_nbr_matches, one_face = None, one_vis = None):
        """contains the match data for the image in the class 'Match Data'"""
        self.filename = one_filename
        self.face = one_face
        self.vis = one_vis
        self.matchvalue = one_matchvalue
        self.nbr_matches = one_nbr_matches

