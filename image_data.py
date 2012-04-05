class Image_Data:
    """Data can be accessed the following way:
    'trained_data[0].facedata[0].keypoints'
    for the keypoints of the first face"""

    def __init__(self, one_filename, one_coordinates, one_image = None):
        """contains the basic data of the train-image"""

        self.filename = one_filename
        self.image = one_image
        self.coordinates = one_coordinates

    def add_face_data(self,one_facedata):
        self.facedata = one_facedata


class Face_Data:
    def __init__(self, one_keypoints, one_descriptors, one_facesize, one_face = None):
        """contains the face data for the image in the class 'Image Data'"""
        self.keypoints = one_keypoints
        self.descriptors = one_descriptors
        self.face = one_face
        self.facesize = one_facesize
