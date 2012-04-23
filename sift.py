from PIL import Image
import os
import numpy as np
import pylab
from subprocess import Popen, PIPE, STARTUPINFO, STARTF_USESHOWWINDOW 


def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	""" Process an image and save the results in a file. """
	if imagename[-3:] != 'pgm':
		# create a pgm file
		im = Image.open(imagename).convert('L')
		im.save('tmp.pgm')
		imagename = 'tmp.pgm'

	cmmd = str("sift "+imagename+" --output="+resultname+" "+params)

        # run the sift.exe file
	if os.name == 'nt':     # only on windows
		startupinfo = STARTUPINFO() 
		startupinfo.dwFlags |= STARTF_USESHOWWINDOW
	p = Popen(cmmd, stdout = PIPE, startupinfo=startupinfo) # run sift.exe in subprocess (no separate window opens)
	p.wait() # wait until subprocess finished

	#os.system(cmmd)  # run sift.exe (previous call of the sift.exe)
	


def read_features_from_file(filename):
	""" Read feature properties and return in matrix form. """
	
	f = np.loadtxt(filename)
	if np.ndim(f) < 2:
		return None, None
	 
	else:
		return f[:,:4],f[:,4:] # feature locations, descriptors


def write_features_to_file(filename,locs,desc):
	""" Save feature location and descriptor to file. """
	savetxt(filename,hstack((locs,desc)))
	

