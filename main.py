import cv2
import argparse
import json
from ContourBbox import word_picker
from Inverted_or_upright import findUpright
#loading the image from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Path to the image",default='./data/1.png')
args = vars(ap.parse_args())
#loading the predefined language base
with open('generated.json') as json_file:
    language_base = json.load(json_file)
    
image = cv2.imread(args["image"],0)
#taking the biggest contour area for processing
cropped = word_picker(image)
language, result = findUpright(language_base,cropped)
print('The language identified for the given image is {} and it is "{}!"'.format(language,result))
