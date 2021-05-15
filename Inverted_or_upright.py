import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

import imutils

sc = ShapeContext()
s=0
z=0
def get_contour_bounding_rectangles(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in cnts[1]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return res

def parse_nums(sc, path,z):
    global s
    img = np.asarray(path)
    # invert image colors
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # making numbers fat for better contour detectiion
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
#     print('After thresholding and dilation...')
#     plt.imshow(img)
#     plt.show()
    # getting our numbers one by one
    rois = get_contour_bounding_rectangles(img)
    grayd = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv2.rectangle(grayd, (r[0], r[1]), (r[2], r[3]),(0, 255, 0), 1)
        nums.append((r[0], r[1], r[2], r[3]))

# #     print('After greying and bounding...')
    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
#     print('bounding box x coords')
    if (z % 3 == 0):
        s = len(nums)
    else:
        pass
    descs = []
    for i, r in enumerate(nums):

        points = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 15)
        descriptor = sc.compute(points).flatten()
        descs.append(descriptor)
    #print(descs)
    return np.array(descs), s

def match(base, current,s):
    """
      Here we are using cosine diff instead of "by paper" diff, cause it's faster
    """
    res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
    # print("min = " + np.argmin(res.reshape(11)))
    char = str(np.argmin(res.reshape(s)))
    # print(char)
#     print(np.min(res.reshape(53)))
    return char, np.min(res.reshape(s))


def language_identification(language_base,testImg):
    d={}
    global z
    global s
    matchs=[]
    langs=[]
    #checking the probabilty to become a language in the predefined base with the test image
    for language,base_url in language_base.items():
        print(language,base_url)
        baseImage = cv2.imread(base_url,0)
        if len(baseImage.shape) !=2:
            baseImage=cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        #parsing the bbox values of base image and test image
        parsed_base,s = parse_nums(sc, baseImage,z)
        z+=1
        recognize,s = parse_nums(sc, testImg,z)
        z+=1
        #parsing the bbox values of base image and rotated test image
        secondImg = cv2.rotate(testImg,cv2.ROTATE_180)
        recognize_inverted,s = parse_nums(sc, secondImg,z)
        z+=1
        txt = ""
        matchFactor = 0
        val = 0
        c=0
        #matching each parsed character of test image with the base image
        for r in recognize:
            c, val = match(parsed_base, r,s)
            txt += c
            matchFactor += val
        txtInverted = ""
        matchFactorInv = 0
        val = 0
        c=0
        #matching each parsed character of rotated test image with the base image
        for r in recognize_inverted:
            c, val = match(parsed_base, r,s)
            txtInverted += c
            matchFactorInv += val 
#         print(matchFactor,matchFactorInv)
#         a = abs(matchFactor-matchFactorInv)
#         d[language]=a
        matchs.append([matchFactor,matchFactorInv])
        langs.append(language)
        langs = list(set(langs))
    #checking for the minimal matchfactor to identify the language
    #matc = min(matchs)
    matc = min(matchs, key = lambda t: t[1])
    index = matchs.index(matc)
    language_identified = langs[index]

    return language_identified,matc
    
            
            

    



def findUpright(language_base,testImg):
    
    language,match1 = language_identification(language_base,testImg)


    print("\nUpright Text Match Value = " + str(match1[0]))
    print("Flip Text Match Value = " + str(match1[1]))
    
    if (match1[0]> match1[1]):
        result = "inverted"
        lang_1 = language
        return lang_1, result
    else:
        result = "upright"
        lang_1 = language
        return lang_1,result


