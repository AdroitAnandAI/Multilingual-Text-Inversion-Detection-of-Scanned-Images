import cv2
import matplotlib.pyplot as plt
import random
import json
from dilationconstant import dilation_constant_selector
z=0
s=0
#function to remove the horizondal and vertical lines in the given image
def line_removal(image_path):
    image = image_path
    #setting up horizondal and vertical kernel for checking the lines
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60,1))
    verti_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,40))

    hori_line = 255-cv2.morphologyEx(image,cv2.MORPH_CLOSE, hori_kernel,iterations=1)

    result = cv2.add(image,hori_line)

    verti_line = 255 - cv2.morphologyEx(image,cv2.MORPH_CLOSE, verti_kernel,iterations=1)

    result = cv2.add(result,verti_line)
    return result


def word_picker(image):
    '''
    The function removes the horizondal and vertical lines in an image and find the contours of the words in it. And returns the largest         contour word.
    '''
    arr = []
    c = 0
    out = image.copy()
    line_removed = line_removal(image)

    imagem = cv2.bitwise_not(line_removed)
    #finding the optimal kernel value by curve fitting method.
    k = dilation_constant_selector(imagem)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    dilate = cv2.dilate(imagem,kernel,iterations=1)

    contours = cv2.findContours(dilate, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours[1])):
        area = cv2.contourArea(contours[1][i])
        x,y,w,h = cv2.boundingRect(contours[1][i])
        #arr.append([x,y,x+w,y+h])
        arr.append([x,y,w,h])
    #sorting the contours on the basis of width and selcting the largest one
    arr.sort(key = lambda x: x[2])
    c = arr[-1]
    #cropping the largest area from the image
    crop_img = out[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
    return crop_img



