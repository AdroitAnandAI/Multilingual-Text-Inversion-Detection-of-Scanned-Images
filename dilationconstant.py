import cv2
from curve_fit import findCurveFit
#helper function to find the optimal kernel for dilation
def dilation_constant_selector(image):
    '''
    This is an helper function to find the optimal K for the dilation. it checks with a list of K values and once the curve is fitted the K is     returned to here
    '''
    ContourCount =[]
    kernel_value =0
    for i in range (1,100):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i,i))
        dilate = cv2.dilate(image,kernel,iterations=1)
        contours = cv2.findContours(dilate,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ContourCount.append(len(contours[1]))

        if i >=4:
            pixelC, trigg = findCurveFit(ContourCount)
            if trigg == True:
                kernel_value=i
                break
    return kernel_value
            
