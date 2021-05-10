import cv2
import numpy as np

from scipy.optimize import curve_fit

def sigmoid(x, L ,x0, k, b):
    #defining the inverse sigmoid function for curve fit calculation
    y = (L / (1 + np.exp(k*(x+x0)))+b)
    return (y)


def isCurveSigmoid(ContourCounts, count):
        '''
            The function will check with the obtained contour values(greater than 4 for minimum count) that if it fits the inverse sigmoid               function.If it fits the event is triggered else it will return false. This is the method used for finding optimal dilation                   kernel size
        '''

        xIndex = len(ContourCounts)

        p0 = [max(ContourCounts), np.median(xIndex),1,min(ContourCounts)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, list(range(xIndex)), ContourCounts, p0, method='lm', maxfev=10000)

        yVals = sigmoid(list(range(xIndex)), *popt)

    

        # May have to check for a value much less than Median to avoid false positives.
        if np.median(yVals[:3]) - np.median(yVals[-3:]) > 15:
            print('Triggered Event')
            return True
        else:

            return False




def findCurveFit(ContourCount):

    triggerEvent = False
    #checking for each contour value list from i=4 (minimum required points) to fit the curve 
    if isCurveSigmoid(ContourCount, len(ContourCount)):
        print('Event Triggered...')
        triggerEvent = True


    return ContourCount, triggerEvent