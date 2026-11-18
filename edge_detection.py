import cv2

def canny_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def prewitt_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernelx = cv2.getDerivKernels(1, 0, 3)[0]
    kernely = cv2.getDerivKernels(0, 1, 3)[0]
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    return prewittx + prewitty
