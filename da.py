import numpy as np
import os
import cv2
import pylab
path="aug" ##originalimage that is need to be croped
listing=os.listdir(path)
i=1
for crop in listing:
    image = cv2.imread("aug/"+crop) ##taking images in the array form
    # image = cv2.imread(itr)
    height,width = image.shape[:2]
    start_row ,start_col =int(height*0.1),int(width*0.15)
    end_row ,end_col =int(height*0.85),int(width*0.9)
  	crop =image[start_row:end_row, start_col:end_col]
  	cv2.imwrite('test/'+str(i)+'.jpg',crop) ##put your folder name where croped image are supposed to be by replacing 'test'
    cv2.destroyAllWindows()
	i+=1
    ##best of luck for team electronic hahaha
