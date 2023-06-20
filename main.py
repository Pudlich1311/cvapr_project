
import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

# main
if __name__ == '__main__':

    # read
    img = cv2.imread('dog.jpg')

    # initialize
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

    # computation
    saliency_map = sm.SMGetSM(img)
    binarized_map = sm.SMGetBinarizedSM(img)
    salient_region = sm.SMGetSalientRegion(img)

    # visualize

    plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input image')

    plt.subplot(2,2,2), plt.imshow(saliency_map, 'gray')
    plt.title('Saliency map')

    plt.subplot(2,2,3), plt.imshow(binarized_map)
    plt.title('Binarilized saliency map')

    plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
    plt.title('Salient region')


    plt.show()
    cv2.destroyAllWindows()