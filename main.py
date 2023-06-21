import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import argparse 

# main
if __name__ == '__main__':


    parser = argparse.ArgumentParser(
                    prog='CVAPR Project',
                    description='Saliency detection with the usage of Gaussian Pyramids')
    
    parser.add_argument('-i', '--image',required=True,help='Input image')
    args = parser.parse_args()
    img = cv2.imread(args.image)

    # initialize
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

    # computation
    saliency_map = sm.Get_Saliency_Map(img)
    binarized_map = sm.Get_Binarized_Map(img)
    salient_region = sm.Get_Salient_Region(img)

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