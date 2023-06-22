import cv2
import numpy as np

class pySaliencyMap:
    # initialization
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.prev_frame = None
        self.SM = None


    # extracting color channels
    def Extracr_RGB(self, inputImage):
        # convert scale of array elements
        src = np.float32(inputImage) * 1./255
        # split
        (B, G, R) = cv2.split(src)
        # extract an intensity image
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # return
        return R, G, B, I

    # feature maps
    ## constructing a Gaussian pyramid
    def Create_Gauss_Pyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1,9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst
    
    ## taking center-surround differences
    def Center_Surround_Diff(self, GaussianMaps):
        dst = list()
        for s in range(2,5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0])  
            
            tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst
    
    ## constructing a Gaussian pyramid + taking center-surround differences
    def Gauss_Pyr_CSD(self, src):
        GaussianMaps = self.Create_Gauss_Pyr(src)
        dst = self.Center_Surround_Diff(GaussianMaps)
        return dst
    
    ## intensity feature maps
    def Intensity_FM(self, I):
        return self.Gauss_Pyr_CSD(I)
    
    ## color feature maps
    def Get_Color_FM(self, R, G, B):
        # max(R,G,B)
        tmp1 = cv2.max(R, G)
        RGBMax = cv2.max(B, tmp1)
        RGBMax[RGBMax <= 0] = 0.0001   
        RGMin = cv2.min(R, G)
        RG = (R - G) / RGBMax
        BY = (B - RGMin) / RGBMax
        # clamp nagative values to 0
        RG[RG < 0] = 0
        BY[BY < 0] = 0
        # obtain feature maps in the same way as intensity
        RGFM = self.Gauss_Pyr_CSD(RG)
        BYFM = self.Gauss_Pyr_CSD(BY)
        # return
        return RGFM, BYFM
    
    #orientation feature map
    def Get_Orientation_FM(self, src):
        # creating a Gaussian pyramid
        GaussianI = self.Create_Gauss_Pyr(src)
        # convoluting a Gabor filter with an intensity image to extract oriemtation features
        GaborOutput0   = [ np.empty((1,1)), np.empty((1,1)) ]  # dummy data: any kinds of np.array()s are OK
        GaborOutput45  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput90  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput135 = [ np.empty((1,1)), np.empty((1,1)) ]
        for j in range(2,9):
            GaborOutput0.append(   cv2.filter2D(GaussianI[j], -1, cv2.getGaborKernel((9, 9), 2,0, 5, 1, 0, ktype=cv2.CV_32F)) )
            GaborOutput45.append(  cv2.filter2D(GaussianI[j], -1, cv2.getGaborKernel((9, 9),  2,45, 5, 1, 0, ktype=cv2.CV_32F))) 
            GaborOutput90.append(  cv2.filter2D(GaussianI[j], -1, cv2.getGaborKernel((9, 9),  2,90, 5, 1, 0, ktype=cv2.CV_32F))) 
            GaborOutput135.append( cv2.filter2D(GaussianI[j], -1, cv2.getGaborKernel((9, 9),  2,135, 5, 1, 0, ktype=cv2.CV_32F))) 
        # calculating center-surround differences for every oriantation
        CSD0   = self.Center_Surround_Diff(GaborOutput0)
        CSD45  = self.Center_Surround_Diff(GaborOutput45)
        CSD90  = self.Center_Surround_Diff(GaborOutput90)
        CSD135 = self.Center_Surround_Diff(GaborOutput135)
        # concatenate
        dst = list(CSD0)
        dst.extend(CSD45)
        dst.extend(CSD90)
        dst.extend(CSD135)
        # return
        return dst

    # conspicuity maps
    ## standard range normalization
    def Range_Normalization(self, src):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src)
        if maxx!=minn:
            dst = src/(maxx-minn) + minn/(minn-maxx)
        else:
            dst = src - minn
        return dst
    
    ## computing an average of local maxima
    def Local_Max(self, src):
        # size
        stepsize = 16
        width = src.shape[1]
        height = src.shape[0]
        # find local maxima
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[y:y+stepsize, x:x+stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        # averaging over all the local regions
        return lmaxmean / numlocal
    
    ## normalization specific for the saliency map model
    def Saliency_Model_Normalization(self, src):
        dst = self.Range_Normalization(src)
        lmaxmean = self.Local_Max(dst)
        normcoeff = (1-lmaxmean)*(1-lmaxmean)
        return dst * normcoeff
    
    ## normalizing feature maps
    def Normalize_Feature_Maps(self, FM):
        NFM = list()
        for i in range(0,6):
            normalizedImage = self.Saliency_Model_Normalization(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            NFM.append(nownfm)
        return NFM
    
    ## intensity conspicuity map
    def Get_Intensity(self, IFM):
        NIFM = self.Normalize_Feature_Maps(IFM)
        ICM = sum(NIFM)
        return ICM
    
    ## color conspicuity map
    def Get_Color(self, CFM_RG, CFM_BY):
        # extracting a conspicuity map for every color opponent pair
        CCM_RG = self.Get_Intensity(CFM_RG)
        CCM_BY = self.Get_Intensity(CFM_BY)
        # merge
        CCM = CCM_RG + CCM_BY
        # return
        return CCM

    
    ## orientation conspicuity map
    def Get_Orientation(self, OFM):
        OCM = np.zeros((self.height, self.width))
        for i in range (0,4):
            # slicing
            nowofm = OFM[i*6:(i+1)*6]  # angle = i*45
            # extracting a conspicuity map for every angle
            NOFM = self.Get_Intensity(nowofm)
            # normalize
            NOFM2 = self.Saliency_Model_Normalization(NOFM)
            # accumulate
            OCM += NOFM2
        return OCM

    def Get_Saliency_Map(self, src):
        # definitions
        size = src.shape
        width  = size[1]
        height = size[0]
        # extracting individual color channels
        R, G, B, I = self.Extracr_RGB(src)
        # extracting feature maps
        IFM = self.Intensity_FM(I)
        CFM_RG, CFM_BY = self.Get_Color_FM(R, G, B)
        OFM = self.Get_Orientation_FM(I)
        # extracting conspicuity maps
        ICM = self.Get_Intensity(IFM)
        CCM = self.Get_Color(CFM_RG, CFM_BY)
        OCM = self.Get_Orientation(OFM)

        SMMat = ICM*0.3 + CCM*0.3 + OCM*0.2
        # normalize
        normalizedSM = self.Range_Normalization(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)
        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        # return
        return self.SM
