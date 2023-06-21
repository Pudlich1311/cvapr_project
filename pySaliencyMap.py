
import cv2
import numpy as np
import pySaliencyMapDefs

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
            now_size = (now_size[1], now_size[0])  ## (width, height)
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
        RGBMax[RGBMax <= 0] = 0.0001    # prevent dividing by 0
        # min(R,G)
        RGMin = cv2.min(R, G)
        # RG = (R-G)/max(R,G,B)
        RG = (R - G) / RGBMax
        # BY = (B-min(R,G)/max(R,G,B)
        BY = (B - RGMin) / RGBMax
        # clamp nagative values to 0
        RG[RG < 0] = 0
        BY[BY < 0] = 0
        # obtain feature maps in the same way as intensity
        RGFM = self.Gauss_Pyr_CSD(RG)
        BYFM = self.Gauss_Pyr_CSD(BY)
        # return
        return RGFM, BYFM
    
    ## motion feature maps
    def Get_FM(self, src):
        # convert scale
        I8U = np.uint8(255 * src)
        cv2.waitKey(10)
        # calculating optical flows
        if self.prev_frame is not None:
            farne_pyr_scale= pySaliencyMapDefs.farne_pyr_scale
            farne_levels = pySaliencyMapDefs.farne_levels
            farne_winsize = pySaliencyMapDefs.farne_winsize
            farne_iterations = pySaliencyMapDefs.farne_iterations
            farne_poly_n = pySaliencyMapDefs.farne_poly_n
            farne_poly_sigma = pySaliencyMapDefs.farne_poly_sigma
            farne_flags = pySaliencyMapDefs.farne_flags
            flow = cv2.calcOpticalFlowFarneback(\
                prev = self.prev_frame, \
                next = I8U, \
                pyr_scale = farne_pyr_scale, \
                levels = farne_levels, \
                winsize = farne_winsize, \
                iterations = farne_iterations, \
                poly_n = farne_poly_n, \
                poly_sigma = farne_poly_sigma, \
                flags = farne_flags, \
                flow = None \
            )
            flowx = flow[...,0]
            flowy = flow[...,1]
        else:
            flowx = np.zeros(I8U.shape)
            flowy = np.zeros(I8U.shape)
        # create Gaussian pyramids
        dst_x = self.Gauss_Pyr_CSD(flowx)
        dst_y = self.Gauss_Pyr_CSD(flowy)
        # update the current frame
        self.prev_frame = np.uint8(I8U)
        # return
        return dst_x, dst_y

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
        stepsize = pySaliencyMapDefs.default_step_local
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

    ## motion conspicuity map
    def Get_Motion(self, MFM_X, MFM_Y):
        return self.Get_Color(MFM_X, MFM_Y)


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
        MFM_X, MFM_Y = self.Get_FM(I)
        # extracting conspicuity maps
        ICM = self.Get_Intensity(IFM)
        CCM = self.Get_Color(CFM_RG, CFM_BY)
        MCM = self.Get_Motion(MFM_X, MFM_Y)
        # adding all the conspicuity maps to form a saliency map
        wi = pySaliencyMapDefs.weight_intensity
        wc = pySaliencyMapDefs.weight_color
        wm = pySaliencyMapDefs.weight_motion
        SMMat = wi*ICM + wc*CCM + wm*MCM
        # normalize
        normalizedSM = self.Range_Normalization(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)
        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        # return
        return self.SM

    def Get_Binarized_Map(self, src):
        # get a saliency map
        if self.SM is None:
            self.SM = self.Get_Saliency_Map(src)
        # convert scale
        SM_I8U = np.uint8(255 * self.SM)
        # binarize
        thresh, binarized_SM = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binarized_SM

    def Get_Salient_Region(self, src):
        # get a binarized saliency map
        binarized_SM = self.Get_Binarized_Map(src)
        # GrabCut
        img = src.copy()
        mask =  np.where((binarized_SM!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = (0,0,1,1)  # dummy
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)
        # post-processing
        mask_out = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask_out)
        return output
