import numpy as np
import cv2
import glob

"""
put number of internal corners of chessboard in first function
put paths, outfile names, test names in the script area

"""
###########################################
##        input parameters here!         ##
###########################################
bs1 = 6
bs2 = 9
boardsize = 76.2 #in mm
DEPTH_VISUALIZATION_SCALE = 1
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OPTIMIZE_ALPHA = -1


###########################################
##              Functions                ##
###########################################

def stereo_camera_coefficients(pathleft,pathright):
    """
    Calculate coefficients for one camera given folder of calibration images
    :param path: path to the calibration checkerboard images
    :type path: str
    :out: ret,mtx,dist,rvecs,tvecs,objpoints,imgpoints,imageSize

    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((bs2*bs1,3), np.float32)
    # if you know size... (* below by it)
    objp[:,:2] = np.mgrid[0:bs1,0:bs2].T.reshape(-1,2)*boardsize

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_l = [] # 2d points in image plane.
    imgpoints_r = []

    #find set of images
    images_l = sorted(glob.glob(pathleft + '*.png'))
    images_r = sorted(glob.glob(pathright + '*.png'))

    for i, fname in enumerate(images_l):
        img_l = cv2.imread(images_l[i])
        img_r = cv2.imread(images_r[i])
        img_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(img_l, (bs1,bs2),None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, (bs1,bs2),None)

        # If found, add object points, image points (after refining them)
        if ret_l == True and ret_r == True:
            objpoints.append(objp)

            corners2_l = cv2.cornerSubPix(img_l,corners_l,(11,11),(-1,-1),criteria)
            imgpoints_l.append(corners2_l)

            corners2_r = cv2.cornerSubPix(img_r,corners_r,(11,11),(-1,-1),criteria)
            imgpoints_r.append(corners2_r)
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (bs1,bs2), corners2, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

        imageSize = img_l.shape[::-1]

    #cv2.destroyAllWindows()

    # get camera matrix, dist coefficients, rotation, translation
    leftret, leftmtx, leftdist, leftrvecs, lefttvecs = cv2.calibrateCamera(objpoints, imgpoints_l, imageSize,None,None)
    rightret, rightmtx, rightdist, rightrvecs, righttvecs = cv2.calibrateCamera(objpoints, imgpoints_r, imageSize,None,None)
    print('Calibrated Individual Cameras.')
    return [objpoints,imgpoints_l,imgpoints_r,leftmtx,leftdist,rightmtx,rightdist,imageSize]


def stereo_calibrate(pathleft,pathright):
    """
    calculate stereo calibration and output mapping 
    :param pathleft: path to left chessboard images
    :param pathright: path to right chessboard images

    """

    x = stereo_camera_coefficients(pathleft,pathright)
    #Stereo Calibrate
    (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7], None, None, None, None, cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5,criteria)

    #Stereo Rectify
    (leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(x[3],x[4],x[5],x[6],x[7], rotationMatrix, translationVector,None, None, None, None, None,0,OPTIMIZE_ALPHA)

    #Stereo Undistort
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(x[3], x[4], leftRectification,leftProjection, x[7], cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(x[5], x[6], rightRectification,rightProjection, x[7], cv2.CV_32FC1)
    leftmtx = np.array([[1713.423138,0.000000,1059.993139], [0.000000,1707.621255,719.751860], [0.000000,0.000000,1.000000]])
    leftdist = np.array([-0.222780,0.120212,-0.000963,0.000254,0.000000])
    leftrect = np.array([[1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000],[0.000000,0.000000,1.000000]])
    leftproj = np.array([[1600.629883,0.000000,1071.946359,0.000000],[0.000000,1640.808960,718.703458,0.000000],[0.000000,0.000000,1.000000,0.000000]])
    rightmtx = np.array([[1712.647094,0.000000,1036.911331],[0.000000,1707.202719,717.219448],[0.000000,0.000000,1.000000]])
    rightdist = np.array([-0.228460,0.108732,-0.000444,0.000898,0.000000])
    rightrect = np.array([[1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000],[0.000000,0.000000,1.000000]])
    rightproj = np.array([[1593.064209,0.000000,1048.802068,0.000000],[0.000000,1637.907715,716.516482,0.000000],[0.000000,0.000000,1.000000,0.000000]])
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftmtx, leftdist, leftrect,leftproj, x[7], cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightmtx, rightdist, rightrect,rightproj, x[7], cv2.CV_32FC1)

    np.savez_compressed(parameters_name, imageSize=x[3],leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)
    print('Saved parameters.')

def calculate_depth_map(outfile,calibfile,imgL,imgR,MinDisp=80,NumDisp=112+16,BSize=1,SpecRange=4,SpecWindow=100,P1=19,P2=20,PreFilterCap=5,disp12MaxDiff=0,uniquenessRatio = 10):
    """
    calculate depth map given calibration mapping, L and R image, specscc

    """

    calibration = np.load(calibfile, allow_pickle=False)
    imageSize = tuple(calibration["imageSize"])
    leftMapX = calibration["leftMapX"]
    leftMapY = calibration["leftMapY"]
    leftROI = tuple(calibration["leftROI"])
    rightMapX = calibration["rightMapX"]
    rightMapY = calibration["rightMapY"]
    rightROI = tuple(calibration["rightROI"])

    fixedLeft = cv2.remap(imgL, leftMapX, leftMapY, cv2.INTER_LANCZOS4)
    fixedRight = cv2.remap(imgR, rightMapX, rightMapY, cv2.INTER_LANCZOS4)

    fixedLeft = fixedLeft[0:717,181:181+946]
    fixedRight = fixedRight[0:717,181:181+946]

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    
    stereoMatcher = cv2.StereoSGBM_create(MinDisp,NumDisp,BSize)
    
    stereoMatcher.setMinDisparity(MinDisp)
    stereoMatcher.setNumDisparities(NumDisp)
    stereoMatcher.setBlockSize(BSize)
    stereoMatcher.setSpeckleRange(SpecRange)
    stereoMatcher.setSpeckleWindowSize(SpecWindow)
    stereoMatcher.setP1(P1)
    stereoMatcher.setP2(P2)
    stereoMatcher.setPreFilterCap(PreFilterCap)
    stereoMatcher.setDisp12MaxDiff(disp12MaxDiff)
    stereoMatcher.setUniquenessRatio(uniquenessRatio)
    stereoMatcher.setMode(3)
    
    lmbda = 80000
    sigma2 = 100
    sigma = 3.0
    """
    rightMatcher = cv2.ximgproc.createRightMatcher(stereoMatcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoMatcher)
    left_depth = stereoMatcher.compute(grayLeft, grayRight)
    right_depth = rightMatcher.compute(grayRight,grayLeft)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    left_depth = np.int16(left_depth)
    right_depth = np.int16(right_depth)
    depth = wls_filter.filter(left_depth,grayLeft,None,right_depth)
    """
    depth = stereoMatcher.compute(grayLeft, grayRight)
    depth = cv2.normalize(depth, None, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    depth = np.uint8(depth)
    depth = cv2.fastNlMeansDenoising(depth,None,40,7,21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kernel)
    depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel2)
    depth = cv2.medianBlur(depth,3)

    cv2.imwrite(outfile,depth)
    cv2.imwrite('depth/grayRight.png',grayRight)
    cv2.imwrite('depth/grayLeft.png',grayLeft)


##########################################
##              Run Script              ##
##########################################

def main(): 
    """
    make get_parameters true to run calibration mapping
    make calculate true to run depth map calculation

    """


    get_parameters = False
    calculate = True
    limit = 230

    #define paths
    pathleft = 'calibrate/left/'
    pathright = 'calibrate/right/'
    testleft = 'left_calibrate_test.png'
    testright = 'right_calibrate_test.png'
    outname_prefix = 'depth/depth_map'
    parameters_name = 'stereo_camera_calibration_parameters_front'

    count = 0

    #run calibration test first
    #stereo_calibrate_test(pathleft,testleft,pathright,testright)


    if get_parameters == True:
        stereo_calibrate(pathleft,pathright)


    if calculate == True:
        left = cv2.VideoCapture("inputvideos/carf_frontL.mp4")
        right = cv2.VideoCapture("inputvideos/carf_frontR.mp4")

    while count <= limit:
        # Read in the left and right images
        success_l, image_l = left.read()
        success_r, image_r = right.read()
        #outname
        outname = outname_prefix + str(count) + '.png'

        calculate_depth_map(outname,parameters_name + '.npz',image_l,image_r)

        count += 1
        if count%20 == 0:
            print([str(count) + '/' + str(limit)])



    video = cv2.VideoWriter('depth_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),24,(946,717),False)
    for j in range(0,limit):
        img = cv2.imread(outname_prefix + str(j) + '.png')
        video.write(img)
    video.release()
    
if __name__=="__main__":
    main()