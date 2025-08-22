#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
#install opencv-contrib-python

fov_deg =52  #vertical FOV of the camera

def get_intrinsics (img):
    #INPUT:imagine of the stars to detect
    #OUTPUT:coordinates of the center of the image in pixels,focal lenght
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)
    fy = h / (2.0 * np.tan(fov_rad / 2.0))      
    fx = fy                  #squared pixels
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

def processing_image (img):
    #INPUT:imagine of the stars to detect
    #OUTPUT:imagine in grayscale (the function works using only brightness) and blurred
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  
    blurred = cv.GaussianBlur(gray, (3, 3), 1)  
    retval,image_processed = cv.threshold(blurred,150, 255, cv.THRESH_BINARY)
    return image_processed

#parameters used for the detection
lk_params = dict( winSize = (31, 31),   #size of the window around each star where the algorithm looks for motion.
                maxLevel =2,            #Lucas–Kanade can use a pyramidal approach: build image pyramids at smaller resolutions to track large motions
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,150, 0.001), #iterative solver when to stop refining: when solution change is smaller than 0.001 or after 150 iterations
                flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS, #compute the minimum eigenvalue of the gradient matrix, which is a feature strength measure
                minEigThreshold=1e-4)    #If the minimum eigenvalue < 1e-4, that point is rejected as too weak/unreliable.

def detector(img,w,h,max_corners,quality_level,min_distance): 
    #INPUT:imagine of the stars to detect with its width and height, parameters
    #OUTPUT: coordinates of the stars in pixel
    p0 = cv.goodFeaturesToTrack(img, 
                            maxCorners=max_corners, 
                            qualityLevel=quality_level, 
                            minDistance=min_distance)
    return p0

def pixel_to_unit_vector(points, cx, cy, fx, fy):
    #INPUT:coordinates of the stars in pixels, center of the image(origin of the camera RF),focal lenght
    #OUTPUT: vectors from the focal point to the stars
    vectors = []
    for u,v in points:
        x = (u - cx)/fx
        y = (v - cy)/fy 
        z = 1
        vec = np.array([x, y, z])
        #vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

def estimate_rotation(v1, v2):
    #INPUT:2 sets of vector 
    #OUTPUT:best matrix that aligns v1 and v2
    v1_mean = np.mean(v1, axis=0)
    v2_mean = np.mean(v2, axis=0)   #Find the average vector in each set.('axis=0'=>computes the mean along the rows, for each column separately.)

    v1_centered = v1 - v1_mean
    v2_centered = v2 - v2_mean      #makes them zero-centered for covariance calculation.

    H = v1_centered.T @ v2_centered #covariance matrix
    
    #compute singular value decomposition
    U, S, Vt = np.linalg.svd(H)     #H= U * S * Vt.(U and Vt are orthogonal matrices, S diagonal.)

    #compute the best matrix that aligns v1 and v2
    R_matrix = Vt.T @ U.T

    #correction in case of reflection
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    rot = R.from_matrix(R_matrix)
    return rot


def pixel_to_cam_ray(points, cx, cy, fx, fy, flip_y=True):
    #INPUT: coordinates of the stars(in pixel),center of the image, focal lenght, flip_y to change the convention
    #OUTPUT: Nx2 array of [u,v] (coordinates of the stars in the camera RF)
    #Convert pixels -> unit rays in camera frame using a pinhole model.

    rays = []
    for u, v in points.reshape(-1, 2):
        x = (u - cx) / fx
        y = (v - cy) / fy
        if flip_y:
            y = -y
        z = 1.0
        vec = np.array([x, y, z], dtype=np.float64)
        vec /= np.linalg.norm(vec)
        rays.append(vec)
    return np.vstack(rays)



def vector_to_radec(vec):
    #INPUT: vector in the celestial RF
    #OUTPUT: coordinates in the equatorial RF (Right Ascension and Declination)
    x, y, z = vec
    ra = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(z/np.linalg.norm(vec)))
    return ra, dec

def hms_to_deg(h, m, s):
     #INPUT: RA in h,m,s
    #OUTPUT: RA in deg
    return (h + m/60 + s/3600) * 15

def dms_to_deg(d, m, s):
    #INPUT: DEC in deg 
    #OUTPUT: DEC in deg (decimal notation)
    sign = 1 if d >= 0 else -1
    return sign * (abs(d) + m/60 + s/3600)


def radec_to_vector(ra_deg, dec_deg):
    #INPUT: RA and DEC in degree (coordinates in the equatorial RF)
    #OUTPUT: Vector in the celestial RF
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    vec = np.array([x, y, z], dtype=np.float64)
    vec /= np.linalg.norm(vec)   # normalize to unit vector
    return vec   # shape (3,)



def draw_center_to_star_vectors(img, points, cx, cy, color=(0, 255, 0)):
    #INPUT:imagine on which the vectors will be printed, coordinates of the stars(where the vectors will point),center of the image, colour used to draw
    #OUTPUT:Draw arrows from the image center to each detected star (feature point). and label them

    # Make a copy so we don’t overwrite the original
    img_out = img.copy()
    
    for i, (x, y) in enumerate(points):
        # Draw the line from image center to point
        cv.line(img_out, (int(cx), int(cy)), (int(x), int(y)), color, 1)

        # Draw a small circle at the star location
        cv.circle(img_out, (int(x), int(y)), 3, color, -1)

        # Put the index number near the star
        cv.putText(img_out, str(i), (int(x) + 5, int(y) - 5), 
            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return img_out




if __name__ == "__main__":
    img1= cv.imread(r"src\test\test_images\Tracking_test\0RA_0DEC_FOV(52.3).png")   #initial orientation
    img2= cv.imread(r"src\test\test_images\Tracking_test\0RA_0.1DEC_FOV(52.3).png") #final orientation
    img3=cv.imread("")
    
    cx,cy,fx,fy=get_intrinsics(img1)   
    img_processed1=processing_image(img1)
    img_processed2=processing_image(img2)
    img_processed3=
    h, w =  img1.shape[:2]
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    #Use detector to get the coordinates of the stars(in pixels) in the first image
    p0=detector(img_processed1,w,h,max_corners = 1000,quality_level =0.001,min_distance =5)

    #Exclude the stars on the edges 
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    filtered_p0 = []
    for pt in p0:
        x, y = pt.ravel()
        if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
            filtered_p0.append([[x, y]])
    filtered_p0 = np.array(filtered_p0, dtype=np.float32)
    p0=filtered_p0

    if p0 is None:
        print("No features found in the first image!")
    else:
        print("Number of features detected:",len(p0))
        
    
    #Display the results
    img_display = img1.copy()
    for pt in p0:
        x, y = pt.ravel()
        cv.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv.imshow("Good Features", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    #Using the Lucas Kanade Algorithm track the stars and get the coordinates of the stars in the second image (p1)
    p1, st, err = cv.calcOpticalFlowPyrLK(img_processed1,img_processed2, p0, None, **lk_params)
    if p1 is None:
        print("Optical flow tracking failed to find points in second image.")
    else:
        tracked_points = np.sum(st)
        print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")
    
    good_old = p0[st == 1]  #filters only the points that were successfully tracked in the first image
    good_new = p1[st == 1]
    
    #Apply LK algorithm to p1 =>get the coordinates of the stars in the first image (p0r) and compare them with p0=> use only the point for which p0r-p0< certain error 
    p0r, st_back, err_back = cv.calcOpticalFlowPyrLK(img_processed2,img_processed1, good_new, None, **lk_params)
    flow_error_thresh = 0.005 * max(w, h)
    p0r = p0r.reshape(-1, 2)
    st_back = st_back.reshape(-1)

    diff = np.linalg.norm(good_old - p0r, axis=1)
    good_indices = diff < flow_error_thresh
    good_old = good_old[good_indices]
    good_new = good_new[good_indices]

    
    img_tracked = img2.copy() #copy of the second image to draw the motion vectors
    for old, new in zip(good_old,good_new):
        a, b = new.ravel()  #to access x and y as individual variables
        c, d = old.ravel()
        cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        cv.circle(img_tracked, (int(a), int(b)), 3, (0, 0, 255), -1)

   #Display the results
    cv.imshow("Tracked Features (Lucas-Kanade)", img_tracked)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_tracking.jpg", img_tracked)
    img_center_vecs = draw_center_to_star_vectors(img2, good_new, cx, cy)
    cv.imshow("Center-to-Star Vectors (img2)", img_center_vecs)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
     #conversion in degree
# HIP: 113136, star 0 RA (on date): 22h 55m 57.99s,Dec (on date): −15° 41′ 23.0″
    ra1_deg = hms_to_deg(22, 55, 57.99)
    dec1_deg = dms_to_deg(-15, 41, 23.0)
    
    #HIP: 3092,RA star 1 (on date): 0h 40m 39.94s,Dec (on date): +31° 00′ 01.6″
    ra2_deg = hms_to_deg(0, 40, 39.94)
    dec2_deg = dms_to_deg(31, 0, 01.6)
    
    #HIP:1067 star 2 Ra:0h 40min 31.28s De:15deg 19' 25.4''
    ra3_deg=hms_to_deg(0, 40, 31.28)
    dec3_deg = dms_to_deg(15, 19, 25.4)
    
     #HIP:5364 star 3
    ra4_deg=hms_to_deg(1,9,51.18)
    dec4_deg = dms_to_deg(-10,3,3.3)
    
     #HIP :112158 star 8 
    ra5_deg=hms_to_deg(22,44,9.59)
    dec5_deg = dms_to_deg(30,21,14.5)
    
      #HIP:115438 star 24 
    ra6_deg=hms_to_deg(23,24,16.61)
    dec6_deg = dms_to_deg(-19, 5, 58.3)
    
      #HIP:114971 star 7 
    ra7_deg=hms_to_deg(23,18,27.24)
    dec7_deg = dms_to_deg(3,25,8.7)
    
    
    #call radec to vector function to get the vectors in the Celestial RF (this inputs should be replaced by the results of LIS)
    v1_eq= radec_to_vector(ra1_deg,dec1_deg)
    v2_eq=radec_to_vector(ra2_deg,dec2_deg)
    v3_eq= radec_to_vector(ra3_deg,dec3_deg)
    v4_eq=radec_to_vector(ra4_deg,dec4_deg)
    v5_eq=radec_to_vector(ra5_deg,dec5_deg)
    v6_eq=radec_to_vector(ra6_deg,dec6_deg)
    v7_eq=radec_to_vector(ra7_deg,dec7_deg)
    
    
    #get the vectors in the camera RF
    v1_cam = pixel_to_cam_ray(np.array([p0[0,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v2_cam = pixel_to_cam_ray(np.array([p0[1,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v3_cam = pixel_to_cam_ray(np.array([p0[2,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v4_cam= pixel_to_cam_ray(np.array([p0[3,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v5_cam= pixel_to_cam_ray(np.array([p0[8,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v6_cam= pixel_to_cam_ray(np.array([p0[24,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v7_cam= pixel_to_cam_ray(np.array([p0[7,0]]), cx, cy, fx, fy, flip_y=False)[0]
    
    

    print ('p0',p0)
    print('first',p0[0,0],p0[1,0])
    print('vcam:',v1_cam,v2_cam)

    print('v_eq',v1_eq,v2_eq,v3_eq,v4_eq,v5_eq,v6_eq,v7_eq)    
    v_eq = np.vstack([v1_eq, v2_eq,v3_eq,v4_eq,v5_eq,v6_eq,v7_eq])
    
    v_cam_p0 = np.vstack([v1_cam, v2_cam,v3_cam,v4_cam,v5_cam,v6_cam,v7_cam])
    print("v_cam_p0, selected:",v_cam_p0)

    #get the rotational matrix that best aligns the 2 RFs
    rot_eq_to_cam= estimate_rotation(v_eq, v_cam_p0) 
    rot_cam_to_eq = rot_eq_to_cam.inv()
    print("Rotation matrix:\n", rot_eq_to_cam.as_matrix())
    
    predicted_cam = rot_eq_to_cam.apply(v_eq)
    print("Predicted camera vectors:\n", predicted_cam)
    print("Actual camera vectors:\n", v_cam_p0)

    # measure error
    errors = np.linalg.norm(predicted_cam - v_cam_p0, axis=1)
    print("Errors per star:", errors)

    #constructing vector to center of the image in the camera RF
    v_center_cam = np.array([0, 0, 1])
    
    #apply the rotational matrix to get the vector in the Celestial RF
    v_center_eq = rot_cam_to_eq.apply(v_center_cam)
    v_center_eq = np.array(v_center_eq)
    
    #Tranform the coordinates in the Celestial RF in RA and DEC
    final_coords = np.array([vector_to_radec(v_center_eq)])
    
    print("final_coords:",final_coords)
    print("final eq coordinates of the center: ", final_coords)
    
  

