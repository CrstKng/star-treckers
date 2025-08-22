#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import os
import re
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from take_image import TakeImage

#install opencv-contrib-python

fov_deg =52
def get_intrinsics (img):
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)# Approx horizontal FOV
    fy = h / (2.0 * np.tan(fov_rad / 2.0))      # fov_rad = horizontal FOV
    fx = fy                      # scale for vertical pixels
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

def processing_image (img):
    # print(f"DEBUG image path is: {img} ")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY )   #trasform it in grayscale(the function works using only brightness)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)  
    retval,image_processed = cv.threshold(blurred,150, 255, cv.THRESH_BINARY)
    return image_processed


def detector(img,w,h,max_corners,quality_level,min_distance): 

    p0 = cv.goodFeaturesToTrack(img, 
                            maxCorners=max_corners, 
                            qualityLevel=quality_level, 
                            minDistance=min_distance)
    return p0

#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 150, 0.01)
#cv.cornerSubPix(gray, p0, (5,5), (-1,-1), criteria)

def pixel_to_unit_vector(points, cx, cy, fx, fy):
    vectors = []
    for u,v in points:
        x = (u - cx)/fx
        y = (v - cy)/fy # flip Y to make up = north
        z = 1
        vec = np.array([x, y, z])
        #vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

def estimate_rotation(v1, v2):
    v1_mean = np.mean(v1, axis=0)
    v2_mean = np.mean(v2, axis=0)   #Find the average vector in each set.('axis=0'=>computes the mean along the rows, for each column separately.)

    v1_centered = v1 - v1_mean
    v2_centered = v2 - v2_mean  #makes them zero-centered for covariance calculation.

    H = v1_centered.T @ v2_centered #covariance matrix
    
    #compute singular value decomposition
    U, S, Vt = np.linalg.svd(H)  #H= U * S * Vt.(U and Vt are orthogonal matrices, S diagonal.)

    #compute the best matrix that aligns v1 and v2
    R_matrix = Vt.T @ U.T

    #correction in case of reflection
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    rot = R.from_matrix(R_matrix)
    return rot

lk_params = dict( winSize = (31, 31),
                maxLevel =2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,150, 0.001),
                flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold=1e-4)     

def draw_center_to_star_vectors(img, points, cx, cy, color=(255, 0, 0)):
    """
    Draw arrows from the image center to each detected star (feature point).
    """
    img_out = img.copy()

    for pt in points:
        x, y = pt.ravel()
        cv.arrowedLine(img_out, (int(cx), int(cy)), (int(x), int(y)), color, 2, tipLength=0.2)
        cv.circle(img_out, (int(x), int(y)), 3, (0, 255, 0), -1)  # mark the star

    return img_out

# ---------- Add these helpers near the top ----------
def pixel_to_cam_ray(points, cx, cy, fx, fy, flip_y=True):
    """
    Convert pixels -> unit rays in camera frame using a pinhole model.
    If your image 'up' is north, set flip_y=True so +y_cam points up.
    points: Nx2 array of [u,v]
    """
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
    x, y, z = vec
    ra = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(z/np.linalg.norm(vec)))
    return ra, dec

def hms_to_deg(h, m, s):
    return (h + m/60 + s/3600) * 15

def dms_to_deg(d, m, s):
    sign = 1 if d >= 0 else -1
    return sign * (abs(d) + m/60 + s/3600)

import numpy as np

def radec_to_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    vec = np.array([x, y, z], dtype=np.float64)
    vec /= np.linalg.norm(vec)   # normalize to unit vector
    return vec   # shape (3,)


def build_cam_to_eq(RA0_deg, DEC0_deg, roll_deg=0.0):
    """
    Build rotation matrix mapping camera axes -> equatorial axes.
    +Z_cam points to boresight (RA0,DEC0), +X_cam is east, +Y_cam is north.
    roll_deg rotates camera around boresight.
    """
    z_eq = radec_to_vector(RA0_deg, DEC0_deg)            # boresight in EQ

    ra = np.radians(RA0_deg); dec = np.radians(DEC0_deg)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=np.float64)
    east /= np.linalg.norm(east)

    north = np.cross(z_eq, east)
    north /= np.linalg.norm(north)

    if abs(roll_deg) > 1e-12:
        roll = R.from_rotvec(np.radians(roll_deg) * z_eq)
        east = roll.apply(east)
        north = roll.apply(north)

    x_eq = east
    y_eq = north
    R_cam2eq = np.column_stack([x_eq, y_eq, z_eq])  # columns are cam axes in EQ
    return R_cam2eq


def draw_center_to_star_vectors(img, points, cx, cy, color=(0, 255, 0)):
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

def track_with_fb_check(img_prev, img_curr, p_prev, lk_params, fb_thresh):
    """
    Tracks features from img_prev → img_curr with LK optical flow,
    then does a forward-backward check to keep only reliable matches.
    """
    # Forward flow
    p_next, st, err = cv.calcOpticalFlowPyrLK(img_prev, img_curr, p_prev, None, **lk_params)
    if p_next is None:
        return np.empty((0,2)), np.empty((0,2))  # nothing tracked

    good_prev = p_prev[st == 1]
    good_next = p_next[st == 1]

    if len(good_next) == 0:
        return np.empty((0,2)), np.empty((0,2))

    # Backward flow
    p_back, st_back, err_back = cv.calcOpticalFlowPyrLK(img_curr, img_prev, good_next, None, **lk_params)
    p_back   = p_back.reshape(-1,2)
    st_back  = st_back.reshape(-1)

    # Forward-backward error (ensure diff is 1D)
    diff = np.linalg.norm(good_prev - p_back, axis=1)  # <-- 1D now
    good_indices = diff < fb_thresh
    good_prev = good_prev[good_indices]
    good_next = good_next[good_indices]
    

    return good_prev, good_next

def extract_ra_dec(filename):
    # Extract RA and DEC numbers from the filename using regex
    # Example filename: "0RA_0.3DEC_FOV(52.3).png"
    match = re.search(r'([-\d.]+)RA_([-\d.]+)DEC', filename)
    if match:
        ra = float(match.group(1))
        dec = float(match.group(2))
        return ra, dec
    else:
        return float('inf'), float('inf')  # if not found, put at the end





# HIP:112961 , star 0 
ra0_deg = hms_to_deg(22, 53, 58.6)
dec0_deg = dms_to_deg(-7, 26, 26.8)

#HIP:5364 ,star 2: 
ra2_deg = hms_to_deg(1, 9, 54)
dec2_deg = dms_to_deg(-10, 2, 35.4)

#HIP:6537 star 3 
ra3_deg=hms_to_deg(1, 25, 19.4)
dec3_deg = dms_to_deg(-8, 2, 51.9)

#HIP:112748 star 4
ra4_deg=hms_to_deg(22,51,15.8)
dec4_deg = dms_to_deg(24,44,17.8)

#HIP: 114971 star 5 
ra5_deg=hms_to_deg(23,18,31.1)
dec5_deg = dms_to_deg(3,25,28.3)

#HIP:1562 star 6 
ra6_deg=hms_to_deg(0,20,45.4)
dec6_deg = dms_to_deg(-8,40,44)

#HIP:112447 star 7
ra7_deg=hms_to_deg(22,47,59.9)
dec7_deg = dms_to_deg(12,18,22.3)

#HIP:112440 star 8
ra8_deg=hms_to_deg(22,47,47.4)
dec8_deg = dms_to_deg(23,42,7)

#HIP:115438 star 9
ra9_deg=hms_to_deg(23,24,20.5)
dec9_deg = dms_to_deg(-19,57,27.7)

#HIP:118268 star 11
ra11_deg=hms_to_deg(0,0,39)
dec11_deg = dms_to_deg(7,0,26.1)

#HIP:1067 star 12
ra12_deg=hms_to_deg(0,14,34.7)
dec12_deg = dms_to_deg(15,19,39.3)

#HIP:113136 star 13
ra13_deg=hms_to_deg(22,56,2.1)
dec13_deg = dms_to_deg(-15,40,54.8)

#HIP:112029 star 15
ra15_deg=hms_to_deg(22,42,45.8)
dec15_deg = dms_to_deg(10,58,1.4)

#HIP:112716 star 17
ra17_deg=hms_to_deg(22,50,58.4)
dec17_deg = dms_to_deg(-13,27,17.3)

#HIP:114724 star 22
ra22_deg=hms_to_deg(23,15,40.5)
dec22_deg = dms_to_deg(-5,54,29.9)

#HIP:4906 star 24
ra24_deg=hms_to_deg(1,4,17.6)
dec24_deg = dms_to_deg(8,1,48.2)

#HIP:3419 star 25
ra25_deg=hms_to_deg(0,44,53.9)
dec25_deg = dms_to_deg(-17,50,32.5)

#HIP:114855 star 28
ra28_deg=hms_to_deg(23,17,15.5)
dec28_deg = dms_to_deg(-8,56,43.7)


#set up equatorial coordinates of stars from img1
v0_eq  = radec_to_vector(ra0_deg, dec0_deg)
v2_eq  = radec_to_vector(ra2_deg, dec2_deg)
v3_eq  = radec_to_vector(ra3_deg, dec3_deg)
v4_eq  = radec_to_vector(ra4_deg, dec4_deg)
v5_eq  = radec_to_vector(ra5_deg, dec5_deg)
v6_eq  = radec_to_vector(ra6_deg, dec6_deg)
v7_eq  = radec_to_vector(ra7_deg, dec7_deg)
v8_eq  = radec_to_vector(ra8_deg, dec8_deg)
v9_eq  = radec_to_vector(ra9_deg, dec9_deg)
v11_eq = radec_to_vector(ra11_deg, dec11_deg)
v12_eq = radec_to_vector(ra12_deg, dec12_deg)
v13_eq = radec_to_vector(ra13_deg, dec13_deg)
v15_eq = radec_to_vector(ra15_deg, dec15_deg)
v17_eq = radec_to_vector(ra17_deg, dec17_deg)
v22_eq = radec_to_vector(ra22_deg, dec22_deg)
v24_eq = radec_to_vector(ra24_deg, dec24_deg)
v25_eq = radec_to_vector(ra25_deg, dec25_deg)
v28_eq = radec_to_vector(ra28_deg, dec28_deg)

if __name__ == "__main__":
  
###what i need to change:

# - step 1 needs to take the first image and set up the camera to eq rotational matrix and all initial stuff
# - step 2 needs to contain a while loop that takes a picture every time and defines previous and current frame 


#### Step 1: calibration from first image

    #take img 1 and set as first image to start off loop
    ImgMgr = TakeImage()
    image_path_0 = ImgMgr.take_first_image()
    print("DEBUG: you are taking the first image")
    img0 = cv.imread(image_path_0)

    ###make a list of unprocessed images and start it with img0 :
    images = []
    images.append(img0)

    #create list of processed images and start with img0:
    processed_images = []
    processed_img0 = processing_image(img0)
    processed_images.append(processed_img0)

    #get intrisic data from img1
    cx,cy,fx,fy=get_intrinsics(img0)

    h, w =  img0.shape[:2]
    print(f"Width of initial image: {w}, Height of initial image: {h}")
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)


    #get p0 from the initial image, which is img1:
    p0=detector(processed_images[0],w,h,max_corners = 1000,quality_level =0.001,min_distance =20)
    #print ('p0',p0)
    
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


    #using p0, get coordinates of stars from img1 in camera system:

    v0_cam  = pixel_to_cam_ray(np.array([p0[0,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v2_cam  = pixel_to_cam_ray(np.array([p0[2,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v3_cam  = pixel_to_cam_ray(np.array([p0[3,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v4_cam  = pixel_to_cam_ray(np.array([p0[4,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v5_cam  = pixel_to_cam_ray(np.array([p0[5,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v6_cam  = pixel_to_cam_ray(np.array([p0[6,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v7_cam  = pixel_to_cam_ray(np.array([p0[7,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v8_cam  = pixel_to_cam_ray(np.array([p0[8,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v9_cam  = pixel_to_cam_ray(np.array([p0[9,0]]),  cx, cy, fx, fy, flip_y=False)[0]
    v11_cam = pixel_to_cam_ray(np.array([p0[11,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v12_cam = pixel_to_cam_ray(np.array([p0[12,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v13_cam = pixel_to_cam_ray(np.array([p0[13,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v15_cam = pixel_to_cam_ray(np.array([p0[15,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v17_cam = pixel_to_cam_ray(np.array([p0[17,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v22_cam = pixel_to_cam_ray(np.array([p0[22,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v24_cam = pixel_to_cam_ray(np.array([p0[24,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v25_cam = pixel_to_cam_ray(np.array([p0[25,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v28_cam = pixel_to_cam_ray(np.array([p0[28,0]]), cx, cy, fx, fy, flip_y=False)[0]

    #function for displaying img1 with stars
    img_display = img0.copy()
    for pt in p0:
        x, y = pt.ravel()
        cv.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)

    #print image with coordinates from img1
    cv.imshow("Stars in img1", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ### Draw vectors + labels from image center to each detected star on img1
    img_display = draw_center_to_star_vectors(img0, p0.reshape(-1, 2), cx, cy)
    cv.imshow("Stars with Vectors in img1", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #equatorial coordinates of stars from img1:
    v_eq = np.vstack([v0_eq, v2_eq, v3_eq, v4_eq, v5_eq, v6_eq, v7_eq, v8_eq, v9_eq, v11_eq, v12_eq, v13_eq, v15_eq, v17_eq, v22_eq, v24_eq, v25_eq, v28_eq])
    
    #camera coordinates of stars from img1:
    v_cam_p0 = np.vstack([v0_cam, v2_cam, v3_cam, v4_cam, v5_cam, v6_cam, v7_cam, v8_cam, v9_cam, v11_cam, v12_cam, v13_cam, v15_cam, v17_cam, v22_cam, v24_cam, v25_cam, v28_cam])

    #create rotational matrix from equatorial system to camera system using image 1
    #get also inverse of matrix for camera to equatorial 
    rot_eq_to_cam= estimate_rotation(v_eq, v_cam_p0)
    rot_cam_to_eq = rot_eq_to_cam.inv()
    #print("Rotation matrix:\n", rot_eq_to_cam.as_matrix())
    
#### Step 2: iterate over the rest of the images

    #create variable for tracking angle of movement through adding the increments between every frame
    total_angle_deg = 0.0

    #i want to:
    # - take an image by calling the function & save it to the "N_camera_images" folder
    # - add the image to the list 
    # - process that image

    # calibration from first image

    rot_prev_to_eq = rot_cam_to_eq   
    p_prev = p0

    # forward-backward tolerance in pixels

    fb_thresh = 0.005 * max(w, h)    

    #for getting total angle from initial and final image
    rot_first_to_eq = rot_prev_to_eq # keep track of the very first orientation
    rot_last_to_eq  = rot_prev_to_eq # will be updated each step

    taking_photos = True
    while (taking_photos == True):
        #take a photo with the camera 
        image_path_curr = ImgMgr.take_images_continually()

        #check if need to exit while
        if image_path_curr is None:
            print("Exit requested — stopping image capture.")
            taking_photos = False
            break  # Optional: break out of loop immediately

        # now you can read the image
        img_curr = cv.imread(str(image_path_curr))

        

        #add image to list
        images.append(img_curr)

        #current image is set now; 
        #previous image is set using the index before the index of the current image from the list 
        #the first time the code runs, previous image will be img1
        img_prev = images[-2]


        ####### no need to name images, but need to count them maybe?
        
        # Forward-backward tracking
        good_prev, good_curr = track_with_fb_check(img_prev, img_curr, p_prev, lk_params, fb_thresh)


        # Convert pixel positions to camera-frame rays
        v_prev = pixel_to_cam_ray(good_prev, cx, cy, fx, fy, flip_y=False)
        v_curr = pixel_to_cam_ray(good_curr, cx, cy, fx, fy, flip_y=False)

        # get relative rotation from frame i-1 to i
        rot_prev_to_curr = estimate_rotation(v_prev, v_curr)

        # relative rotation in equatorial system (change of basis)
        rot_prev_to_curr_eq = rot_prev_to_eq * rot_prev_to_curr * rot_prev_to_eq.inv()

        # extract angle of rotation between frame i-1 and i
        rotvec = rot_prev_to_curr_eq.as_rotvec()
        angle_rad = np.linalg.norm(rotvec)
        angle_deg = np.degrees(angle_rad)

        print(f"Frame: rotation angle in equatorial system = {angle_deg:.6f} degrees")

        # 1) accumulate total angle of movement
        total_angle_deg += angle_deg

        # Update orientation of current frame in equatorial system
        rot_curr_to_eq = rot_prev_to_curr * rot_prev_to_eq

        # 2) update "last orientation"
        rot_last_to_eq = rot_curr_to_eq

        # Convert tracked stars into RA/Dec
        v_eq_curr = rot_curr_to_eq.apply(v_curr)
        coords_curr = np.array([vector_to_radec(v) for v in v_eq_curr])

        # Prepare for next iteration
        p_prev = good_curr.reshape(-1,1,2).astype(np.float32)
        rot_prev_to_eq = rot_curr_to_eq

        k = cv.waitKey(1) & 0xFF
        if k in (27, ord(' ')):
            taking_photos = False



#total degree of movement from increments:
print(f"Total rotation across all frames = {total_angle_deg:.6f} degrees")

#total degree of movement between initial and final:
rot_first_to_last_eq = rot_last_to_eq * rot_first_to_eq.inv()
net_rotvec = rot_first_to_last_eq.as_rotvec()
net_angle_deg = np.degrees(np.linalg.norm(net_rotvec))

print(f"Net rotation (first → last frame) = {net_angle_deg:.6f} degrees")

###what Step 2 should ouput:
# - the angle difference between every previous and current image
# - the equatorial coordinates of every "current" image
# - the equatorial coordinate of the center of last given image 

    













