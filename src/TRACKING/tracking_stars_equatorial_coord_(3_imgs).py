#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
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
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #trasform it in grayscale(the function works using only brightness)
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




if __name__ == "__main__":
    img1= cv.imread(r"src\test\test_images\Tracking_test\0RA_0DEC_FOV(52.3).png")
    img2= cv.imread(r"src\test\test_images\Tracking_test\0RA_0.1DEC_FOV(52.3).png")
    img3=cv.imread(r'src\test\test_images\Tracking_test\0RA_0.3DEC_FOV(52.3).png')

    cx,cy,fx,fy=get_intrinsics(img1)
    img_processed1=processing_image(img1)
    img_processed2=processing_image(img2)
    img_processed3=processing_image(img3)
    h, w =  img1.shape[:2]
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    p0=detector(img_processed1,w,h,max_corners = 1000,quality_level =0.001,min_distance =5)
    
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

    #function for displaying img1 with stars
    img_display = img1.copy()
    for pt in p0:
        x, y = pt.ravel()
        cv.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)

    
    cv.imshow("Stars in img1", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    p1, st, err = cv.calcOpticalFlowPyrLK(img_processed1,img_processed2, p0, None, **lk_params)
    if p1 is None:
        print("Optical flow tracking failed to find points in second image.")
    else:
        tracked_points = np.sum(st)
        print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")
    
    good_old = p0[st == 1]  #filters only the points that were successfully tracked in the first image
    good_new = p1[st == 1]
    
    p0r, st_back, err_back = cv.calcOpticalFlowPyrLK(img_processed2,img_processed1, good_new, None, **lk_params)
    flow_error_thresh = 0.005 * max(w, h)
    p0r = p0r.reshape(-1, 2)
    st_back = st_back.reshape(-1)

    diff = np.linalg.norm(good_old - p0r, axis=1)
    good_indices = diff < flow_error_thresh
    good_old = good_old[good_indices]
    good_new = good_new[good_indices]

    
    #printing predicted positions of stars in img2 
    img_tracked = img2.copy() #copy of the second image to draw the motion vectors
    for old, new in zip(good_old,good_new):
        a, b = new.ravel()  #to access x and y as individual variables
        c, d = old.ravel()
        cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        cv.circle(img_tracked, (int(a), int(b)), 3, (0, 0, 255), -1)

    # --- Step 5: Display the result ---
    cv.imshow("Predicted stars in img2 (Lucas-Kanade)", img_tracked)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_tracking.jpg", img_tracked)
    img_center_vecs2 = draw_center_to_star_vectors(img2, good_new, cx, cy)
    cv.imshow("Center-to-Star Vectors (img2)", img_center_vecs2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    p2, st2, err2 = cv.calcOpticalFlowPyrLK(img_processed2,img_processed3, p1, None, **lk_params)
    if p2 is None:
        print("Optical flow tracking failed to find points in second image.")
    else:
        tracked_points = np.sum(st)
        print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")
    
    p2 = p2[st == 1]
    
    #Apply LK algorithm to p2 =>get the coordinates of the stars in the first image (p1r) and compare them with p1=> use only the point for which p0r-p0< certain error 
    p1r, st_back, err_back = cv.calcOpticalFlowPyrLK(img_processed3,img_processed2, p2, None, **lk_params)
    flow_error_thresh = 0.005 * max(w, h)
    p1r = p1r.reshape(-1, 2)
    st_back = st_back.reshape(-1)

    diff = np.linalg.norm(p1 - p1r, axis=1)
    good_indices = diff < flow_error_thresh
    good_old1 = p1.reshape(-1, 2)[good_indices]
    good_new = p1.reshape(-1, 2)[st.flatten() == 1]

    #printing predicted positions of stars in img3
    img_tracked = img3.copy() #copy of the second image to draw the motion vectors
    for old, new in zip(p1,p2):
        a, b = new.ravel()  #to access x and y as individual variables
        c, d = old.ravel()
        cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (255, 0, 0), 2)
        cv.circle(img_tracked, (int(a), int(b)), 3, (255, 0, 0), -1)

   #Display the results
    cv.imshow("Predicted stars in img3 (Lucas-Kanade)", img_tracked)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_tracking.jpg", img_tracked)
    
    img_center_vecs3 = draw_center_to_star_vectors(img3, p2, cx, cy,color=150)
    cv.imshow("Center-to-Star Vectors (img3)", img_center_vecs3)
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
    
     #HIP:5364 star 3 Ra:h min s De:deg ' ''
    ra4_deg=hms_to_deg(1,9,51.18)
    dec4_deg = dms_to_deg(-10,3,3.3)
    
     #HIP :112158 star 8 Ra:h min s De:deg ' ''
    ra5_deg=hms_to_deg(22,44,9.59)
    dec5_deg = dms_to_deg(30,21,14.5)
    
      #HIP:115438 star 24 Ra:h min s De:deg ' ''
    ra6_deg=hms_to_deg(23,24,16.61)
    dec6_deg = dms_to_deg(-19, 5, 58.3)
    
      #HIP:114971 star 7 Ra:h min s De:deg ' ''
    ra7_deg=hms_to_deg(23,18,27.24)
    dec7_deg = dms_to_deg(3,25,8.7)
    
    
    #call radec to vector function
    v1_eq= radec_to_vector(ra1_deg,dec1_deg)
    v2_eq=radec_to_vector(ra2_deg,dec2_deg)
    v3_eq= radec_to_vector(ra3_deg,dec3_deg)
    v4_eq=radec_to_vector(ra4_deg,dec4_deg)
    v5_eq=radec_to_vector(ra5_deg,dec5_deg)
    v6_eq=radec_to_vector(ra6_deg,dec6_deg)
    v7_eq=radec_to_vector(ra7_deg,dec7_deg)
    
  
    
    v1_cam = pixel_to_cam_ray(np.array([p0[0,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v2_cam = pixel_to_cam_ray(np.array([p0[1,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v3_cam = pixel_to_cam_ray(np.array([p0[2,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v4_cam= pixel_to_cam_ray(np.array([p0[3,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v5_cam= pixel_to_cam_ray(np.array([p0[8,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v6_cam= pixel_to_cam_ray(np.array([p0[24,0]]), cx, cy, fx, fy, flip_y=False)[0]
    v7_cam= pixel_to_cam_ray(np.array([p0[7,0]]), cx, cy, fx, fy, flip_y=False)[0]
    
    

    print ('p0',p0)
    print('coordinates of first 2 stars:',p0[0,0],p0[1,0])
    print('vcam:',v1_cam,v2_cam)

    #print equatorial coordinates of stars from img1
    print('v_eq',v1_eq,v2_eq,v3_eq,v4_eq,v5_eq,v6_eq,v7_eq)    
    v_eq = np.vstack([v1_eq, v2_eq,v3_eq,v4_eq,v5_eq,v6_eq,v7_eq])
    
    #print camera coordinates of stars from img1:
    v_cam_p0 = np.vstack([v1_cam, v2_cam,v3_cam,v4_cam,v5_cam,v6_cam,v7_cam])
    print("v_cam_p0, selected:",v_cam_p0)

    #create rotational matrix from equatorial system to camera system using image 1
    #get also inverse of matrix for camera to equatorial 
    rot_eq_to_cam= estimate_rotation(v_eq, v_cam_p0)
    rot_cam_to_eq = rot_eq_to_cam.inv()
    print("Rotation matrix:\n", rot_eq_to_cam.as_matrix())
    
    #apply rotational matrix to equatorial coordinates of stars in image 1 to test accuracy
    #if the predicted coordinates are the same as the real coordinates, it works 
    predicted_cam = rot_eq_to_cam.apply(v_eq)
    print("Predicted camera vectors:\n", predicted_cam)
    print("Actual camera vectors:\n", v_cam_p0)

    # measure error of rotational matrix from camera to equatorial, created using img1
    errors = np.linalg.norm(predicted_cam - v_cam_p0, axis=1)
    print("Errors per star:", errors)

    #creating the vectors in camera system for img2
    v_cam_p1 = []
    for i in range(len(p1)):
        vi_cam = pixel_to_cam_ray(np.array([p1[i,0]]), cx, cy, fx, fy, flip_y=False)[0]
        v_cam_p1.append(vi_cam)

    #print vector coordinates of stars from img2:
    v_cam_p1 = np.vstack(v_cam_p1)
    print("v_cam_p1, all:",v_cam_p1)

    #apply rotational matrix from camera to equatorial to v_cam_p1 to get equatorial coordinates of stars from img2
    v_eq_p1 = rot_cam_to_eq.apply(v_cam_p1)
    v_eq_p1 = np.array(v_eq_p1)
    print(v_eq_p1.shape)  # (N, 3)
    print("v_eq_p1:", v_eq_p1)

    #convert 3d vectors of stars in img2 to RA/DEC 
    #final_coords = eq coordinates of stars in img2
    final_coords = np.array([vector_to_radec(vec) for vec in v_eq_p1])
    print("final_coords:",final_coords)

### adding the third image ###

    #Get the camera coordinates of the stars in img3
    v_cam_p2 = pixel_to_cam_ray(p2.reshape(-1, 2), cx, cy, fx, fy, flip_y=False)
    print("v_cam_p2 shape:", v_cam_p2.shape)
    
    #get the rotational matrix between camera coord of image 2 and camera coord of image 3
    rot_cam1_to_cam2=estimate_rotation(v_cam_p1,v_cam_p2)
    print("Rotation matrix between cam1 and cam2:\n", rot_cam1_to_cam2.as_matrix())
    
    #Get the rotational matrix between camera coordinates of stars in img3 and the Celestial RF
    #do this by multiplying (rotation between img2 and img3) * (rotation between camera RF and equatorial RF)
    rot_cam2_to_eq=rot_cam1_to_cam2 * rot_cam_to_eq
    print("Rotation matrix between cam 2 and celestial RF:\n", rot_cam2_to_eq.as_matrix())
    
    #print equatorial coordinates of stars from img3
    v_eq_p2=rot_cam2_to_eq.apply(v_cam_p2)
    print('v_eq_p2:',v_eq_p2)

    #final_coords_2 = eq coordinates of stars in img3
    final_coords2 = np.array([vector_to_radec(vec) for vec in v_eq_p2])
    print('Final coord2:',final_coords2)

    #to check( if they overlap=>it works)
    import matplotlib.pyplot as plt

    plt.scatter(final_coords[:,0], final_coords[:,1], label="Cam1->Eq", marker="o")
    plt.scatter(final_coords2[:,0], final_coords2[:,1], label="Cam2->Eq", marker="x")
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.legend()
    plt.show()



    
    
    '''
  
    #print("Quaternion [x, y, z, w]:", rot.as_quat())

    print("Euler angles [°]:", rot.as_euler('zyx', degrees=True))

    ra_delta = rot.as_euler('zyx', degrees=True)[1]  # Y-axis:  RA (for the open cv coordinate system)
    dec_delta = rot.as_euler('zyx', degrees=True)[0]  # Z-axis: DE

    print(f"ΔRA ≈ {ra_delta:.4f}°, ΔDEC ≈ {dec_delta:.4f}°")'''