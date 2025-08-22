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
    fx = fy * (w / h)                       # scale for vertical pixels
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
        x = (u - cx)
        y = (v - cy) # flip Y to make up = north
        z = 0
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



if __name__ == "__main__":
    img1 = cv.imread(r"src/test/test_images/Tracking_test/0RA_0DEC_FOV(52.3).png")
    img2 = cv.imread(r"src/test/test_images/Tracking_test/0RA_0.1DEC_FOV(52.3).png")

    # img1= cv.imread(r"src\test\test_images\Tracking_test\0RA_0DEC_FOV(52.3).png")
    # img2= cv.imread(r"src\test\test_images\Tracking_test\0RA_0.1DEC_FOV(52.3).png")

    cx,cy,fx,fy=get_intrinsics(img1)
    img_processed1=processing_image(img1)
    img_processed2=processing_image(img2)
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
    img_display = img1.copy()
    for pt in p0:
        x, y = pt.ravel()
        cv.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)


    cv.imshow("Good Features", img_display)
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
    
    
    img_tracked = img2.copy() #copy of the second image to draw the motion vectors
    for old, new in zip(good_old,good_new):
        a, b = new.ravel()  #to access x and y as individual variables
        c, d = old.ravel()
        cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        cv.circle(img_tracked, (int(a), int(b)), 3, (0, 0, 255), -1)

    # --- Step 5: Display the result ---
    cv.imshow("Tracked Features (Lucas-Kanade)", img_tracked)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_tracking.jpg", img_tracked)
    

# --- Draw on second image using tracked points ---
    img_center_vecs = draw_center_to_star_vectors(img2, good_new, cx, cy)

    cv.imshow("Center-to-Star Vectors", img_center_vecs)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_center_vectors.jpg", img_center_vecs)

    img_center_vecs2=draw_center_to_star_vectors(img1, good_old, cx, cy,color=(0, 0, 255))
    cv.imshow("Center-to-Star Vectors", img_center_vecs2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_center_vectors.jpg", img_center_vecs2)
    
    displacement = good_new - good_old     # shape (N,2), each row is (dx, dy)
    deg_pixel = fov_deg / h
    real_displacement = displacement * deg_pixel

    # Compute total displacement (magnitude) for each vector
    magnitudes = np.linalg.norm(real_displacement, axis=1)

    # Print displacement for every vector
    for i, m in enumerate(magnitudes):
        print(f"Vector {i}: displacement = {m}")
    
    # Compute and print mean displacement
    mean_disp = np.mean(magnitudes)
    print(f"\nMean displacement = {mean_disp}")
    
    









