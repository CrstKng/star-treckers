import sys
from pathlib import Path
import os
from datetime import datetime
from imu_readings import *
import time
from star_processing import process_star_image, capture_image 
from quaternion_calculations import propagate_orientation , quat_to_euler
import star_processing

import platform
if platform.system() == "Darwin":
     from publish_udp import OrientationPublisher, MAC_PORT, MAC_IP

# from imu_readings import get_quaternion , calibrate
DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def main(use_camera=False, image_path=DEFAULT_IMAGE_PATH):
     
     timer = time.time()
     if use_camera:
          print("Starting IMU Daemon...")
          start_imu_daemon()
          print("Waiting for IMU to calibrate...")
          while not get_calibration_status():
               print("Still calibrating... Waiting 1 second.")
               sys_cal, gyro_cal, accel_cal, mag_cal = get_detailed_calibration_status()
               print(f"Calibrating: SYS:{sys_cal} , GYRO: {gyro_cal} , ACCL: {accel_cal} MAG: {mag_cal}")
               time.sleep(1)

     quaternion_star, rotation_matrix = star_processing.process_star_image(use_camera=use_camera, visualize=True)
     print(f"Time taken: {time.time() - timer} seconds")
     if quaternion_star is not None and rotation_matrix is not None:
          print("Processing complete!")
          print("Quaternion:", quaternion_star)
          print("Rotation Matrix:", rotation_matrix)
     else:
          print("Failed to process image.")

#     delta_quaternion = calculate_delta_quaternion( quarterion_star, get_quaternion())
     if use_camera:
               useIMU= True
               useIMGTracking = True
               if useIMU:
                    from quaternion_calculations import quat2dict
                    Q_STAR_REF = quat2dict(quaternion_star)
                    Q_IMU_REF  = get_latest_quaterlion()   

                    publisher = OrientationPublisher(mac_ip=MAC_IP, mac_port=MAC_PORT)
                    while True:
                         Q_IMU_CURR = get_latest_quaterlion()  
                         Q_BODY_CURR = propagate_orientation(Q_STAR_REF, Q_IMU_REF, Q_IMU_CURR)
                         yaw, pitch, roll = quat_to_euler(Q_BODY_CURR)
                         print(
                              f"Quaternion : {Q_BODY_CURR}   |   "
                              f"Yaw ψ={yaw:6.1f}°  Pitch θ={pitch:6.1f}°  Roll φ={roll:6.1f}°"
                         )
                         publisher.send_quaternion(Q_BODY_CURR["w"], Q_BODY_CURR["x"], Q_BODY_CURR["y"], Q_BODY_CURR["z"])
                         time.sleep(0.25)
               if useIMGTracking:
                    pass
                    #put your tracking coder= calls here:
                    #CODE... 

if __name__ == "__main__":
     try:
          if len(sys.argv) > 1 and sys.argv[1] == "--capture":
               main(use_camera=True)
          elif len(sys.argv) > 1 and sys.argv[1] == "--test":
               from quaternion_calculations import quat_to_euler, quat2dict
               import numpy as np
               import time
               print("[TEST MODE] Simulating continuous rotation...")
               t = 0
               publisher = OrientationPublisher(mac_ip= MAC_IP, mac_port=MAC_PORT)
               while True:
                    angle_deg = 10 + t  # Start at 10°, increase by 1° each loop
                    angle_rad = np.deg2rad(angle_deg)
                    w = np.cos(angle_rad/2)
                    x = 0.0
                    y = 0.0
                    z = np.sin(angle_rad/2)
                    simulated_quat = {"w": w, "x": x, "y": y, "z": z}
                    yaw, pitch, roll = quat_to_euler(simulated_quat)
                    print(f"[TEST MODE] Simulated Quaternion: {simulated_quat}")
                    print(f"[TEST MODE] Yaw ψ={yaw:6.1f}°  Pitch θ={pitch:6.1f}°  Roll φ={roll:6.1f}°")
                    publisher.send_quaternion(w, x, y, z)
                    time.sleep(0.5)
                    t += 1
          else:
               main(use_camera=False)

     except KeyboardInterrupt:
          print("\nShutting down...")

     