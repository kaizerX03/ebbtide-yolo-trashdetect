import cv2
import numpy as np
import glob
import os
import yaml

# Chessboard calibration parameters
chessboard_size = (6, 9)  # (rows, columns) of inner corners
square_size = 2.4  # cm
image_dir = 'calibration_images'
output_dir = 'calibration_output'
os.makedirs(output_dir, exist_ok=True)

# Prepare object points (0,0,0), (0,1,0), ... (5,8,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob(os.path.join(image_dir, '*.jpg'))
print(f'Found {len(images)} images for calibration.')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        out_path = os.path.join(output_dir, f'corners_{os.path.basename(fname)}')
        cv2.imwrite(out_path, img)
        print(f'Chessboard detected in {fname}')
    else:
        print(f'Chessboard NOT detected in {fname}')

if len(objpoints) < 5:
    print('Not enough valid images for calibration. Need at least 5.')
    exit(1)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('Calibration successful.')
print('Camera matrix:')
print(mtx)
print('Distortion coefficients:')
print(dist.ravel())

# Save calibration results
np.savez(os.path.join(output_dir, 'calibration_data.npz'), camera_matrix=mtx, dist_coeffs=dist)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)
print(f'Total reprojection error: {mean_error:.4f}')

# Save calibration results in readable YAML format
calib_data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeffs': dist.ravel().tolist(),
    'reprojection_error': float(mean_error),
    'rotation_vectors': [rvec.tolist() for rvec in rvecs],
    'translation_vectors': [tvec.tolist() for tvec in tvecs],
    'image_size': list(gray.shape[::-1])
}
with open(os.path.join(output_dir, 'calibration_data.yaml'), 'w') as f:
    yaml.dump(calib_data, f)
print(f"Calibration data saved to {os.path.join(output_dir, 'calibration_data.yaml')}")
