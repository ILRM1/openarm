import numpy as np

# head_img_width = 1920     
# head_img_height = 1536
    
hfov = float(np.deg2rad(115))         # Horizontal field of view
vfov = float(np.deg2rad(90.5))          # Vertical field of view
efl_mm = 3.          # Effective focal length
# max_distortion = -0.793         # -3% barrel (negligible → pinhole is accurate)

# FX = float((head_img_width  / 2.0) / np.tan(hfov / 2.0))
# FY = float((head_img_height / 2.0) / np.tan(vfov / 2.0))
# CX = head_img_width  / 2.0
# CY = head_img_height / 2.0

# head_cam_intrinsic_matrix = [[FX, 0.0, CX], 
#                             [0.0, FY, CY], 
#                             [0.0, 0.0, 1.0]]

horizontal_aperture = float(2.0 * efl_mm * np.tan(hfov / 2.0))
vertical_aperture = float(2.0 * efl_mm * np.tan(vfov / 2.0))

print(horizontal_aperture, vertical_aperture)