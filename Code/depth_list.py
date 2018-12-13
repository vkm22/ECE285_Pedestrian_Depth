import numpy as np

def depth_list(lbbox,rbbox,depth):
    """
    Given depth image, list of left and right bbox coordinates and indexes,
    find depth per object and return according to index per image
    """
    out = np.zeros(20,3)
    k = 0
    for i in range(0,lbbox.shape[0]):
        for j in range((0,lbbox.shape[0]):
            l_obj = lbbox[i]
            r_obj = rbbox[j]

            l_x1 = l_obj[0]
            l_x2 = l_obj[2]
            l_y1 = l_obj[1]
            l_y2 = l_obj[3]
            r_x1 = r_obj[0]
            r_x2 = r_obj[2]
            r_y1 = r_obj[1]
            r_y2 = r_obj[3]

            l_length = abs(l_x2 - l_x1)
            l_width = abs(l_y2 - l_y1)
            l_centroid_x = l_x1 + 0.5*l_length 
            l_centroid_y = l_y1 + 0.5*l_width

            r_length = abs(l_x2 - l_x1)
            r_width = abs(l_y2 - l_y1)
            r_centroid_x = l_x1 + 0.5*l_length 
            r_centroid_y = l_y1 + 0.5*l_width

            thresh = 3

            if abs(l_length - r_length) < thresh & abs(l_width - r_width) < thresh & abs(l_centroid_x - r_centroid_x) < thresh & abs(l_centroid_y - r_centroid_y) < thresh:
                depth_x1 = (l_x1 + r_x1)/2.0
                depth_x2 = (l_x2 + r_x2)/2.0
                depth_y1 = (l_y1 + r_y1)/2.0
                depth_y2 = (l_y2 + r_y2)/2.0
                depth_l_index = i
                depth_r_index = j
                depth_obj_matrix = depth[depth_x1:depth_x2,depth_y1:depth_y2]
                depth_obj = np.mean(depth_obj_matrix[depth_obj_matrix > 0])
                out[k,:] = [depth_obj, depth_l_index, depth_r_index]

    
    return out