import os
import numpy as np
import pickle
import open3d as o3d
import json
from CPN import *
from absl import app
from absl import flags

flags.DEFINE_string('task', 'stack-block-pyramid', '')
flags.DEFINE_string('data_dir', './training_datasets/voxel_grids', '')
flags.DEFINE_string('data_source', './training_datasets/rgbd', '')
flags.DEFINE_integer('num_per_class', '100', '')


FLAGS = flags.FLAGS

def normalize(arr):
    """
    normalize an array to [0, 1]
    """
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def load_rgb_segm_depth(path, num):

#   rgb_image_batches = []
  segm_image_batches = []
  depth_image_batches = []
  class_paths = []
  file_name_batches = []

  sub_paths = os.listdir(path)

  data_types = ['positive', 'negative']

  img_types = ['segm', 'depth']

  for sub_path in sub_paths:
    for data_type in data_types:

    #   rgb_images = []
      segm_images = []
      depth_images = []
      file_names = []
      class_path = os.path.join(path, sub_path, data_type)
      print(class_path)

      for img_type in img_types:

        img_path = os.path.join(class_path, img_type)
        images = os.listdir(img_path)
        for i ,img_name in enumerate(images):
          if i == num:
            break
          
          with open(os.path.join(img_path, img_name), 'rb') as f:
            file_names.append(img_name[:img_name.find('.')])
            img = pickle.load(f)

        #   if img_type == 'color':
        #     rgb_images.append(img)
          if img_type == 'segm':
            segm_images.append(img)
          elif img_type == 'depth':
            depth_images.append(img)

    #   rgb_image_batches.append(rgb_images)
      segm_image_batches.append(segm_images)
      depth_image_batches.append(depth_images)
      file_name_batches.append(file_names)
      class_paths.append(class_path)
      
  return segm_image_batches, depth_image_batches, file_name_batches, class_paths

def get_segmented_voxel_grids(segm_img, depth_img):
  
  segm_img = np.repeat(segm_img[..., np.newaxis], 3, axis=2)
    
  rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
  o3d.geometry.Image(segm_img),
  o3d.geometry.Image(depth_img),
  convert_rgb_to_intensity=False)

  # x_dim = 50
  # y_dim = 45
  # z_dim = 100

  x_dim = 35
  y_dim = 30
  z_dim = 70

  # # xyz
  # camera_pose = np.array([[-6.12323400e-17, -1.00000000e+00,  1.22464680e-16, 1],
  #       [-7.07106781e-01,  1.29893408e-16,  7.07106781e-01, 0],
  #       [-7.07106781e-01, -4.32978028e-17, -7.07106781e-01, 0.75],
  #       [0,0,0,1]])

  #xzy
  camera_pose = np.array([[-6.12323400e-17, -1.22464680e-16,  1.00000000e+00,1 ],
  [-7.07106781e-01, -7.07106781e-01, -1.29893408e-16,0],
  [ 7.07106781e-01, -7.07106781e-01, -4.32978028e-17,0.75],
  [0,0,0,1]])

  intrinsic_matrix = np.array([[450., 0, 320.],
                                    [0, 450., 240.],
                                    [0, 0, 1]])
  camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsic_matrix)

  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd_img, camera_intrinsic, camera_pose)
  

  ids = np.unique(np.asarray(pcd.colors)[:,0])

  # o3d.visualization.draw_geometries([pcd])
  min_bound = pcd.get_min_bound()
  max_bound = pcd.get_max_bound()

  min_bound[0] += 0.000
  max_bound[0] -= 0.0018

  min_bound[1] += 0.00000105
  max_bound[1] -= 0.000275

  min_bound[2] += 0.0011
  max_bound[2] -= 0.0011

  cropping_bound = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

  # Crop the point cloud using the bounding box
  cropped_point_cloud = pcd.crop(cropping_bound)

  # o3d.visualization.draw_geometries([cropped_point_cloud])

  # Get point cloud coordinates
  points = np.asarray(cropped_point_cloud.points)

  # Get point cloud colors
  colors = np.asarray(cropped_point_cloud.colors)

  

  # Define voxel size
  voxel_size = 0.000015 #(35, 30, 70)
  # voxel_size = 0.0000105 #(50, 45, 100)

  # Calculate voxel grid dimensions
  max_bound = np.max(points, axis=0)
  min_bound = np.min(points, axis=0)
  dimensions = np.ceil((max_bound - min_bound) / voxel_size)

  print(f'\tdimentsions of downsampling scene: {dimensions}')
  # Calculate voxel grid indices for each point
  indices = np.floor((points - min_bound) / voxel_size)

  # Initialize voxel grid with zeros
  voxel_grid = np.zeros((int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), 3))

  # Fill voxel grid with colors
  for i in range(points.shape[0]):
      voxel_grid[int(indices[i][0]), int(indices[i][1]), int(indices[i][2])] = colors[i]

  # Convert voxel grid to 3D numpy array
  voxel_grid = np.asarray(voxel_grid)
  voxel_grid = np.pad(voxel_grid, ((0, max(0,x_dim-voxel_grid.shape[0])), (0, max(0,y_dim-voxel_grid.shape[1])), (0, max(0,z_dim-voxel_grid.shape[2])), (0, 0)), mode='constant')

  return voxel_grid, ids

def trans_to_id(segm_voxel_grids, img_id_list, mask_id_list):
    # Flatten the voxel grid to a 1D array
    flat_voxel_grid = segm_voxel_grids.flatten()

    # Sort the unique values in ascending order
    sorted_values = np.sort(mask_id_list)
    # print(f'sorted values: {sorted_values}')

    # Create a dictionary to map the unique values to the ids in id_list
    value_to_id = {sorted_values[i]: img_id_list[i] for i in range(len(img_id_list))}

    # Map the values in the flattened voxel grid to the ids in id_list using the value_to_id dictionary
    id_array = np.array([value_to_id[value] for value in flat_voxel_grid])

    # Reshape the id array to the same shape as the original voxel grid
    id_voxel_grid = id_array.reshape(segm_voxel_grids.shape)

    return id_voxel_grid

def get_instances_roi_list(segm_voxel_grids):
    # Initialize an empty dictionary to store the coordinates, area, and size of each object id
    object_dict = {}
    # Loop through each voxel in the segmentation voxel grid
    for i in range(segm_voxel_grids.shape[0]):
        for j in range(segm_voxel_grids.shape[1]):
            for k in range(segm_voxel_grids.shape[2]):
                # Get the object id of the current voxel
                object_id = int(segm_voxel_grids[i, j, k])
                # If the object id is not in the dictionary yet, add it with the current voxel coordinates, area, and size
                if object_id not in object_dict:
                    object_dict[object_id] = {'front_bottom_left': [i, j, k], 'back_top_right': [i, j, k], 'size': [1, 1, 1]}
                # If the object id is already in the dictionary, update the bottom right coordinates, area, and size if necessary
                else:
                    object_dict[object_id]['back_top_right'][0] = max(object_dict[object_id]['back_top_right'][0], i)
                    object_dict[object_id]['back_top_right'][1] = max(object_dict[object_id]['back_top_right'][1], j)
                    object_dict[object_id]['back_top_right'][2] = max(object_dict[object_id]['back_top_right'][2], k)
                    # Update the top left coordinates if necessary
                    object_dict[object_id]['front_bottom_left'][0] = min(object_dict[object_id]['front_bottom_left'][0], i)
                    object_dict[object_id]['front_bottom_left'][1] = min(object_dict[object_id]['front_bottom_left'][1], j)
                    object_dict[object_id]['front_bottom_left'][2] = min(object_dict[object_id]['front_bottom_left'][2], k)
    # object_list = list(object_dict.values())
    for key in object_dict.keys():
       if object_dict[key]['front_bottom_left'][1] > object_dict[key]['back_top_right'][1]:
          print(f'object {key}') 
    return object_dict

def main(unused_argv):
  segm_image_batches, depth_image_batches, file_name_batches, class_paths = load_rgb_segm_depth(os.path.join(FLAGS.data_source, FLAGS.task), FLAGS.num_per_class)

  for idx, class_path in enumerate(class_paths):
    for n in range(FLAGS.num_per_class):
      print(f'Processing {class_path}...')


      img_ids = np.unique(segm_image_batches[idx][n])
      # print(f'unique ids: {img_ids}')

      segmented_voxel_grids, mask_ids = get_segmented_voxel_grids(segm_image_batches[idx][n], depth_image_batches[idx][n])

      voxel_mask = segmented_voxel_grids[:, :, :, 0]


      voxel_mask = trans_to_id(voxel_mask, img_ids, mask_ids)

      roi_dict = get_instances_roi_list(voxel_mask)

      print("\tGet ROIs done")

      contact_point_threshold = 0.4

      mask_ids = np.unique(voxel_mask)

      contact_points_pair = get_CPN(voxel_mask, mask_ids, contact_point_threshold)

      print("\tGet CPNs done")

      merge_CPNs = merge_contact_points(contact_points_pair, mask_ids)

      edge_distances = compute_edge_distances(merge_CPNs, mask_ids)

      print("\tGet edge of CPNs done")


      save_path = os.path.join(FLAGS.data_dir, class_path)
      if not os.path.exists(save_path):
          os.makedirs(save_path)

      # Write the JSON string to a file
      with open(os.path.join(save_path, file_name_batches[idx][n] + '_ROI.json'), 'w') as f:
        json.dump(roi_dict, f, indent=2)
      with open(os.path.join(save_path, file_name_batches[idx][n] + '_CPN.json'), 'w') as f:
        json.dump(contact_points_pair, f, indent=2)
      with open(os.path.join(save_path, file_name_batches[idx][n] + '_CPN_edge.json'), "w") as f:
        json.dump(edge_distances, f, indent=2)


      # save_path = os.path.join(file_path, file_name[:file_name.find('.')] + '.npy')
      # voxel_grid = np.mean(voxel_grid, axis=-1, keepdims=True)
      # np.save(os.path.join(save_path, file_name_batches[idx][n] + '.npy'), voxel_grid)

if __name__ == '__main__':
  app.run(main)