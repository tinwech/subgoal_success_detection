from pathlib import Path
import os
import numpy as np
import torch
import json
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset


class MyDataset(Dataset):
    def __init__(self, data, label, file_img_path, roi_dict):
        self.data = data
        self.label = label
        self.img_path = file_img_path
        self.roi_dict = roi_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Assuming each data point is a tuple of (image, label)
        voxels, label, img_path, roi_dict = self.data[idx], self.label[idx], self.img_path[idx], self.roi_dict[idx]
        
        # Normalize voxel data to have zero mean and unit variance
        # print(np.unique(voxels))
        # print(voxels.shape)
        voxels = (voxels - np.mean(voxels)) / np.std(voxels)

        # Convert voxel data to PyTorch tensor
        voxels = torch.from_numpy(voxels).float()
        
        # Transpose voxel data from (depth, height, width, channels) to (channels, depth, height, width)
        voxels = voxels.permute(3, 0, 1, 2)


        # Convert label to tensorct
        label = torch.tensor(label).long()

        proposals = torch.tensor(roi_dict).float()


        return (voxels, label, img_path, proposals)

class MyGraphDataset(Dataset):
    def __init__(self, data, label, file_img_path, roi_dict, node_feature_list, edge_index_list, edge_attr_list):
        self.data = data
        self.label = label
        self.img_path = file_img_path
        self.roi_dict = roi_dict
        self.node_feature_list = node_feature_list
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Assuming each data point is a tuple of (image, label)
        voxels, label, img_path, roi_dict, node_feature, edge_index, edge_attr = self.data[idx], self.label[idx], self.img_path[idx], self.roi_dict[idx], self.node_feature_list[idx], self.edge_index_list[idx], self.edge_attr_list[idx]
        
        # Normalize voxel data to have zero mean and unit variance
        # print(np.unique(voxels))
        # print(voxels.shape)
        
        voxels = (voxels - np.mean(voxels)) / np.std(voxels)

        # Convert voxel data to PyTorch tensor
        voxels = torch.from_numpy(voxels).float()
        node_feature = torch.from_numpy(node_feature).float()
        
        # Transpose voxel data from (depth, height, width, channels) to (channels, depth, height, width)
        voxels = voxels.permute(3, 0, 1, 2)

        # node_feature = node_feature.permute(3 ,0,1,2)

        # Convert label to tensor
        label = torch.tensor(label).long()

        proposals = torch.tensor(roi_dict).float()

        # edge_indices = torch.tensor(edge_indices)
        # edge_attrs = torch.tensor(edge_attrs)

        return (voxels, label, img_path, proposals, node_feature, edge_index, edge_attr)

class BuildGraphDataset:
    def __init__(self):
        pass

    def parse_roi_dict(self, roi_dict):
        roi_list = []
        for i in range(4, 12):
        # for i in range(4, 5):
            id = str(i)
            if roi_dict.get(id) is None:
                # padding_dict = {"front_top_left": [0,0,0], "back_bottom_right": [0,0,0], "size": [0,0,0]}
                roi_list.append([0, 0, 0, 0, 0, 0])
            else:
                list1 = roi_dict[id]["front_bottom_left"]
                list2 = roi_dict[id]["back_top_right"]
                roi_list.append(list1 + list2)
        
        return roi_list
    
    def create_CPN_node_features(self, CPNs):
        num_node = 8
        num_point = 8
        node_features = np.zeros((num_node, num_point, 3))
        for i in CPNs.keys():
            num_contact_point = 0
            for j in CPNs[i].keys():
                CPN = CPNs[i][j]
                if len(CPN) == 0:
                    continue
                for pos in CPN:
                    if num_contact_point == 8:
                        break
                    node_features[int(i) - 4, int(num_contact_point), :] = np.array(pos)
                    num_contact_point += 1
            if num_contact_point == 8:
                break

        return node_features
        
    
    def create_CPN_voxels(self, data, CPNs):
        """
        Attach a channel to the data, which indicates the contact point

        Args:
            data: (depth, height, width, 1)
            CPN: a dict with the contact point
                key: id
                value: (x, y, z)

        Returns:
            new_data: (depth, height, width, 1)
        """
        contact_point_voxels = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        for i in CPNs.keys():
            for j in CPNs[i].keys():
                CPN = CPNs[i][j]
                if len(CPN) == 0:
                    continue
                for x, y, z in CPN:
                    x, y, z = int(x), int(y), int(z)
                    contact_point_voxels[x, y, z, 0] = 1

        return contact_point_voxels
    
    def parse_edges(self, edge_dict):
        f = []
        t = []
        edge_attrs = []
        for id1 in edge_dict.keys():
            for id2 in edge_dict[id1].keys():
                f.append(int(id1)-4)
                t.append(int(id2)-4)
                edge_attrs.append(edge_dict[id1][id2])
        if len(edge_attrs) == 0:
            f.append(0)
            t.append(0)
            edge_attrs.append([0, 0, 0])

        edge_attrs = torch.tensor(edge_attrs)
        f = torch.tensor(f, dtype=torch.long)
        t = torch.tensor(t, dtype=torch.long)
        edge_index = torch.concat([f, t], dim=0)
        edge_index = torch.reshape(edge_index, (2,-1))
        padded_edge_index = torch.zeros(2, 8*8)
        padded_edge_index[:, :edge_index.shape[1]] = edge_index

        # print(edge_attrs)
        # print(f'edge attr shape: {edge_attrs.shape}')

        padded_edge_attr = torch.zeros(8*8, 3)
        # print(f'padded edge attr shape: {padded_edge_attr.shape}')
        padded_edge_attr[:edge_attrs.shape[0],:] = edge_attrs

        # print(edge_index)
        return padded_edge_index, padded_edge_attr
                
    def attach_gripper_roi(self, data, roi_dict):
        """
        Attach a channel to the data, which indicates the gripper's ROI

        Args:
            data: (depth, height, width, 1)
            roi_dict: (number of instances, 6)

        Returns:
            new_data: (depth, height, width, 2)
        """
        new_channel = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        new_data = np.concatenate([data, new_channel], axis=-1)
        lx, ly, lz, rx, ry, rz = roi_dict[0]
        new_data[lx:rx, ly:ry, lz:rz, -1] = 1

        return new_data
    
    def attach_intance_occupancy(self, data, roi_dict):
        """
        Attach a channel to the data, which indicates occupancy
        0/1: represents that the voxel is occupied

        Args:
            data: (depth, height, width, 1)
            roi_dict: (number of instances, 6)

        Returns:
            new_data: (depth, height, width, 2)
        """
        new_channel = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        new_data = np.concatenate([data, new_channel], axis=-1)
        for i in range(len(roi_dict)):
            lx, ly, lz, rx, ry, rz = roi_dict[i]
            new_data[lx:rx, ly:ry, lz:rz, -1] = 1

        return new_data


    def __call__(self, task_path: str, data_type, flg_attach_gripper, flg_instance_occupancy):
        folder_path = Path(task_path)
        label = -1
        path = "."
        data_list = []
        labels = []
        file_img_paths = []
        roi_dicts = []
        node_feature_list = []
        edges_indices_list = []
        edges_attrs_list = []
        file_paths = sorted(list(folder_path.glob("**/*.npy")))

        Data_list = []

        for _, file_path in enumerate(file_paths):

            parent_path = os.path.dirname(file_path)
            if path != parent_path:
                label += 1
                path = parent_path

            data = np.load(file_path)

            file_path = str(file_path)

            
            with open(file_path[:file_path.find('.')] + '_ROI.json', 'r') as f:
                # print(file_path[:file_path.find('.')] + '.json')
                # print(json.load(f))
                # print(type(json.load(f)))
                roi_dict = self.parse_roi_dict(json.load(f))


            with open(file_path[:file_path.find('.')] + '_CPN.json', 'r') as f:
                CPN = json.load(f)
            with open(file_path[:file_path.find('.')] + '_CPN_edge.json', 'r') as f:
                edge_indices, edge_attrs = self.parse_edges(json.load(f))

            if "train" in file_path:
                start_index = file_path.index("train/") + len("train/")
                parent_index = parent_path.index("train/") + len("train/")
            elif "test" in file_path:
                start_index = file_path.index("test/") + len("test/")
                parent_index = parent_path.index("test/") + len("test/")

            if flg_attach_gripper:
                data = self.attach_gripper_roi(data, roi_dict)
            if flg_instance_occupancy:
                data = self.attach_intance_occupancy(data, roi_dict)

            sub_path = file_path[start_index:]
            file_name = os.path.basename(sub_path)
            parent_path = parent_path[parent_index:]

            save_path = os.path.join("./rgbd", parent_path, "color", f"{file_name[:6]}.pkl")
            file_img_paths.append(save_path)
            data_list.append(data)
            labels.append(label)
            roi_dicts.append(roi_dict)

            # cpn = self.create_CPN_voxels(data, CPN)
            node_feature = self.create_CPN_node_features(CPN)

            node_feature_list.append(node_feature)
            edges_indices_list.append(edge_indices)
            edges_attrs_list.append(edge_attrs)
        
        labels = np.array(labels)
        data_list = np.array(data_list)
        node_feature_list = np.array(node_feature_list)
        return MyGraphDataset(data_list, labels, file_img_paths, roi_dicts, node_feature_list, edges_indices_list, edges_attrs_list)

class BuildHistGraphDataset:
    def __init__(self):
        pass

    def parse_roi_dict(self, roi_dict):
        roi_list = []
        for i in range(4, 12):
        # for i in range(4, 5):
            id = str(i)
            if roi_dict.get(id) is None:
                # padding_dict = {"front_top_left": [0,0,0], "back_bottom_right": [0,0,0], "size": [0,0,0]}
                roi_list.append([0, 0, 0, 0, 0, 0])
            else:
                list1 = roi_dict[id]["front_bottom_left"]
                list2 = roi_dict[id]["back_top_right"]
                roi_list.append(list1 + list2)
        
        return roi_list

    def __call__(self, task_path: str, data_type, flg_attach_gripper, flg_instance_occupancy):
        folder_path = Path(task_path)
        label = -1
        path = "."
        data_list = []
        labels = []
        file_img_paths = []
        roi_dicts = []
        node_feature_list = []
        edges_indices_list = []
        edges_attrs_list = []
        file_paths = sorted(list(folder_path.glob("**/*.npy")))

        for _, file_path in enumerate(file_paths):

            parent_path = os.path.dirname(file_path)
            if path != parent_path:
                label += 1
                path = parent_path

            file_path = str(file_path)

            
            with open(file_path[:file_path.find('.')] + '_ROI.json', 'r') as f:
                # print(file_path[:file_path.find('.')] + '.json')
                # print(json.load(f))
                # print(type(json.load(f)))
                data = json.load(f)
                roi_dict = self.parse_roi_dict(data)
                node_feature = []
                for i in range(4, 12):
                    node_feature.append([17, 15, 25])
                keys_int = []
                keys = list(data.keys())
                for key in keys:
                    keys_int.append(int(key))
                keys_int = sorted(keys_int)
                for key in keys_int:
                    if key < 4 or key > 11: continue
                    idx = key - 4
                    l = data[str(key)]['front_bottom_left']
                    r = data[str(key)]['back_top_right']
                    l = np.array(l)
                    r = np.array(r)
                    mid = l + (r - l) / 2
                    node_feature[idx] = mid.tolist()
                node_feature_list.append(list(node_feature))

            with open(file_path[:file_path.find('.')] + '_hist.json', 'r') as f:
                hist = json.load(f)
                keys = list(hist.keys())
                edges_indices = [[], []]
                edges_attrs = []
                cnt = 0
                for key in keys:
                    src = int(key.split(',')[0])
                    dest = int(key.split(',')[1])
                    if src < 4 or src > 11: continue
                    if dest < 4 or dest > 11: continue
                    cnt += 1
                    src -= 4
                    dest -= 4
                    edges_indices[0].append(src)
                    edges_indices[1].append(dest)
                    attrs = [hist[key]['x'], hist[key]['y'], hist[key]['z'], hist[key]['xy'], hist[key]['yz'], hist[key]['xz'], hist[key]['xyz']]
                    edges_attrs.append(list(attrs))
                for i in range(cnt, 80):
                    edges_indices = [[], []]
                    edges_attrs = []
                    edges_indices[0].append(0)
                    edges_indices[1].append(0)
                    attrs = [list(range(100)), list(range(100)), list(range(100)), list(range(100)), list(range(100)),list(range(100)), list(range(100))]
                    edges_attrs.append(list(attrs))
                edges_indices_list.append(torch.tensor(list(edges_indices)))
                edges_attrs_list.append(torch.tensor(list(edges_attrs)))


            if "train" in file_path:
                start_index = file_path.index("train/") + len("train/")
                parent_index = parent_path.index("train/") + len("train/")
            elif "test" in file_path:
                start_index = file_path.index("test/") + len("test/")
                parent_index = parent_path.index("test/") + len("test/")

            sub_path = file_path[start_index:]
            file_name = os.path.basename(sub_path)
            parent_path = parent_path[parent_index:]

            save_path = os.path.join("./rgbd", parent_path, "color", f"{file_name[:6]}.pkl")
            file_img_paths.append(save_path)

            data = np.load(file_path)
            data_list.append(data)
            labels.append(label)
            roi_dicts.append(roi_dict)

        labels = np.array(labels)
        data_list = np.array(data_list)
        node_feature_list = np.array(node_feature_list)

        return MyGraphDataset(data_list, labels, file_img_paths, roi_dicts, node_feature_list, edges_indices_list, edges_attrs_list)



class BuildDataset:
    def __init__(self):
        pass

    def parse_roi_dict(self, roi_dict):
        roi_list = []
        for i in range(4, 12):
        # for i in range(4, 5):
            id = str(i)
            if roi_dict.get(id) is None:
                # padding_dict = {"front_top_left": [0,0,0], "back_bottom_right": [0,0,0], "size": [0,0,0]}
                roi_list.append([0, 0, 0, 0, 0, 0])
            else:
                list1 = roi_dict[id]["front_bottom_left"]
                list2 = roi_dict[id]["back_top_right"]
                roi_list.append(list1 + list2)
        
        return roi_list
    
    def create_CPN_voxels(self, data, CPNs):
        """
        Attach a channel to the data, which indicates the contact point

        Args:
            data: (depth, height, width, 1)
            CPN: a dict with the contact point
                key: id
                value: (x, y, z)

        Returns:
            new_data: (depth, height, width, 1)
        """
        contact_point_voxels = torch.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        for i in CPNs.keys():
            for j in CPNs[i].keys():
                CPN = CPNs[i][j]
                if len(CPN) == 0:
                    continue
                for x, y, z in CPN:
                    x, y, z = int(x), int(y), int(z)
                    contact_point_voxels[x, y, z, 0] = 1

        return contact_point_voxels
    
    def parse_edges(self, edge_dict):
        f = []
        t = []
        edge_attrs = []
        for id1 in edge_dict.keys():
            for id2 in edge_dict[id1].keys():
                f.append(id1)
                t.append(id2)
                edge_attrs.append(edge_dict[id1][id2])
        # edge_index = [[row[i] for row in edge_index] for i in range(len(edge_index[0]))]
        edge_index = [f, t]
        return edge_index, edge_attrs
                
    
    def attach_gripper_roi(self, data, roi_dict):
        """
        Attach a channel to the data, which indicates the gripper's ROI

        Args:
            data: (depth, height, width, 1)
            roi_dict: (number of instances, 6)

        Returns:
            new_data: (depth, height, width, 2)
        """
        new_channel = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        new_data = np.concatenate([data, new_channel], axis=-1)
        lx, ly, lz, rx, ry, rz = roi_dict[0]
        new_data[lx:rx, ly:ry, lz:rz, -1] = 1

        return new_data
    
    def attach_intance_occupancy(self, data, roi_dict):
        """
        Attach a channel to the data, which indicates occupancy
        0/1: represents that the voxel is occupied

        Args:
            data: (depth, height, width, 1)
            roi_dict: (number of instances, 6)

        Returns:
            new_data: (depth, height, width, 2)
        """
        new_channel = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
        new_data = np.concatenate([data, new_channel], axis=-1)
        for i in range(len(roi_dict)):
            lx, ly, lz, rx, ry, rz = roi_dict[i]
            new_data[lx:rx, ly:ry, lz:rz, -1] = 1

        return new_data


    def __call__(self, task_path: str, data_type, flg_attach_gripper, flg_instance_occupancy):
        folder_path = Path(task_path)
        label = -1
        path = "."
        data_list = []
        labels = []
        file_img_paths = []
        roi_dicts = []
        CPN_list = []
        edges_indices_list = []
        edges_attrs_list = []
        file_paths = sorted(list(folder_path.glob("**/*.npy")))


        for _, file_path in enumerate(file_paths):

            parent_path = os.path.dirname(file_path)
            if path != parent_path:
                label += 1
                path = parent_path

            data = np.load(file_path)

            file_path = str(file_path)

            
            with open(file_path[:file_path.find('.')] + '_ROI.json', 'r') as f:
                # print(file_path[:file_path.find('.')] + '.json')
                # print(json.load(f))
                # print(type(json.load(f)))
                roi_dict = self.parse_roi_dict(json.load(f))


            with open(file_path[:file_path.find('.')] + '_CPN.json', 'r') as f:
                CPN = json.load(f)
            with open(file_path[:file_path.find('.')] + '_CPN_edge.json', 'r') as f:
                edge_indices, edge_attrs = self.parse_edges(json.load(f))

            if "train" in file_path:
                start_index = file_path.index("train/") + len("train/")
                parent_index = parent_path.index("train/") + len("train/")
            elif "test" in file_path:
                start_index = file_path.index("test/") + len("test/")
                parent_index = parent_path.index("test/") + len("test/")

            if flg_attach_gripper:
                data = self.attach_gripper_roi(data, roi_dict)
            if flg_instance_occupancy:
                data = self.attach_intance_occupancy(data, roi_dict)

            sub_path = file_path[start_index:]
            file_name = os.path.basename(sub_path)
            parent_path = parent_path[parent_index:]
            # print(parent_path)

            save_path = os.path.join("/media/d435/Data1/subgoal_detection_dataset/testing_rgbd", parent_path, "color", f"{file_name[:6]}.pkl")
            # print(save_path)
            file_img_paths.append(save_path)
            data_list.append(data)
            labels.append(label)

            for i in range(8):
            # for i in range(1):
                if roi_dict[i][1] > roi_dict[i][4]:
                    print(f'{i}  {file_path}')
            roi_dicts.append(roi_dict)

        labels = np.array(labels)
        data_list = np.array(data_list)
        return MyDataset(data_list, labels, file_img_paths, roi_dicts)
