import sys
import torch
from torch_geometric import loader
from torch.utils.data import DataLoader
from baseline_model import BaselineModel
from dataset import BuildDataset, BuildGraphDataset, BuildHistGraphDataset
from metric_learning import MetricLearning, GraphMetricLearning
import argparse
from scene_graph_model import SceneGraphModel
from baseline_roi_model import BaselineROI
from CPN_model import CPNModel
from histogram_model import HistogramModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="baseline")
    parser.add_argument("-d", "--dataset", default='larger')
    parser.add_argument("-g", "--attach_gripper", type=bool, default=False)
    parser.add_argument("-i", "--attach_instance_occupancy", type=bool, default=False)

    args = parser.parse_args()

    print("model:", args.model)
    print("dataset:", args.dataset)
    
    if args.model == "baseline_roi":
        model = BaselineROI()
    elif args.model == "scene_graph":
        model = SceneGraphModel()
    elif args.model == "cpn":
        model = CPNModel()
    elif args.model == "histogram":
        model = HistogramModel()
    else:
        model = BaselineModel()
    
    print(model.__class__.__name__)

    if args.dataset == 'larger':
        train_dataset_name = '(larger noise35*30*50)stack-block-pyramid'
        eval_dataset_name1 = '(larger noise35*30*50)stack-block-pyramid'
        eval_dataset_name2 = '(larger noise35*30*50)stack-block-door'

    elif args.dataset == 'smaller':
        train_dataset_name = '(smaller noise35*30*50)stack-block-pyramid'
        eval_dataset_name1 = '(smaller noise35*30*50)stack-block-pyramid'
        eval_dataset_name2 = '(smaller noise35*30*50)stack-block-door'

    if args.model == "cpn":
        dataset = BuildGraphDataset()
        train_data = dataset(f"./rgb_voxel_data/train/{train_dataset_name}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        train_loader = loader.DataLoader(train_data, batch_size=200, shuffle=False, drop_last=True)
        eval_data_1 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name1}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_1 = loader.DataLoader(eval_data_1, batch_size=200, shuffle=False, drop_last=True)
        eval_data_2 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name2}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_2 = loader.DataLoader(eval_data_2, batch_size=200, shuffle=False, drop_last=True)

    elif args.model == 'histogram':
        dataset = BuildHistGraphDataset()
        train_data = dataset(f"./rgb_voxel_data/train/{train_dataset_name}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        train_loader = DataLoader(train_data, batch_size=200, shuffle=False, drop_last=True)
        eval_data_1 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name1}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_1 = DataLoader(eval_data_1, batch_size=200, shuffle=False, drop_last=True)
        eval_data_2 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name2}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_2 = DataLoader(eval_data_2, batch_size=200, shuffle=False, drop_last=True)

    else:
        dataset = BuildDataset()
        train_data = dataset(f"./rgb_voxel_data/train/{train_dataset_name}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        train_loader = DataLoader(train_data, batch_size=200, shuffle=False, drop_last=True)
        eval_data_1 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name1}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_1 = DataLoader(eval_data_1, batch_size=200, shuffle=False, drop_last=True)
        eval_data_2 = dataset(f"./rgb_voxel_data/test/{eval_dataset_name2}", args.dataset, args.attach_gripper, args.attach_instance_occupancy)
        eval_loader_2 = DataLoader(eval_data_2, batch_size=200, shuffle=False, drop_last=True)

    if args.model == 'cpn':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    elif args.model == 'histogram':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"using device: {device}")

    
    for margin in range(9, 10):
    # for margin in range(3, 10):
        print(f"margin: {margin}")
        if args.model == "cpn" or args.model == 'histogram':
            learn = GraphMetricLearning(model, train_loader, optimizer, margin, device)
        else:
            learn = MetricLearning(model, train_loader, optimizer, margin, device)

        # learn.train(model_name=args.model, data_type=args.dataset)

        learn.data_loader = eval_loader_1
        learn.evaluate(name="eval_pyramid", model_name=args.model, data_type=args.dataset)
        
        learn.data_loader = eval_loader_2
        learn.evaluate(name="eval_arch", model_name=args.model, data_type=args.dataset)
