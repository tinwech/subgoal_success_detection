import os
import torch
import pickle
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def triplet_loss(anchor, positive, negative, margin):
    """
    Calculates triplet loss for metric learning.

    Args:
    anchor (torch.Tensor): Tensor containing the anchor embeddings. Shape: (batch_size, embedding_size).
    positive (torch.Tensor): Tensor containing the positive embeddings. Shape: (batch_size, embedding_size).
    negative (torch.Tensor): Tensor containing the negative embeddings. Shape: (batch_size, embedding_size).
    margin (float): Margin to use in the triplet loss formula.

    Returns:
    loss (torch.Tensor): The triplet loss for the batch.
    """
    # Calculate the distances between the anchor and positive embeddings
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)

    # Calculate the distances between the anchor and negative embeddings
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

    # Calculate the loss
    loss = torch.mean(torch.max(pos_dist - neg_dist + margin, torch.tensor(0.0).to(anchor.device)))

    return loss


def split_triplets(embeddings, labels):
    # Split the embeddings and labels into anchor, positive, and negative examples
    anchors = []
    positives = []
    negatives = []
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        label = labels[i]

        # Find all embeddings with the same label as the anchor
        mask_pos = (labels == label)

        # Find all embeddings with a different label than the anchor
        if label % 2 == 0:  # fail
            continue
        else:  # success
            mask_neg = (labels == label - 1)

        if mask_pos.sum() <= 1 or mask_neg.sum() == 0:
            # Skip this anchor if there are no positive or negative examples
            continue

        # Choose a positive example at random from the same label group as the anchor
        positive = embeddings[mask_pos][torch.randint(0, mask_pos.sum(), (1,))]

        # Choose a negative example at random from the different label group than the anchor
        negative = embeddings[mask_neg][torch.randint(0, mask_neg.sum(), (1,))]

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    if len(anchors) == 0:
        # Return None if no triplets could be created
        return None

    # Convert the lists of tensors to a single tensor
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives


def save_img_pair(right_pos_pairs, right_neg_pairs, wrong_pos_pairs, wrong_neg_pairs, name, model_name):
    print(f'saving image cases for {name}-{model_name}...')

    def save_img(dir_path, pairs):

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for i, img_pair in enumerate(pairs):
            path1, path2 = img_pair
            with open(os.path.join(name, path1), "rb") as f:
                img1 = pickle.load(f)
            with open(os.path.join(name, path2), "rb") as f:
                img2 = pickle.load(f)
            img1 = Image.fromarray(img1, "RGB")
            img2 = Image.fromarray(img2, "RGB")
            img1.save(f"./{dir_path}/{i}.png")
            img2.save(f"./{dir_path}/{i}-anchor.png")

    save_img(f"./images/{name}-{model_name}/TP_pairs", right_pos_pairs)
    save_img(f"./images/{name}-{model_name}/TF_pairs", right_neg_pairs)
    save_img(f"./images/{name}-{model_name}/FN_pairs", wrong_pos_pairs)
    save_img(f"./images/{name}-{model_name}/FP_pairs", wrong_neg_pairs)


class MetricLearning:
    def __init__(self, model, data_loader, optimizer, margin, device, epoch=20):

        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.margin = margin
        self.device = device
        self.model.to(self.device)
        self.epoch = epoch

    def __call__(self):
        pass

    def train(self, model_name, data_type):
        self.model.train()
        for epoch in tqdm(range(1, self.epoch + 1)):
            for i, (data, labels, _, proposals) in enumerate(self.data_loader): #, total=len(self.data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                proposals = proposals.to(self.device)
                self.optimizer.zero_grad()
                embeddings = self.model(data, proposals)

                # Split the embeddings into anchor, positive, and negative examples
                try:
                    anchors, positives, negatives = split_triplets(embeddings, labels)
                except:
                    continue

                loss = triplet_loss(anchors, positives, negatives, self.margin)
                loss.backward()
                
                self.optimizer.step()

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"record/eval_pyramid/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl")
                torch.save(self.model.state_dict(), f"record/eval_arch/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl")


    def tsne_projection(self, name, model_name):
        self.model.eval()

        with torch.no_grad():
            for i, (data, labels, _, proposals) in enumerate(self.data_loader):
                data = data.to(self.device)
                proposals = proposals.to(self.device)

                embeddings = self.model(data, proposals)
                embeddings_np = embeddings.cpu().numpy()
                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                embedding_2d = tsne.fit_transform(embeddings_np)
                plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 0], \
                            [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 0], \
                                c=['#454545' for i in range(100)], label='fail scene model embedding')
                plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 1], \
                            [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 1], \
                                c=['#FF6000' for i in range(100)], label='success scene model embedding')
                plt.legend(loc='lower left')
                plt.legend(fontsize='large')
                plt.title("TSNE projection of successful and failed scene embeddings")
                plt.xticks([])
                plt.yticks([])
                # if not os.path.exists(f"./tsne/{name}"):
                #     os.mkdir("./tsne")
                if not os.path.exists(f"./tsne/{name}"):
                    os.mkdir(f"./tsne/{name}")
                if not os.path.exists(f"./tsne/{name}/{model_name}"):
                    os.mkdir(f"./tsne/{name}/{model_name}")
                plt.savefig(f"./tsne/{name}/{model_name}/{i+1}.png")
                plt.clf()


    def evaluate(self, name, model_name, data_type):
        line_colors = ["red", "orange", "yellow", "green", "blue", 'black']
        p_dist = []
        n_dist = []
        
        

        # Set the model to evaluation mode
        self.model.eval()
        f1 = [[] for i in range(10, self.epoch + 1, 10)]
        for epoch in range(10, self.epoch + 1, 10):
            # img_path_pairs = []
            # wrong_pos_pairs = []
            # wrong_neg_pairs = []
            # right_pos_pairs = []
            # right_neg_pairs = []

            macro_f1_list = []
            micro_f1_list = []
            acc_list = []
            TP_list = []
            TN_list = []
            FP_list = []
            FN_list = []
            for iter in range(0, 5):
                threshold = iter * 0.1 + 0.4
               
                self.model.load_state_dict(torch.load(f"record/{name}/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl", map_location=torch.device('cuda')))

                y_true = []
                y_prob = []
                with torch.no_grad():
                    for i, (data, labels, img_paths, proposals) in enumerate(self.data_loader):
                        data = data.to(self.device)
                        proposals = proposals.to(self.device)

                        embeddings = self.model(data, proposals)
                        for j, embedding in enumerate(embeddings):
                            
                            if j < 100:
                                anchor_idx = list(range(100, 200))
                                buf = np.array([])
                                # indices = np.array([])
                                for idx in anchor_idx:
                                    n_distance = torch.sqrt(torch.sum((embedding.cuda() - embeddings[idx]) ** 2))
                                    n_probability = (2 - n_distance) / 2
                                    buf = np.append(buf, n_probability.to('cpu'))
                                    # indices = np.append(indices, idx)
                                n = np.median(buf)
                                # n_idx = int(indices[np.where(buf < n)][-1])
                                y_prob.append(n)
                                y_true.append(0)
                                n = 2 - 2 * n
                                n_dist.append(n)
                                # if threshold == 0.4 and epoch == 20:
                                #     img_path_pairs.append((img_paths[j], img_paths[n_idx]))
                            else:
                                anchor_idx = list(range(100, 200))
                                anchor_idx.pop(j - 100)
                                buf = np.array([])
                                # indices = np.array([])
                                for idx in anchor_idx:
                                    p_distance = torch.sqrt(torch.sum((embedding.cuda() - embeddings[idx]) ** 2))
                                    p_probability = (2 - p_distance) / 2
                                    buf = np.append(buf, p_probability.to('cpu'))
                                    # indices = np.append(indices, idx)
                                p = np.median(buf)
                                # p_idx = int(indices[np.where(buf == p)][0])
                                # print(f'p_idx: {p_idx}')

                                y_prob.append(p)
                                y_true.append(1)
                                p = 2 - 2 * p
                                p_dist.append(p)
                            # pos refers to success scene, neg refers to failed scene
                                # if threshold == 0.4 and epoch == 20:
                                #     img_path_pairs.append((img_paths[j], img_paths[p_idx]))

                # make predictions
                y_pred = np.where(np.array(y_prob) > threshold, 1, 0)
                
                cnt = [0, 0, 0, 0]
                # record right and wrong predictions' image paths
                for idx in range(len(y_true)):
                    if y_true[idx] == y_pred[idx]:  # right prediction
                        if (idx % 200) >= 100:  # pos scene
                            # if threshold == 0.4 and epoch == 20: 
                            #     right_pos_pairs.append(img_path_pairs[idx])
                            cnt[0] += 1
                        else:  # neg scene
                            # if threshold == 0.4 and epoch == 20: 
                            #     right_neg_pairs.append(img_path_pairs[idx])
                            cnt[1] += 1
                    else:  # wrong prediction
                        if (idx % 200) >= 100:  # pos scene
                            # if threshold == 0.4 and epoch == 20: 
                            #     wrong_pos_pairs.append(img_path_pairs[idx])
                            cnt[2] += 1
                        else:  # neg scene
                            # if threshold == 0.4 and epoch == 20: 
                            #     wrong_neg_pairs.append(img_path_pairs[idx])
                            cnt[3] += 1

                macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
                micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
                acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                acc_list.append(acc)
                macro_f1_list.append(macro_f1)
                micro_f1_list.append(micro_f1)
                TP_list.append(cnt[0])
                TN_list.append(cnt[1])
                FP_list.append(cnt[3])
                FN_list.append(cnt[2])

                
                f1[int(epoch/10 - 1)].append(micro_f1) # for plt.plot

            
            # if epoch == 20: 
            #     save_img_pair(right_pos_pairs, right_neg_pairs, wrong_pos_pairs, wrong_neg_pairs, name, model_name)

            
            # plt.hist(p_dist, bins=200, color='r', alpha=0.5, label='anchor and positive embeddings distance')
            # plt.hist(n_dist, bins=200, color='b', alpha=0.5, label='anchor and negative embeddings distance')
            # plt.hist(np.log10(p_dist), bins=200, color='r', alpha=0.5)
            # plt.hist(np.log10(n_dist), bins=200, color='b', alpha=0.5)
            # plt.xlabel('Distance')
            # plt.ylabel('Count (log)')
            # plt.title('Two scenes\' embedding distances distribution')
            # # plt.legend()
            # plt.savefig(f'record/{name}/{model_name}/{data_type}/margin_{self.margin}_dist_hist.png')
            # plt.clf()

            output = {'acc': acc_list, 'macro_f1': macro_f1_list, 'micro_f1': micro_f1_list, 'TP': TP_list, 'TN': TN_list, 'FP': FP_list, 'FN':FN_list}
            df = pd.DataFrame.from_dict(output)
            df.index = [f'threshold {i * 0.1 + 0.4}' for i in range(0, 5)]
            df.to_csv(f'record/{name}/{model_name}/{data_type}/margin_{self.margin}.csv', index=True, header=True)

            # if epoch == 20:
            #     self.tsne_projection(name, model_name)
        f1 = np.array(f1)
        # print(f1)
        # print(f1[:, iter])
        # print(list(range(10, self.epoch + 1, 10)))
        for iter in range(0, 5):
            plt.plot(list(range(10, self.epoch + 1, 10)), f1[:, iter], color=line_colors[iter], label=f"thres. = {iter * 0.1 + 0.4}")
        plt.ylim(0, 1)
        plt.legend(loc="best")
        plt.xlabel("Epoch")
        plt.ylabel("Micro F1 score")
        plt.title("F1 Curve")
        # plt.savefig(f"record/{name}/{model_name}/{data_type}/epoch-{self.margin}-{epoch}_evaluate_f1.png")
        plt.clf()

class GraphMetricLearning:
    def __init__(self, model, data_loader, optimizer, margin, device, epoch=20):

        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.margin = margin
        self.device = device
        self.model.to(self.device)
        self.epoch = epoch

    def __call__(self):
        pass

    def train(self, model_name, data_type):
        self.model.train()
        for epoch in tqdm(range(1, self.epoch + 1)):
            for i, (data, labels, _, proposals, node_features, edge_indices, edge_attrs) in enumerate(self.data_loader): #, total=len(self.data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                proposals = proposals.to(self.device)
                node_features = node_features.to(self.device)
                edge_indices = edge_indices.to(self.device)
                edge_attrs = edge_attrs.to(self.device)
                self.optimizer.zero_grad()
                embeddings = self.model(data, proposals, node_features, edge_indices, edge_attrs)

                # Split the embeddings into anchor, positive, and negative examples
                try:
                    anchors, positives, negatives = split_triplets(embeddings, labels)
                except:
                    continue

                loss = triplet_loss(anchors, positives, negatives, self.margin)
                loss.backward()
                
                self.optimizer.step()

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"record/eval_pyramid/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl")
                torch.save(self.model.state_dict(), f"record/eval_arch/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl")


    def tsne_projection(self, name, model_name):
        self.model.eval()

        with torch.no_grad():
            for i, (data, labels, _, proposals, node_features, edge_indices, edge_attrs) in enumerate(self.data_loader):
                data = data.to(self.device)
                proposals = proposals.to(self.device)
                node_features = node_features.to(self.device)
                edge_indices = edge_indices.to(self.device)
                edge_attrs = edge_attrs.to(self.device)
                embeddings = self.model(data, proposals, node_features, edge_indices, edge_attrs)
                embeddings_np = embeddings.cpu().numpy()
                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                embedding_2d = tsne.fit_transform(embeddings_np)
                # plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 0], \
                #             [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 0], \
                #                 c=['#454545' for i in range(100)], label='fail scene model embedding')
                # plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 1], \
                #             [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 1], \
                #                 c=['#FF6000' for i in range(100)], label='success scene model embedding')
                plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 0], \
                            [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 0], \
                                c=['#454545' for i in range(100)])
                plt.scatter([embedding_2d[i, 0] for i in range(200) if labels[i]%2 == 1], \
                            [embedding_2d[i, 1] for i in range(200) if labels[i]%2 == 1], \
                                c=['#FF6000' for i in range(100)])
                # plt.legend(loc='lower left')
                plt.title("TSNE projection of successful and failed scene embeddings")
                plt.xticks([])
                plt.yticks([])
                # if not os.path.exists(f"./tsne/{name}"):
                #     os.mkdir("./tsne")
                if not os.path.exists(f"./tsne/{name}"):
                    os.mkdir(f"./tsne/{name}")
                if not os.path.exists(f"./tsne/{name}/{model_name}"):
                    os.mkdir(f"./tsne/{name}/{model_name}")
                plt.savefig(f"./tsne/{name}/{model_name}/{i+1}.png")
                plt.clf()


    def evaluate(self, name, model_name, data_type):
        line_colors = ["red", "orange", "yellow", "green", "blue", 'black']
        p_dist = []
        n_dist = []
        
        

        # Set the model to evaluation mode
        self.model.eval()
        f1 = [[] for i in range(10, self.epoch + 1, 10)] 
        for epoch in range(10, self.epoch + 1, 10):

            pos_img_path_pairs = []
            neg_img_path_pairs = []
            wrong_pos_pairs = []
            wrong_neg_pairs = []
            right_pos_pairs = []
            right_neg_pairs = []

            macro_f1_list = []
            micro_f1_list = []
            acc_list = []
            TP_list = []
            TN_list = []
            FP_list = []
            FN_list = []
            for iter in range(0, 5):
                threshold = iter * 0.1 + 0.4
               
                self.model.load_state_dict(torch.load(f"record/{name}/{model_name}/{data_type}/epoch-{self.margin}-{epoch}.pkl", map_location=torch.device('cuda')))

                y_true = []
                y_prob = []
                with torch.no_grad():
                    for i, (data, labels, img_paths, proposals, node_features, edge_indices, edge_attrs) in enumerate(self.data_loader):
                        data = data.to(self.device)
                        proposals = proposals.to(self.device)
                        node_features = node_features.to(self.device)
                        edge_indices = edge_indices.to(self.device)
                        edge_attrs = edge_attrs.to(self.device)
                        embeddings = self.model(data, proposals, node_features, edge_indices, edge_attrs)
                        for j, embedding in enumerate(embeddings):
                            if j < 100:
                                anchor_idx = list(range(100, 200))
                                buf = np.array([])
                                for idx in anchor_idx:
                                    n_distance = torch.sqrt(torch.sum((embedding.cuda() - embeddings[idx]) ** 2))
                                    n_probability = (2 - n_distance) / 2
                                    buf = np.append(buf, n_probability.to('cpu'))
                                n = np.median(buf)
                                y_prob.append(n)
                                y_true.append(0)
                                n = 2 - 2 * n
                                n_dist.append(n)
                            else:
                                anchor_idx = list(range(100, 200))
                                anchor_idx.pop(j - 100)
                                buf = np.array([])
                                for idx in anchor_idx:
                                    p_distance = torch.sqrt(torch.sum((embedding.cuda() - embeddings[idx]) ** 2))
                                    p_probability = (2 - p_distance) / 2
                                    buf = np.append(buf, p_probability.to('cpu'))
                                p = np.median(buf)
                                y_prob.append(p)
                                y_true.append(1)
                                p = 2 - 2 * p
                                p_dist.append(p)

                            # pos refers to success scene, neg refers to failed scene
                            # if threshold == 0.8 and epoch == 20: 
                                # pos_img_path_pairs.append((img_paths[j], img_paths[pos_idx]))
                                # neg_img_path_pairs.append((img_paths[j], img_paths[neg_idx]))

                # make predictions
                y_pred = np.where(np.array(y_prob) > threshold, 1, 0)
                
                cnt = [0, 0, 0, 0]
                # record right and wrong predictions' image paths
                for idx in range(len(y_true)):
                    if y_true[idx] == y_pred[idx]:  # right prediction
                        if idx % 2 == 0:  # pos scene
                            # if threshold == 0.8 and epoch == 20: 
                            #     right_pos_pairs.append(pos_img_path_pairs[idx])
                            cnt[0] += 1
                        else:  # neg scene
                            # if threshold == 0.8 and epoch == 20: 
                            #     right_neg_pairs.append(neg_img_path_pairs[idx])
                            cnt[1] += 1
                    else:  # wrong prediction
                        if idx % 2 == 0:  # pos scene
                            # if threshold == 0.8 and epoch == 20: 
                            #     wrong_pos_pairs.append(pos_img_path_pairs[idx])
                            cnt[2] += 1
                        else:  # neg scene
                            # if threshold == 0.8 and epoch == 20: 
                            #     wrong_neg_pairs.append(neg_img_path_pairs[idx])
                            cnt[3] += 1

                macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
                micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
                acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                acc_list.append(acc)
                macro_f1_list.append(macro_f1)
                micro_f1_list.append(micro_f1)
                TP_list.append(cnt[0])
                TN_list.append(cnt[1])
                FP_list.append(cnt[3])
                FN_list.append(cnt[2])

                
                f1[int(epoch/10 - 1)].append(micro_f1) # for plt.plot

            
            # if epoch == 20:
            #     save_img_pair(right_pos_pairs, right_neg_pairs, wrong_pos_pairs, wrong_neg_pairs, name)

            
            # plt.hist(p_dist, bins=200, color='r', alpha=0.5, label='anchor and positive embeddings distance')
            # plt.hist(n_dist, bins=200, color='b', alpha=0.5, label='anchor and negative embeddings distance')
            # plt.hist(p_dist, bins=200, color='r', alpha=0.5)
            # plt.hist(n_dist, bins=200, color='b', alpha=0.5)
            # plt.xlabel('Distance')
            # plt.ylabel('Count (log)')
            # plt.title('Two scenes\' embedding distances distribution')
            # # plt.legend()
            # plt.savefig(f'record/{name}/{model_name}/{data_type}/margin_{self.margin}_dist_hist.png')
            # plt.clf()

            output = {'acc': acc_list, 'macro_f1': macro_f1_list, 'micro_f1': micro_f1_list, 'TP': TP_list, 'TN': TN_list, 'FP': FP_list, 'FN':FN_list}
            df = pd.DataFrame.from_dict(output)
            df.index = [f'threshold {i * 0.1 + 0.4}' for i in range(0, 5)]
            # df.to_csv(f'record/{name}/{model_name}/{data_type}/margin_{self.margin}.csv', index=True, header=True)

            if epoch == 20:
                self.tsne_projection(name, model_name)
        f1 = np.array(f1)
        # print(f1)
        for iter in range(0, 5):
            plt.plot(list(range(10, self.epoch + 1, 10)), f1[:, iter], color=line_colors[iter], label=f"thres. = {iter * 0.1 + 0.4}")
        plt.ylim(0, 1)
        plt.legend(loc="best")
        plt.xlabel("Epoch")
        plt.ylabel("Micro F1 score")
        plt.title("F1 Curve")
        # plt.savefig(f"record/{name}/{model_name}/{data_type}/epoch-{self.margin}-{epoch}_evaluate_f1.png")
        plt.clf()

