import collections
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import os.path
import numpy as np
import collections


class Constant:
    data_path = './data/usps_processed/'
    dataset_name = "usps"


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def retrieve_info(cluster_labels,y_train):
    reference_labels = {}
    # For loop to run through each label of cluster label
    # for i in range(len(np.unique(cluster_labels))):
    for i in np.unique(cluster_labels):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num

    return reference_labels


def normalize_prediction(reference_labels, pred_logits, num_classes):
    np.nan_to_num(pred_logits, copy=False, nan=0)
    true_pred_ref = collections.defaultdict(list)

    for pred, true in reference_labels.items():
        true_pred_ref[true].append(pred)

    result = np.zeros((num_classes, len(pred_logits)))


    for true in range(num_classes):
        result[true] = np.sum(pred_logits[:,np.array(true_pred_ref[true])],1)
        # for pred in true_pred_ref[true]:
        #     result[true] += pred_logits[:, pred]

    result = result.clip(min=0.0000001, max=0.9999)
    return np.transpose(result)




def check_mnist_dataset_exists(path_data='./data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt')
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'mnist/train_data.pt')
        torch.save(train_label,path_data + 'mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'mnist/test_data.pt')
        torch.save(test_label,path_data + 'mnist/test_label.pt')
    return path_data


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).double().to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)



def print_msg(dataset_name, dataset_size, ft, ft_params, bin_data, noise_val):
    acc = bin_data['avg_accuracy']
    ece = bin_data['expected_calibration_error']
    avg_conf = bin_data['avg_confidence']

    msg = 'Dataset: {dataset_name}\t' \
          'Size: {dataset_size}\t' \
          'Noise: {noise_val}\t' \
          'fine-tune: {ft}\t' \
          'FT-params: {ft_params}\t' \
          'ECE: {ece:.2f}\t' \
          'Acc: {acc:.5f}\t' \
          'Avg-Conf: {avg_conf:.5f}\t' \
        .format(dataset_name=dataset_name,
                dataset_size=dataset_size,
                noise_val=noise_val,
                ft=ft,
                ft_params=ft_params,
                ece=ece,
                acc=acc,
                avg_conf=avg_conf
                )
    print(msg)

class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmaxes = F.softmax(logits, dim=1)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece*100

def sample(mu, var, nb_samples=500):
    """
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    out = []
    for i in range(nb_samples):
        out += [torch.normal(mu, var.sqrt())]
    return out


def get_toy_dataset(fen=100):
    train_mus = [[-1, 0], [1, 0]]
    test_mus = [[-math.cos(math.pi / 3), math.sin(math.pi / 3)], [math.cos(math.pi / 3), -math.sin(math.pi / 3)]]
    val_mus = [[-math.cos(math.pi / 6), math.sin(math.pi / 6)], [math.cos(math.pi / 6), -math.sin(math.pi / 6)]]
    vars = 0.5
    train_vars = [[vars, vars], [vars, vars]]
    test_vars = [[vars, vars], [vars, vars]]
    # val_vars = [[vars*0.5, vars*0.5], [vars*0.5, vars*0.5]]
    val_vars = [[vars, vars], [vars, vars]]
    n_components, n_samples = 2, 1000
    train_data = generate_datasets(train_mus, train_vars, n_components=n_components, n_samples=n_samples)
    test_data = generate_datasets(test_mus, test_vars, n_components=n_components, n_samples=n_samples)
    val_data = generate_datasets(val_mus, val_vars, n_components=n_components, n_samples=n_samples // fen)
    train_data = torch.stack(train_data, dim=0)
    test_data = torch.stack(test_data, dim=0)
    val_data = torch.stack(val_data, dim=0)
    train_label = torch.zeros(n_samples * 2)
    train_label[n_samples:] = 1
    val_label = torch.zeros(n_samples // fen * 2)
    val_label[n_samples // fen:] = 1
    return n_components, test_data, train_data, train_label, val_data, val_label


def generate_datasets(mus, vars, n_components, n_samples):
    data = []

    for i in range(n_components):
        mu = torch.Tensor(mus[i])
        var = torch.Tensor(vars[i])
        data.extend(sample(mu, var, n_samples))
    return data