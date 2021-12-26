import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from utils import _ECELoss
from os.path import exists
# sns.set(style="white", font="Arial")
# colors = sns.color_palette("Paired", n_colors=12).as_hex()

import numpy as np
import torch

# from gmm import GaussianMixture
from math import sqrt
from utils import *
from reliability_diagram import reliability_diagrams, reliability_diagram




def save_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Lambda(lambda x: x.repeat(3,1,1)),
         # torchvision.transforms.Resize((32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         AddGaussianNoise(0,2)])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=2)


    # trainset = torchvision.datasets.MNIST(root='./data/mnist_p', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
    #                                           shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.MNIST(root='./data/mnist_p', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
    #                                          shuffle=False, num_workers=2)
    # trainset = torchvision.datasets.USPS(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
    #                                           shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.USPS(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
    #                                          shuffle=False, num_workers=2)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    vgg16 = models.vgg16(pretrained=True)


    loaders = [trainloader, testloader]
    # loaders = [testloader]
    features = []
    labels = []

    for loader in loaders:
        # feature = []
        # for images, labels in loader:
        #     images = images.to(device)
        #     feature.append(vgg16(images).cpu())
        # # labels = labels.to(device)
        # feature_tensor = torch.stack(feature)
        dataiter = iter(loader)
        images, label = dataiter.next()
        features.append(vgg16(images))
        labels.append(label)

    data_path = Constant.data_path

    torch.save(features[0].detach(), data_path + Constant.dataset_name + '_train_features_vgg16.pt')
    torch.save(features[1].detach(), data_path + Constant.dataset_name + '_test_features_vgg16.pt')
    torch.save(labels[0].detach(), data_path + Constant.dataset_name + '_train_labels_vgg16.pt')
    torch.save(labels[1].detach(), data_path + Constant.dataset_name + '_test_labels_vgg16.pt')
    # torch.save(features[0].detach(), data_path + 'cifar10_train_features_vgg16.pt')
    # torch.save(features[1].detach(), data_path + 'cifar10_test_features_vgg16.pt')
    # torch.save(labels[0].detach(), data_path + 'cifar10_train_labels_vgg16.pt')
    # torch.save(labels[1].detach(), data_path + 'cifar10_test_labels_vgg16.pt')




    return features, labels

num_features = 100




def load_data(dataset_name):

    # data_path = './data/cifar10_processed/'
    # if not os.path.exists(data_path + dataset_name + '_train_features_vgg16.pt'):
    #     save_data()


    if dataset_name == 'mnist':
        data_path = './data/mnist_p/mnist_processed/'
    if dataset_name == 'usps':
        data_path = './data/usps_processed/'
    if dataset_name == 'cifar10':
        data_path = './data/cifar10_processed/'
    if dataset_name == 'cifar10c':
        data_path = './data/cifar10c_processed/'

    train_features = torch.load(data_path + dataset_name + '_train_features_vgg16.pt')
    test_features = torch.load(data_path + dataset_name + '_test_features_vgg16.pt')
    train_label = torch.load(data_path + dataset_name + '_train_labels_vgg16.pt')
    test_label = torch.load(data_path + dataset_name + '_test_labels_vgg16.pt')

    pca = PCA(n_components=num_features)
    train_features = pca.fit_transform(train_features.detach().numpy())
    test_features = pca.transform(test_features.detach().numpy())


    return torch.Tensor(train_features).numpy(), torch.Tensor(test_features).numpy(), train_label.numpy(), test_label.numpy()

def main():

    # train_data, test_data, train_label, test_label = load_data('cifar10')

    from utils import check_mnist_dataset_exists
    data_path = check_mnist_dataset_exists()

    train_img = torch.load(data_path + 'mnist/train_data.pt')
    train_label = torch.load(data_path + 'mnist/train_label.pt')
    test_img = torch.load(data_path + 'mnist/test_data.pt')
    test_label = torch.load(data_path + 'mnist/test_label.pt')

    train_img_fla = train_img.view(train_img.shape[0],784).numpy()
    test_img_fla = test_img.view(test_img.shape[0],784).numpy()
    train_label = train_label.numpy()
    test_label = test_label.numpy()

    pca = PCA(n_components=num_features)
    train_feature = pca.fit_transform(train_img_fla)
    test_feature = pca.transform(test_img_fla)

    model = GaussianMixture(n_components=100, random_state=0, verbose=2, max_iter=5, covariance_type='diag', warm_start=True)


    model.fit(train_feature)

    train_predict  = model.predict(train_feature)

    reference_labels = retrieve_info(train_predict, train_label)

    names = ['train', 'test']
    datasets = [train_feature, test_feature]
    labels = [train_label, test_label]
    for i in range(2):
        p_y = model.predict_proba(datasets[i])
        confidences = np.amax(p_y,1)
        pred_labels = p_y.argmax(1)
        pred_labels = np.array([reference_labels[x] for x in pred_labels])
        curr_label = labels[i]

        reliability_diagram(curr_label, pred_labels, confidences, num_bins=10,
                            draw_ece=True, draw_bin_importance=False,
                            draw_averages=True, title=names[i],
                            figsize=(6, 6), dpi=72, return_fig=False)

        pass


    # val_test_data, val_data, val_test_label, val_label = load_data('usps')

    for noise_val in [0.3]:
        val_test_data = train_img_fla + np.random.normal(0,noise_val,train_img_fla.shape)
        val_test_data = pca.transform(val_test_data)

        val_test_label = train_label
        val_data = test_img_fla + np.random.normal(0,noise_val, test_img_fla.shape)
        val_data = pca.transform(val_data)
        val_label = test_label

        val_data = val_data
        val_label = val_label

        for fine_tune in [False, True]:

            if fine_tune: model.fit(val_data)

            names = ['val ft_' + str(fine_tune), 'val_test ft_' + str(fine_tune)]
            datasets = [val_data, val_test_data]
            labels = [val_label, val_test_label]

            for i in range(2):
                p_y = model.predict_proba(datasets[i])
                confidences = np.amax(p_y,1)
                pred_labels = p_y.argmax(1)
                pred_labels = np.array([reference_labels[x] for x in pred_labels])
                curr_label = labels[i]

                reliability_diagram(curr_label, pred_labels, confidences, num_bins=10,
                                    draw_ece=True, draw_bin_importance=False,
                                    draw_averages=True, title=names[i],
                                    figsize=(6, 6), dpi=72, return_fig=False)

                pass








def get_labels_for_minst(model, data_array, label_array):
    labels = torch.zeros(len(label_array))

    num_component = max(label_array)

    for label in range(1, num_component + 1):
        select = label_array == label
        data = data_array[select]

        predicts = model.predict(data)
        corresponding_label = torch.mode(predicts).values.item()
        labels[select] = corresponding_label

    return labels




if __name__ == "__main__":
    main()
