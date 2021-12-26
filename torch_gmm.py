# matplotlib.use('Agg')
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture
from utils import _ECELoss
# sns.set(style="white", font="Arial")
# colors = sns.color_palette("Paired", n_colors=12).as_hex()

from gmm import GaussianMixture
from utils import *
from reliability_diagram import reliability_diagram




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


    return features, labels

num_features = 40




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

criterion = _ECELoss(10)
def main():

    # train_data, test_data, train_label, test_label = load_data('cifar10')
    dataset = "toy"

    noise_value = (0.3, 0.3)
    if dataset == "mnist":
        model = GaussianMixture(n_components=200, n_features=num_features, covariance_type='diag')
        (train_feature, train_label), (val_feature, val_label), (test_feature, test_label) = \
            get_mninst_data(num_features, noise_value, 1000)
        num_classes = 10
    if dataset == "toy":
        model = GaussianMixture(n_components=2, n_features=2, covariance_type='full')
        n_components, test_feature, train_feature, train_label, val_feature, val_label = get_toy_dataset()
        test_label = train_label
        num_classes = 2

    train_feature_cuda = train_feature.to(device='cuda')
    model = model.to(device='cuda')
    model.fit(train_feature_cuda, n_iter=100)

    train_predict  = model.predict(train_feature_cuda).to(device='cpu')
    model.to(device='cpu')

    reference_labels = retrieve_info(train_predict, train_label)

    names = ['train', 'test']
    datasets = [train_feature, test_feature]
    labels = [train_label, test_label]
    for i in range(2):
        p_y = model.predict_proba(datasets[i])
        confidences, curr_label, logits_tensor, pred_labels = get_preds_true_conf(i, labels, reference_labels, p_y,num_classes)

        _, bin_data = reliability_diagram(curr_label, pred_labels, confidences, num_bins=10,
                            draw_ece=True, draw_bin_importance=False,
                            draw_averages=True, title=names[i],
                            figsize=(6, 6), dpi=72, return_fig=False)

        ece_loss = criterion(logits_tensor, torch.tensor(curr_label))
        print_msg(names[i], len(datasets[i]), False, "", bin_data, noise_value)
        pass

    for noise_val in [0.4]:

        for fine_tune in [True]:

            if fine_tune: model.fit(val_feature, warm_start=True, fix_mean=True, n_iter=100)

            # train_predict = model.predict(train_feature)
            # reference_labels = retrieve_info(train_predict.numpy(), train_label)

            # names = ['dev ft_' + str(fine_tune), 'dev_test ft_' + str(fine_tune)]
            names = ['dev ', 'dev_test']
            datasets = [val_feature, test_feature]
            labels = [val_label, test_label]

            for i in range(2):
                p_y = model.predict_proba(datasets[i])
                confidences, curr_label, logits_tensor, pred_labels = get_preds_true_conf(i, labels, reference_labels,
                                                                                          p_y,num_classes)
                _, bin_data = reliability_diagram(curr_label, pred_labels, confidences, num_bins=10,
                                    draw_ece=True, draw_bin_importance=False,
                                    draw_averages=True, title=names[i],
                                    figsize=(6, 6), dpi=72, return_fig=False)

                ece_loss = criterion(logits_tensor, torch.tensor(curr_label))
                print_msg(names[i], len(datasets[i]), fine_tune, "", bin_data, noise_value)
                pass

                pass


def get_preds_true_conf(i, labels, reference_labels, raw_probs, num_classes=10):
    curr_label = labels[i]
    logits = normalize_prediction(reference_labels, raw_probs.numpy(), num_classes)
    logits_tensor = torch.Tensor(logits)
    confidences = np.amax(logits, 1)
    pred_labels = np.argmax(logits, 1)

    curr_label = curr_label.numpy() if torch.is_tensor(curr_label) else curr_label


    return confidences, curr_label, logits_tensor, pred_labels


def get_mninst_data(num_features, noise_val, dev_sp_size):
    from utils import check_mnist_dataset_exists
    data_path = check_mnist_dataset_exists()
    large_set_img = torch.load(data_path + 'mnist/train_data.pt')
    large_set_label = torch.load(data_path + 'mnist/train_label.pt')
    small_set_img = torch.load(data_path + 'mnist/test_data.pt')
    small_set_label = torch.load(data_path + 'mnist/test_label.pt')

    ls_img_fla = large_set_img.view(large_set_img.shape[0], 784).numpy()
    ss_img_fla = small_set_img.view(small_set_img.shape[0], 784).numpy()
    large_set_label = large_set_label.numpy()
    small_set_label = small_set_label.numpy()

    pca = PCA(n_components=num_features)
    large_set_feature = pca.fit_transform(ls_img_fla)

    # fix training features and lable
    train_feature, train_label = large_set_feature[:30000], large_set_label[:30000]

    # add noise for dev and test set
    test_feature = feature_from_noisy_img(pca, ls_img_fla[30000:], noise_val[0])
    test_label = large_set_label[30000:]

    # Sample and transform dev set
    dev_img, dev_label = sample_all_label(ss_img_fla, small_set_label, dev_sp_size)
    dev_feature = feature_from_noisy_img(pca, dev_img, noise_val[1])

    return (torch.Tensor(train_feature), torch.Tensor(train_label)), \
           (torch.Tensor(dev_feature), torch.Tensor(dev_label)), \
           (torch.Tensor(test_feature), torch.Tensor(test_label))



    return pca, ss_img_fla, small_set_label, large_set_feature, ls_img_fla, large_set_label

def sample_all_label(data, label, sample_size):
    val_selected_label = np.array([])
    val_selected_data = np.array([])
    while len(np.unique(val_selected_label)) < 10:
        selected = np.random.choice(np.arange(len(data)), sample_size)
        val_selected_data, val_selected_label = data[selected], label[selected]
    label = val_selected_label
    data = val_selected_data
    return data, label

def feature_from_noisy_img(pca, imgs, noise_val):
    noise_img = imgs + np.random.normal(0, noise_val, imgs.shape)
    noise_img = pca.transform(noise_img)
    return noise_img


if __name__ == "__main__":
    main()
