import models



if __name__ == '__main__':
    # model = models.get_mnist_local(mode='train', dataset_name='mnist')
    model = models.get_drebin_local(mode='train', dataset_name='drebin')
