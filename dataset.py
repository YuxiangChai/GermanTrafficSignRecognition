import torch.utils.data as data
import matplotlib.pyplot as plt
import csv
from PIL import Image


def read_traffic_signs(rootpath):
    """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels"""
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 43 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(int(row[7]))  # the 8th column is the label
        gtFile.close()
    return images, labels


# dataset = DataSet(path='./GTSRB/Final_Training/Images', transform=transform)
class DataSet(data.Dataset):
    def __init__(self, path, transform=None):
        self.images, self.labels = read_traffic_signs(path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        lbl = self.labels[index]
        if self.transform:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return len(self.images)
