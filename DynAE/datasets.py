import numpy as np
from scipy import ndimage, misc
import re
import matplotlib.image as mpimg
import os
import math 
import pandas as pd

def rgb2gray(rgb):
    #convert image from rgp to grayscale
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x[0:1000]])
    x1 = preprocess_input(x1)
    features = feature_model.predict(x1)
    for i in range(int(x.shape[0] / 1000)-1):
      x1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x[1000*(i+1):1000*(i+2)]])
      x1 = preprocess_input(x1)  # data - 127. #data/255.#
      features1 = feature_model.predict(x1)
      features = np.concatenate((features, features1)).astype(float)
    print('Features shape = ', features.shape)
    return features

def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y}

        )

def load_mice_protein(data_path='/content/drive/My Drive/Colab/ADEC/data/mice-protein'):
    mouse_data = pd.read_csv(data_path + '/Data_Cortex_Nuclear.csv')
    x = mouse_data.to_numpy()[:, 1:78]
    y =  mouse_data['class'].to_numpy()
    for i in range(1080):
      if y[i] == 'c-CS-s':
        y[i] = 0  
      elif y[i] == 'c-CS-m':
        y[i] = 1
      elif y[i] == 't-CS-s':
        y[i] = 2
      elif y[i] == 't-CS-m':
        y[i] = 3
      elif y[i] == 'c-SC-s':
        y[i] = 4
      elif y[i] == 'c-SC-m':
        y[i] = 5
      elif y[i] == 't-SC-s':
        y[i] = 6
      elif y[i] == 't-SC-m':
        y[i] = 7
      for j in range(77):
        if math.isnan(x[i, j]):
          x[i, j] = 0.
    p = np.random.permutation(x.shape[0])
    x = x[p,:]
    y = y[p]
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('Mice-protein samples ', x.shape))
    return x, y

def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y

def load_coil(data_path='/content/drive/My Drive/Colab/ADEC/data/COIL'):
    if not os.path.exists(data_path + '/data.npy'):
        X = np.zeros((1440, 128 , 128))
        Y = np.zeros((1440, ), dtype=int)
        i = 0
        for filename in os.listdir(data_path):
            image = mpimg.imread(data_path + "/" + filename)
            X[i, :] = image
            Y[i] = int(filename[3: filename.rfind('_')-1]) - 1
            i = i + 1
            print(i)
        p = np.random.permutation(X.shape[0])
        X = X.reshape([-1, 128, 128, 1]) / 255.0
        X = X[p]
        Y = Y[p]
        data = {'X': X, 'Y': Y} 
        file_name = data_path + '/data.npy'
        np.save(file_name, data)
    else:
        file_name = data_path + '/data.npy'
        data = np.load(file_name, allow_pickle=True)[()]
        X = data['X']
        Y = data['Y']
    print('COIL samples', X.shape)
    return X, Y

def load_coil_100_color(data_path='/content/drive/My Drive/Colab/ADEC/data/coil-100'):
    if not os.path.exists(data_path + '/data_color.npy'):
        X = np.zeros((7200, 128 , 128, 3))
        Y = np.zeros((7200, ), dtype=int)
        i = 0
        for filename in os.listdir(data_path):
            #if re.search("\.(jpg|gif|jpeg|png|bmp|tiff)$", filename):
            image = mpimg.imread(data_path + "/" + filename)
            X[i, :] = image
            Y[i] = int(filename[3: filename.rfind('_')-1]) - 1
            i = i + 1
            print(i)
        p = np.random.permutation(X.shape[0])
        X = X.reshape([-1, 128, 128, 3]) / 255.0
        X = X[p]
        Y = Y[p]
        data = {'X': X, 'Y': Y} 
        file_name = data_path + '/data_color.npy'
        np.save(file_name, data)
    else:
        file_name = data_path + '/data_color.npy'
        data = np.load(file_name, allow_pickle=True)[()]
        X = data['X']
        Y = data['Y']
    print('COIL-100 samples', X.shape)
    return X, Y

def load_coil_100_grayscale(data_path='/content/drive/My Drive/Colab/ADEC/data/coil-100'):
    if not os.path.exists(data_path + '/data.npy'):
        X = np.zeros((7200, 128 , 128))
        Y = np.zeros((7200, ), dtype=int)
        i = 0
        for filename in os.listdir(data_path):
            image = mpimg.imread(data_path + "/" + filename)
            image = rgb2gray(image) 
            X[i, :] = image
            Y[i] = int(filename[3: filename.rfind('_')-1]) - 1
            print(i)
            i = i + 1
        p = np.random.permutation(X.shape[0])
        X = X.reshape([-1, 128, 128, 1]) / 255.0
        X = X[p]
        Y = Y[p]
        data = {'X': X, 'Y': Y} 
        file_name = data_path + '/data.npy'
        np.save(file_name, data)
    else:
        file_name = data_path + '/data.npy'
        data = np.load(file_name, allow_pickle=True)[()]
        X = data['X']
        Y = data['Y']
    print('COIL-100 samples', X.shape)
    return X, Y

def load_mnist_test():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y

def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y

def load_pendigits(data_path='/content/drive/My Drive/Colab/DAE/data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y

def load_usps(data_path='/content/drive/My Drive/Colab/DAE/data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y

def load_reuters(data_path='/content/drive/My Drive/Colab/DAE/data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y

def load_cifar10(data_path='/content/drive/My Drive/Colab/DAE/data/cifar10'):
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy'):
        return np.load(data_path + '/cifar10_features.npy', allow_pickle=True), y

    # extract features
    features = np.zeros((60000, 4096))
    for i in range(6):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/cifar10_features.npy', features)
    print('features saved to ' + data_path + '/cifar10_features.npy')

    return features, y

def load_stl(data_path='/content/drive/My Drive/Colab/DAE/data/stl'):
    import os
    assert os.path.exists(data_path + '/stl_features.npy') or os.path.exists(data_path + '/train_X.bin'), \
        "No data! Use %s/get_data.sh to get data ready, then come back" % data_path

    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))
    y1 = None
    y2 = None

    # if features are ready, return them
    if os.path.exists(data_path + '/stl_features.npy'):
        return np.load(data_path + '/stl_features.npy', allow_pickle=True), y

    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)
    x1 = None
    x2 = None
    
    # extract features
    features = extract_vgg16_features(x)

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/stl_features.npy', features)
    print('features saved to ' + data_path + '/stl_features.npy')

    return features, y

def load_data_conv(dataset, datapath):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps(datapath)
    elif dataset == 'pendigits':
        return load_pendigits(datapath)
    elif dataset == 'reuters10k' or dataset== 'reuters':
        return load_reuters(datapath)
    elif dataset == 'stl':
        return load_stl(datapath)
    elif dataset == 'cifar10':
        return load_cifar10(datapath)
    elif dataset == 'coil':
        return load_coil(datapath)
    elif dataset == 'coil-100':
        return load_coil_100_grayscale(datapath)
    elif dataset == 'coil-100-color':
        return load_coil_100_color(datapath)
    elif dataset == 'mice-protein':
        return load_mice_protein(datapath)
    else:
        raise ValueError('Not defined for loading %s' % dataset)

def load_data(dataset, datapath):
    x, y = load_data_conv(dataset, datapath)
    return x.reshape([x.shape[0], -1]), y

def generate_data_batch(x, y=None, batch_size=256):
    index_array = np.arange(x.shape[0])
    index = 0
    while True:
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
        if y is None:
            yield x[idx]
        else: 
            yield x[idx], y[idx]

def generate_transformed_batch(x, datagen, batch_size=256):
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            yield batch_x
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
            yield batch_x      

def generate(x, datagen, batch_size=256):
    gen1 = generate_data_batch(x, batch_size=batch_size)
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x2 = next(gen1)
            yield (batch_x1, batch_x2)
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x1 = np.reshape(batch_x1, [batch_x1.shape[0], x.shape[-1]])
            batch_x2 = next(gen1)
            yield (batch_x1, batch_x2)

def convert_shape_fc_to_conv(image_shape):
    width = int(np.sqrt(image_shape[-1]))
    if width * width == image_shape[-1]:  # Gray
        im_shape = (image_shape[0], width, width, 1)
    else:  # RGB
        width = int(np.sqrt(image_shape[-1] / 3.0))
        im_shape = (image_shape[0], width, width, 3)
    return im_shape
  
def convert_shape_conv_to_fc(image_shape):
    im_shape = (image_shape[0], np.prod(image_shape[1:]))
    return im_shape 
