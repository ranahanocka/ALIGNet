import h5py
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_image' , type=str, default='pumpkin.png', help='')
parser.add_argument('--hdf5file' , type=str, default='pumpkin.h5', help='')
opt = parser.parse_args()


def save_data(h5filename, key_values):
    f = h5py.File(h5filename, "w")
    for k,v in key_values:
        f.create_dataset(k, data=v)
    f.close()

def get_border(alpha):
    thr = 0.9
    shp = alpha.shape
    maxes = []
    for d in [(0,shp[0]-1,1),(shp[0]-1,-1,-1)]:
        for i in range(d[0],d[1],d[2]):
            if alpha[i,:].max() >= thr:
                maxes.append(i/float(shp[0]))
                break
    for d in [(0,shp[1]-1,1),(shp[1]-1,-1,-1)]:
        for i in range(d[0],d[1],d[2]):
            if alpha[:,i].max() >= thr:
                maxes.append(i/float(shp[1]))
                break
    return maxes


def png_to_hdf5(imagefile, hdf5file,noresize=False):
    sz = [128, 128]
    des = [11, 115, 11, 116]  # Where to put the shape

    mimg = mpimg.imread(imagefile)
    if mimg.shape[2] != 4:
        ValueError('currently only supports images with alpha')
    alphamap = mimg[:, :, 3]
    maxes = get_border(alphamap)
    p0 = abs(np.array(maxes[0:2]) * sz[0] - np.array(des[0:2]))
    p1 = abs(np.array(maxes[2:]) * sz[1] - np.array(des[2:]))
    psz = [int(sz[0] - p0[1] - p0[0]), int(sz[1] - p1[1] - p1[0])]
    img = Image.open(imagefile)
    imgr = img.resize(psz, Image.BICUBIC)
    new_im = Image.new(img.mode, sz)
    new_im.paste(imgr, ((sz[0] - psz[0]) // 2, (sz[1] - psz[1]) // 2))

    if noresize:
        new_im = mimg

    processed_im = np.array(new_im)
    processed_im = np.transpose(processed_im, (1, 0, 2)) # we transpose the image for the network.
    bin_im = np.array(processed_im[:, :, 3])
    if bin_im.max() > 10:
        bin_im = bin_im / 255
    datapaths = [['/dep/view1', bin_im], ['/txtr/view1', processed_im[:, :, 0:3]]]
    save_data(hdf5file, datapaths)


def view_from_hdf5(hdf5file):
    import matplotlib.pyplot as plt
    f = h5py.File(hdf5file, 'r')
    keys = ['/dep/view1','/txtr/view1']
    for key in keys:
        if key in f:
            plt.imshow(f[key])
            plt.show()
    f.close()


def convert_hdf5_to_im(hdf5file):
    f = h5py.File(hdf5file, 'r')
    data = f['/dep/view1']
    data = np.transpose(data, (1, 0))
    alpha = 255*np.array(np.expand_dims(data, axis=2),dtype='uint8')
    rgb = np.array(255*(1-np.repeat(alpha, 3, axis=2)),dtype='uint8')
    rgba = np.concatenate((rgb, alpha), axis=2)
    im = Image.fromarray(rgba)
    im.save("temp.png",gamma=0.45455)


if __name__ == '__main__':
    png_to_hdf5(opt.input_image, opt.hdf5file,noresize=True)
    view_from_hdf5(opt.hdf5file)
    # view_from_hdf5('/mnt/data/silhouettes/rendered/vase/h5files/0118.h5')
    # convert_hdf5_to_im('/mnt/data/silhouettes/rendered/vase/h5files/0118.h5')

