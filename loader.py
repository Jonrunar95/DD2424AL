import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from shutil import copy
from scipy import stats
from tqdm import tqdm

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

class ImageBase:
    def __init__(self, meta_loc, image_dir):
        self.meta_loc = meta_loc
        self.image_dir = image_dir
        self.meta   = self.load_meta(meta_loc)
        #self.images = self.load_images(image_dir)

    def load_meta(self, loc):
        return pd.read_csv(loc)

    def load_images(self, names):
        #load_path = os.path.join('data', '{0}_dat.pickle'.format(self.image_dir))
        imgs = {}
        for f in tqdm(names):
            img = Image.open(os.path.join(self.image_dir,f))
            img.load()
            imgs[f] = img
            #print('Loaded {0}/{1} images'.format(i+1, len(names)))

        return imgs
    def display_image_statistics(self, resolution = (100, 100)):
        print('Computing image statistics...')
        names = list(self.meta[self.meta['modality'] == 'X-ray']['filename'])
        pixels_list = []
        shapes = []
        for f in tqdm(names):
            img = Image.open(os.path.join(self.image_dir,f))
            img.load()
            
            pixels = np.prod(img.size)
            shapes.append([img.size[0],img.size[1]])
            pixels_list.append(pixels)
        
        shapes = np.array(shapes)

        stat = stats.describe(np.array(pixels_list))
        
        print('-- min pixel count: ', stat.minmax[0])
        print('-- max pixel count: ', stat.minmax[1])
        print('-- mean pixel count: ', stat.mean)
        print('-- variance pixel count: ', stat.variance)

        plt.hist2d(shapes[:,0], shapes[:,1], resolution, cmap=plt.cm.jet)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Distribution of image resolution')
        plt.colorbar()
        plt.show()
        print('Done!')

        
    def add_meta_field(self, col_name, default = ''):
        if col_name in self.meta:
            raise Exception("Field already exists in meta.")
        else:
            self.meta[col_name] = default
            self.meta.to_csv(self.meta_loc, index = False)

    def get_category(self, union, restrictions, shape, size = None, RGB_to_grayscale=True):
        '''
        Returns the union of reshaped images corresponding to a field having a certain property

        fields -> dictionary with key as desired field and container for desired properties
        '''

        imgs = set()

        '''
        Gather all outer data
        '''
        for fields in union:
            for field in fields:
                for prop in fields[field]:
                    for filename in list(self.meta[self.meta[field] == prop]['filename']):
                        imgs.add(filename)

        '''
        Impose restrictions
        '''
        for fields in restrictions:
            for field in fields:
                restrict_imgs = set()
                for prop in fields[field]:
                    for filename in list(self.meta[self.meta[field] == prop]['filename']):
                        restrict_imgs.add(filename)

                imgs = imgs.intersection(restrict_imgs)

        if size != None:
            try:
                imgs = np.random.choice([x for x in imgs], size = size, replace = False)
            except ValueError as identifier:
                raise Exception('{0} ---- max query size = {1}'.format(identifier, len(imgs)))
                
        images = self.load_images(imgs)
        arr = np.zeros((len(imgs), *shape))

        for i, img in enumerate(imgs):
            scaled_im = images[img].resize(shape, Image.ANTIALIAS)
            if RGB_to_grayscale:
                scaled_im = scaled_im.convert('L')
            scaled_im = np.asarray(scaled_im, dtype = 'int32')

            if len(scaled_im.shape) != len(shape):
                raise Exception('Dimension of images not correctly sepcified. Perhaps you fetched images with different channels or forgot to set RGB_to_grayscale = True?')
            else:
                arr[i, :, :] = scaled_im

        return arr

    def add_image(self, fields, im_loc):
        heading = {x : '' for x in self.meta}
        for key in fields:
            heading[key] = fields[key]

        self.meta = self.meta.append(heading, ignore_index = True)

        copy(im_loc, self.image_dir)
        self.meta.to_csv(self.meta_loc, index = False)
    ''' 
    def expand_meta(self):
        modifier = 'finding'
        source = 'filename'
        

        for x,y in zip(self.meta[modifier], self.meta[source]):
            if 'virus' in y and x == 'PNEUMONIA':
                self.meta.loc[self.meta[source] == y, modifier] += ', {0}'.format('VIRUS')
            if 'bacteria' in y and x == 'PNEUMONIA':
                self.meta.loc[self.meta[source] == y, modifier] += ', {0}'.format('BACTERIA')
        self.meta.to_csv(self.meta_loc, index = False)
    '''
    @staticmethod
    def montage(ims, figsize = (5,5), is_row = False, title = ''):
        shape = ims.shape[1:]
        num   = int(ims.shape[0]**0.5)**2

        rows = int(num**0.5)
        cols = rows

        if not is_row:
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize = figsize)
            
        else:
            num = ims.shape[0]
            fig, axes = plt.subplots(nrows = 1, ncols = num, figsize = figsize)
            rows = 1

        fig.suptitle(title)
        for idx in range(num):
            im = ims[idx]

            row = idx // cols
            col = idx % cols

            #print(axes.shape, row, col)
            if not is_row:
                axes[row, col].axis('off')
                axes[row, col].imshow(im)
            else:
                axes[idx].axis('off')
                axes[idx].imshow(im)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

if __name__ == "__main__":
    base = ImageBase('metadata.csv', 'images')
    #base.expand_meta()
    base.display_image_statistics()
    #ims = base.get_category(union = [{'modality' : ['X-ray']}], restrictions = [{'finding' : ['PNEUMONIA, BACTERIA']}] , shape = (256,256))

    #mageBase.montage(ims, figsize = (15, 15))
    '''
    dic = {'modality' : 'X-ray', 'finding' : 'NORMAL', 'filename' : ''}

    directory = 'kaggle_data'
    c = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        c += len(filenames)
    for dirpath, dirnames, filenames in os.walk(directory):
        if 'NORMAL' in dirpath:
            dic['finding'] = 'NORMAL'
            for f in filenames:
                print('{0} files remaining'.format(c))
                dic['filename'] = f
                base.add_image(dic, os.path.join(dirpath, f))
                c-=1
        else:
            dic['finding'] = 'PNEUMONIA'
            for f in filenames:
                print('{0} files remaining'.format(c))
                dic['filename'] = f
                base.add_image(dic, os.path.join(dirpath, f))
                c-=1
    '''