import random
#from skimage import io
import numpy as np
from glob import glob
import SimpleITK as sitk
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os

class Pipeline(object):

    
    def __init__(self, list_train ,Normalize=True):
        self.scans_train = list_train
        self.train_im=self.read_scans(Normalize)


    def read_scans(self,Normalize):

        train_im=[]
        for i in range(len( self.scans_train)):
            #if i%10==0:
            #    print('iteration [{}]'.format(i))

            #print(len(self.scans_train[i]))

            flair = glob( self.scans_train[i] + '/*_flair.nii.gz')
            t2 = glob( self.scans_train[i] + '/*_t2.nii.gz')
            gt = glob( self.scans_train[i] + '/*_seg.nii.gz')
            t1 = glob( self.scans_train[i] + '/*_t1.nii.gz')
            t1c = glob( self.scans_train[i] + '/*_t1ce.nii.gz')

            t1s=[scan for scan in t1 if scan not in t1c]
            #print(len(flair)+len(t2)+len(gt)+len(t1s)+len(t1c))
            if (len(flair)+len(t2)+len(gt)+len(t1s)+len(t1c))<5:
                print("there is a problem here!!! the problem lies in this patient :", self.scans_train[i])
                continue
            scans = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
            
            #read a volume composed of 4 modalities
            tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]

            #crop each volume to have a size of (146,192,152) to discard some unwanted background and thus save some computational power ;)
            z0=1
            y0=29
            x0=42
            z1=147
            y1=221  
            x1=194  
            #tmp=np.array(tmp)
            #tmp=tmp[:,z0:z1,y0:y1,x0:x1]

            #normalize each slice
            if Normalize==True:
                tmp=self.norm_slices(tmp)

            train_im.append(tmp)
            del tmp    
        # plt.imshow(np.array(train_im).transpose((0, 2, 3, 4, 1))[0, 78, :, :, 0])
        # plt.show()

        return  np.array(train_im)
    
    
    def sample_patches_randomly(self, num_patches, d , h , w ):

        '''
        INPUT:
        num_patches : the total number of samled patches
        d : this correspnds to the number of channels which is ,in our case, 4 MRI modalities
        h : height of the patch
        w : width of the patch
        OUTPUT:
        patches : np array containing the randomly sampled patches
        labels : np array containing the corresping target patches
        '''
        patches, labels = [], []
        count = 0

        #swap axes to make axis 0 represents the modality and axis 1 represents the slice. take the ground truth
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]
        #print(gt_im.shape)

        #take flair image as mask
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        #save the shape of the grounf truth to use it afterwards
        tmp_shp = gt_im.shape

        #reshape the mask and the ground truth to 1D array
        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        # maintain list of 1D indices while discarding 0 intensities
        indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
        del msk

        # shuffle the list of indices of the class
        np.random.shuffle(indices)

        #reshape gt_im
        gt_im = gt_im.reshape(tmp_shp)

        #a loop to sample the patches from the images
        i = 0
        pix = len(indices)
        while (count<num_patches) and (pix>i):
            #randomly choose an index
            ind = indices[i]
            i+= 1
            #reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            # get the patient and the slice id
            patient_id = ind[0]
            slice_idx=ind[1]
            p = ind[2:]
            #construct the patch by defining the coordinates
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x=list(map(int,p_x))
            p_y=list(map(int,p_y))
            
            #take patches from all modalities and group them together
            tmp = self.train_im[patient_id][0:4, slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
            #take the coresponding label patch
            lbl=gt_im[patient_id,slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]

            #keep only paches that have the desired size
            if tmp.shape != (d, h, w) :
                continue
            patches.append(tmp)
            labels.append(lbl)
            count+=1
        patches = np.array(patches)

        labels=np.array(labels)
        return patches, labels
        
        

    def norm_slices(self,slice_not): 
        '''
            normalizes each slice , excluding gt
            subtracts mean and div by std dev for each slice
            clips top and bottom one percent of pixel intensities
        '''
        normed_slices = np.zeros((5, 155, 240, 240)).astype(np.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(146):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1]=slice_not[-1]

        return normed_slices    
   


    def _normalize(self,slice):
        '''
            input: unnormalized slice 
            OUTPUT: normalized clipped slice
        '''
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice)==0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            #since the range of intensities is between 0 and 5000 ,the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            #the min is replaced with -9 just to keep track of 0 intensities so that we can discard those intensities afterwards when sampling random patches
            #tmp[tmp==tmp.min()]=-9
            return (tmp + abs(np.min(tmp)))/(np.max(tmp) + abs(np.min(tmp)))


def generate_whole_images(val = False):

    # Paths for Brats2017 dataset
    path_train = glob('/media/balaji/CamelyonProject/brats_2018/train/**')

    path_val = glob('/media/balaji/CamelyonProject/brats_2018/val/**')

    savepath_train = '/media/balaji/CamelyonProject/parth/slices_scaled/train/'
    
    savepath_val = '/media/balaji/CamelyonProject/parth/slices_scaled/val/'

    if val == True:
        path_all = path_val
        savepath = savepath_val
    else:
        path_all = path_train
        savepath = savepath_train

    os.makedirs(savepath, exist_ok = True)

    # shuffle the dataset
    np.random.seed(2022)
    np.random.shuffle(path_all)

    print(len(path_all))

    print('Extracting Patches.....')
    for i in range(len(path_all)):
        try:
            start = i
            end = (i + 1)

            pipe = Pipeline(list_train=path_all[start:end], Normalize=True)
            train_im = np.squeeze(pipe.train_im).transpose((1, 2, 3, 0))
            #print(train_im.shape)
            # Separate image and mask
            Patches = train_im[:, :, :, :4]
            Y_labels = train_im[:, :, :, 4]

            # transform the data to channels_last keras format
            # Patches=np.transpose(Patches,(0,2,3,1)).astype(np.float32)

            # since the brats2017 dataset has only 4 labels,namely 0,1,2 and 4 as opposed to previous datasets
            # this transormation is done so that we will have 4 classes when we one-hot encode the targets
            Y_labels[Y_labels == 4] = 3

            # transform y to one_hot enconding for keras
            shp = Y_labels.shape[0]
            Y_labels = Y_labels.reshape(-1)
            Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
            Y_labels = Y_labels.reshape(shp, 240, 240, 4)

            # plt.imshow(Y_labels[78][:, :, 1])
            # plt.show()
            #Save images, masks
            counter = 0
            for j in range(Patches.shape[0]):
                if np.std(Patches[j]) == 0:
                    if counter<5:
                        np.save(savepath + "patches/patch_%d_%d.npy" % (i, j), Patches[j])
                        np.save(savepath + "masks/label_%d_%d.npy" % (i, j), Y_labels[j])
                        counter +=1
                else:
                    np.save(savepath + "patches/patch_%d_%d.npy" % (i, j), Patches[j])
                    np.save(savepath + "masks/label_%d_%d.npy" % (i, j), Y_labels[j])
                #print(Patches[j].shape)
        except Exception as e:
            print(e)



def generate_patches(val = False):

    # Paths for Brats2017 dataset
    path_train = glob('/home/parth/Interpretable_ML/BraTS_2018/train/**')

    path_val = glob('/home/parth/Interpretable_ML/BraTS_2018/val/**')

    if val == True:
        path_all = path_val
    else:
        path_all = path_train

    np.random.seed(2022)
    np.random.shuffle(path_all)

    print(len(path_all))

    for i in range(len(path_all)):
        try:
            start = i
            end = (i + 1)

            # set the total number of patches
            # this formula extracts approximately 3 patches per slice
            num_patches = 146 * (end - start) * 3
            # define the size of a patch
            h = 128
            w = 128
            d = 4

            pipe = Pipeline(list_train=path_all[start:end], Normalize=True)
            Patches,Y_labels=pipe.sample_patches_randomly(num_patches,d, h, w)
            # print(Patches.shape)

            # transform the data to channels_last keras format
            Patches=np.transpose(Patches,(0,2,3,1)).astype(np.float32)

            # since the brats2017 dataset has only 4 labels,namely 0,1,2 and 4 as opposed to previous datasets
            # this transormation is done so that we will have 4 classes when we one-hot encode the targets
            Y_labels[Y_labels == 4] = 3

            # transform y to one_hot enconding for keras
            shp = Y_labels.shape[0]
            Y_labels = Y_labels.reshape(-1)
            Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
            Y_labels = Y_labels.reshape(shp, h, w, 4)

            for j in range(Patches.shape[0]):
                np.save("/media/parth/DATA/brats_slices/val/slice_%d_%d.npy" % (i, j), Patches[j])
                np.save("/media/parth/DATA/brats_slices/val_labels/slice_%d_%d.npy" % (i, j), Y_labels[j])
        except:
            pass


if __name__ == '__main__':

    # For 240 x 240 slices
    generate_whole_images(val=False)
    generate_whole_images(val=True)

    # For 128 x 128 patches
    #generate_patches()



    '''
    #Paths for Brats2017 dataset
    #path_HGG = glob('/home/parth/Interpretable_ML/BraTS_2018/train/**')
    path_LGG = glob('/home/parth/Interpretable_ML/BraTS_2018/val/**')
    path_all=path_LGG

    #shuffle the dataset
    np.random.seed(2022)
    np.random.shuffle(path_all)
    np.random.seed(1555)

    index = random.randint(0, len(path_all) + 1)

    print(len(path_all))


    for i in range(len(path_all)):
        try:
            start=i
            end=(i+1)

            #set the total number of patches
            #this formula extracts approximately 3 patches per slice
            num_patches=146*(end-start)*3
            #define the size of a patch
            h=128
            w=128
            d=4

            pipe=Pipeline(list_train=path_all[start:end],Normalize=True)
            #Patches,Y_labels=pipe.sample_patches_randomly(num_patches,d, h, w)
            #print(Patches.shape)
            Patches = pipe.train_im[:, :4, :, :, :].reshape((155, 240, 240, 4))
            Y_labels = pipe.train_im[:, 4, :, :, :].reshape((155, 240, 240, 1))
            #transform the data to channels_last keras format
            #Patches=np.transpose(Patches,(0,2,3,1)).astype(np.float32)

            # since the brats2017 dataset has only 4 labels,namely 0,1,2 and 4 as opposed to previous datasets
            # this transormation is done so that we will have 4 classes when we one-hot encode the targets
            Y_labels[Y_labels==4]=3


            #transform y to one_hot enconding for keras
            shp=Y_labels.shape[0]
            Y_labels=Y_labels.reshape(-1)
            Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
            Y_labels=Y_labels.reshape(shp,240,240,4)

            for j in range(Patches.shape[0]):
                np.save("/media/parth/DATA/brats_slices/val/slice_%d_%d.npy" %(i,j), Patches[j])
                np.save( "/media/parth/DATA/brats_slices/val_labels/slice_%d_%d.npy" %(i,j),Y_labels[j])
                print(Patches[j].shape)

            


            print("Size of the patches : ",Patches.shape)
            print("Size of their correponding targets : ",Y_labels.shape)
        except Exception as e:
            print(e)

            #save to disk as npy files
            #np.save( "/media/parth/DATA/brats_as_npy_low_res/train/x_dataset_{}.npy".format(i),Patches )
            #np.save( "/media/parth/DATA/brats_as_npy_low_res/train/y_dataset_{}.npy".format(i),Y_labels)'''





