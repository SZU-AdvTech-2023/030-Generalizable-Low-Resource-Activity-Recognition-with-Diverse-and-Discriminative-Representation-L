import numpy as np
import os



def load_data(root_path):
    data = np.load('../data/hhar/data_20_120.npy')
    label = np.load('../data/hhar/label_20_120.npy')
    
    X = data
    y = label[:, :, 2][:, 0]
    s = label[:, :, 0][:, 0]
    
    return X, y, s



if __name__ == '__main__':
    root_path = 'D:/xuxw/datasets/DSADS/data/'
    save_path = 'D:/xuxw/code/PyCharm/ddlearn-xw/data/process/hhar/'
    
    x, y ,s = load_data(root_path)
    
    if os.path.exists(save_path+'hhar_processwin.npz'):
        pass
    else:
        np.savez(save_path+'hhar_processwin.npz', x=x, y=y, s=s)