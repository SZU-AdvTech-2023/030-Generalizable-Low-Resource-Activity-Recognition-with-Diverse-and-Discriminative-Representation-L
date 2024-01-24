# Now this file only supports dsads and uschad datasets.
# Free to extend it for more datasets.

import sys
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(sys.path[0]))
import utils
from main import args_parse


# ====== DSADS ======
def merge_split_dsads(seed, root_path, save_file, n_domain=4):
    """
     For every activity, divide data into training, testing and validation sets.
    Data of 8 persons exist for every activity.
    """
    # load the npz file of dsads made in deal_dsads.py
    d = np.load(root_path)

    # extract values of x, y, s (labels start from 0)
    x, y, s = d['x'], (d['y'] - 1).reshape(-1,), (d['s'] - 1).reshape(-1,)

    data_lst = []
    for i in range(n_domain):
        data_i = []
        # divide 8 persons into pairs
        # [0, 1], [2, 3], [4, 5], [6, 7]
        d_index = np.argwhere((s == 2 * i) | (s == 2 * i + 1)).reshape(-1,)
        x_i = x[d_index, :, :]
        y_i = y[d_index]
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)

    divide_train_val_test(data_lst, n_domain, save_file, seed)


def divide_train_val_test(data_lst, n_domain, save_file, seed):
    domain_dic = []

    for i in range(n_domain):
        train_i, val_i, test_i, dic_i = [], [], [], []
        x_i, y_i = data_lst[i][0], data_lst[i][1]

        # 60% for training, 40% for testing and validation
        x_train_i, x_tmp_i, y_train_i, y_tmp_i = train_test_split(
            x_i, y_i, test_size=0.4, random_state=seed, stratify=y_i)
        # 50-50 for test and val
        x_val_i, x_test_i, y_val_i, y_test_i = train_test_split(
            x_tmp_i, y_tmp_i, test_size=0.5, random_state=seed, stratify=y_tmp_i)

        train_i.append(x_train_i)
        train_i.append(y_train_i)

        val_i.append(x_val_i)
        val_i.append(y_val_i)

        test_i.append(x_test_i)
        test_i.append(y_test_i)

        dic_i.append(train_i)
        dic_i.append(val_i)
        dic_i.append(test_i)
        domain_dic.append(dic_i)

    with open(save_file, 'wb') as f:
        pickle.dump(domain_dic, f)


# ====== USC-HAD ======
def merge_split_uschad(seed, root_path, save_file, n_domain=5):
    d = np.load(root_path)
    x, y, s = d['x'], (d['y']-1).reshape(-1,), d['s'].reshape(-1,)
    data_lst = []
    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    d_index_0 = np.argwhere((s == 1) | (s == 3) | (s == 10)).reshape(-1,)
    x_0 = x[d_index_0]
    y_0 = y[d_index_0]
    data_0.append(x_0)
    data_0.append(y_0)
    data_lst.append(data_0)

    d_index_1 = np.argwhere((s == 2) | (s == 5) | (s == 13)).reshape(-1,)
    x_1 = x[d_index_1]
    y_1 = y[d_index_1]
    data_1.append(x_1)
    data_1.append(y_1)
    data_lst.append(data_1)

    d_index_2 = np.argwhere((s == 4) | (s == 7) | (s == 9)).reshape(-1,)
    x_2 = x[d_index_2]
    y_2 = y[d_index_2]
    data_2.append(x_2)
    data_2.append(y_2)
    data_lst.append(data_2)

    d_index_3 = np.argwhere((s == 6) | (s == 8) | (s == 14)).reshape(-1,)
    x_3 = x[d_index_3]
    y_3 = y[d_index_3]
    data_3.append(x_3)
    data_3.append(y_3)
    data_lst.append(data_3)

    d_index_4 = np.argwhere((s == 11) | (s == 12)).reshape(-1,)
    x_4 = x[d_index_4]
    y_4 = y[d_index_4]
    data_4.append(x_4)
    data_4.append(y_4)
    data_lst.append(data_4)

    divide_train_val_test(data_lst, n_domain, save_file, seed)
    

# ============ PAMAP2 ===============


def merge_split_pamap(seed, root_path, save_file, n_domain=4):
    d = np.load(root_path)
    x, y, s = d['x'], d['y'].reshape(-1,), d['s'].reshape(-1,)
    x_new, y_new, s_new = select_sub_act(x, y, s)
    y_new = y_new-1
    s_new = s_new-1
    data_lst = []
    for i in range(n_domain):
        data_i = []
        d_index = np.argwhere((s_new == 2*i) | (s_new == 2*i+1)).reshape(-1,)
        x_i = x_new[d_index, :, :]
        y_i = y_new[d_index]
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)

    divide_train_val_test(data_lst, n_domain, save_file, seed)


def select_sub_act(x, y, s):
    x_new, y_new, s_new = [], [], []
    sub_list = [1, 2, 3, 4, 5, 6, 7, 8]
    act_list = [1, 2, 3, 4, 12, 13, 16, 17]
    for index in range(len(y)):
        if (s[index] in sub_list) and (y[index] in act_list):
            x_new.append(x[index])
            y_new.append(y[index])
            s_new.append(s[index])
        else:
            continue
    x_new, y_new, s_new = np.array(x_new), np.array(y_new), np.array(s_new)
    index_5 = np.argwhere(y_new == 12)
    y_new[index_5] = 5
    index_6 = np.argwhere(y_new == 13)
    y_new[index_6] = 6
    index_7 = np.argwhere(y_new == 16)
    y_new[index_7] = 7
    index_8 = np.argwhere(y_new == 17)
    y_new[index_8] = 8
    return x_new, y_new, s_new


# ====== HHAR ======
def merge_split_hhar(seed, root_path, save_file, n_domain=4):
    # load the npz file of dsads made in deal_hhar.py
    d = np.load(root_path)
    
    # extract values of x, y, s (labels start from 0)
    x, y, s = d['x'], d['y'], d['s']
    
    data_lst = []
    factor = y.shape[0] / n_domain
    s_index = 0
    e_index = int(s_index + factor)
    
    for i in range(n_domain):
        # simply divide dataset into 4 chunks
        data_i = []
        
        x_i = x[s_index:e_index, :, :]
        y_i = y[s_index:e_index]
        
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)
        
        s_index = e_index
        e_index = int(s_index + factor) if i != n_domain - 1 else y.shape[0]
    
    divide_train_val_test(data_lst, n_domain, save_file, seed)



if __name__ == '__main__':
    args = args_parse()

    for dataset in ['dsads', 'uschad', 'pamap', 'hhar']:
        for args.seed in range(1, 4, 1):
            utils.set_random_seed(args.seed)
            root_path = 'D:/xuxw/code/PyCharm/ddlearn-xw/data/process/' + \
                f'{dataset}/{dataset}_processwin.npz'
            save_file = 'D:/xuxw/code/PyCharm/ddlearn-xw/data/process/' + \
                f'{dataset}/{dataset}_subject_final_seed{args.seed}.pkl'
            if dataset == 'dsads':
                merge_split_dsads(args.seed, root_path, save_file, 4)
            elif dataset == 'pamap':
                merge_split_pamap(args.seed, root_path, save_file, 4)
            elif dataset == 'uschad':
                merge_split_uschad(args.seed, root_path, save_file, 5)
            elif dataset == 'hhar':
                merge_split_hhar(args.seed, root_path, save_file, 4)
            else:
                print('error')