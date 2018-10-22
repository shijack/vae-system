# coding=utf-8

import numpy as np

img_batch_r = np.arange(27 * 3).reshape((-1, 3, 3, 3))

tmp_a = img_batch_r[:img_batch_r.shape[0] / 2 + 1, ...]
tmp_b = img_batch_r[img_batch_r.shape[0] - img_batch_r.shape[0] / 2:, ...]
# todo if the tmp_b.shape[0]=1 can not mess up how to do ?
tmp_sim_a = np.ones(tmp_a.shape[0])
tmp_sim_b = np.zeros(tmp_b.shape[0])
b_sim = np.concatenate((tmp_sim_a, tmp_sim_b), axis=0)

mess_ok = False
tmp_index = list(np.random.permutation(tmp_b.shape[0]))

while not mess_ok and tmp_index:
    for i in range(tmp_b.shape[0] * tmp_b.shape[1] * tmp_b.shape[2] * tmp_b.shape[3]):
        print i == tmp_index[i]
        if i == tmp_index[i]:
            tmp_index = list(np.random.permutation(tmp_b.shape[0]))
            continue
    mess_ok = True
print tmp_index
tmp_b = tmp_b[tmp_index]

b_r = np.concatenate((tmp_a, tmp_b), axis=0)

print b_r
print b_sim
