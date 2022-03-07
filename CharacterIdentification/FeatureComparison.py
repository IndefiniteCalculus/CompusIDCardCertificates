import math
import numpy as np
from CharacterIdentification import feature_type
# projection way is unavaliable, find char in a pre_given set from database may work but not worth to try
def which_char(quary_feature:list, gallary_feature:list, label:list)->[]:
    '''return available char list from quary, unavailable char would be replaced by None'''
    if feature_type.what_feature() == 'knn':
        K = 3
        char_list = []
        for q_feature in quary_feature:
            idx = 0
            dist = []
            for g_feature in gallary_feature:
                # 计算出所有数据集和待查字符的距离
                dist.append(np.sum(np.power(q_feature - g_feature, 2)))
            pass
            idx += 1
            min_dist_idx = []
            for top_k in range(K):
                # 找到距离最近的k个
                min_dist_idx = [None] + min_dist_idx
                for dist_idx in range(len(dist)):
                    if min_dist_idx[0] is None:
                        min_dist_idx[0] = dist_idx
                        continue
                    if dist[min_dist_idx[0]] > dist[dist_idx] and dist_idx not in min_dist_idx:
                        min_dist_idx[0] = dist_idx
            char_list.append((label[min_dist_idx[0]]))

    if feature_type.what_feature() == 'projection':
        char_list = []
        for q_feat in quary_feature:
            q0, q1 = q_feat[0], q_feat[1]
            min_dist = float("inf")
            g_target_idx,idx = 0,0

            for g_feat in gallary_feature:
                g0, g1 = g_feat[0], g_feat[1]
                d0 = np.sum(np.power(g0-q0, 2))
                d1 = np.sum(np.power(g1-q1, 2))
                if d0 < min_dist:
                    min_dist = d0 + d1
                    g_target_idx = idx
                idx += 1
            char_list.append(label[g_target_idx])
    pass
    if feature_type.what_feature() == 'pca':
        char_list = []
        for q_feat in quary_feature:
            # args = np.dot(gallary_feature,q_feat) 维度不匹配(100, 3982) (400,1)
            pass

def test_dist(quary_proj:list, gallary_proj:list, label:list):
    '''test usage'''
    xue = quary_proj[0]
    for idx in range(len(label)):
        if label[idx] == '学':
            a = label[idx]
            break
    xue0 = gallary_proj[idx]
    pass

