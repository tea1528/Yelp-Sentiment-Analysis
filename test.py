import numpy as np


# def normalize(P):
#     norm = np.sum(P, axis=0, keepdims=False)
#     return P / norm

def normalize(P):
    K = P.shape[0]
    norm = np.sum(P, axis=0, keepdims=True)
    return (P+1) / (norm + K)


# P = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
data_delta = np.array([[1, 0],
                       [1, 0],
                       [0, 1],
                       [1, 0]])
P_y = normalize(np.sum(data_delta, axis=0, keepdims=False))

data_matrix = np.array([[0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0]])
P_xy = normalize(data_matrix.transpose().dot(data_delta))

log_P_y = np.expand_dims(np.log(P_y), axis=0)
log_P_xy = np.log(P_xy)
log_P_dy = data_matrix.dot(log_P_xy)
log_P = log_P_y + log_P_dy

print(log_P_y)
print(log_P_dy)
print(log_P)