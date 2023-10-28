import numpy as np
import cv2
import json

def decomposeP(P):
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    return K, R, t

P = np.array([
    [6.29238090e+02, -5.27036820e+02, -4.59938064e+00, 1.41321630e+03],
    [3.72013478e+02, 9.42970300e+00, -5.59685543e+02, 1.07637845e+03],
    [9.99925370e-01, 1.22165356e-02, 1.06612091e-04, 1.84000000e+00]
])



K, R, t = decomposeP(P)


extrinsic = np.identity(4)
extrinsic[:3,:3] = R
extrinsic[:3,3] = t

new_calb = {}
new_calb['intrinsic'] = np.round(K.flatten(), 4).tolist()
new_calb['extrinsic'] = np.round(extrinsic.flatten(), 4).tolist()

with open('cam-front-undistort.json', 'w') as f:
    json.dump(new_calb, f, indent=2)




# K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

# # Normalize the intrinsic matrix K
# K /= K[2, 2]

# print("Intrinsic Matrix K:")
# print(K)

# print("\nRotation Matrix R:")
# print(R)

# print("\nTranslation Vector T:")
# print(T)


# print(P)
# print(np.concatenate([R, T], axis=1))
# print(K@np.concatenate([R, T], axis=2))