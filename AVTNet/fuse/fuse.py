import glob
import numpy as np
from sklearn.cluster import KMeans


data_root = '/home/insomnia/Video_Position/code/fuse/submission_B_3/'
save_path = '/home/insomnia/Video_Position/code/fuse/submission_B_3/submission.csv'
all_submission_path = glob.glob(data_root + '*.csv')
print(all_submission_path)

submissions = []
for path in all_submission_path:
    with open(path, 'r') as file:
        submissions.append(file.readlines())

with open(save_path, 'w') as file:
    for i in range(len(submissions[0])):
        positions = []
        for j in range(len(submissions)):
            positions.append(float(submissions[j][i].strip().split(',')[-1]))
        positions = np.stack(positions, axis=0).reshape((-1,1))
        out = KMeans(n_clusters=2).fit_predict(positions)
        mean_position = 0
        if np.sum(out) > out.shape[0] / 2:
            # 1 多
            for k in range(out.shape[0]):
                if out[k] == 1:
                    mean_position += positions[k]
            mean_position /= np.sum(out)
        elif np.sum(out) < out.shape[0] / 2:
            # 0 多
            for k in range(out.shape[0]):
                if out[k] == 0:
                    mean_position += positions[k]
            mean_position /= np.sum(1 - out)
        file.write(','.join(submissions[0][i].strip().split(',')[:2]) + ',' + str(round(mean_position[0], 3)) + '\n')
