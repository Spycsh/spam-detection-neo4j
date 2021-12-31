import numpy as np
def pad_adjlist(x_data):
    # Get lengths of each row of data
    lens = np.array([len(x_data[i]) for i in range(len(x_data))])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        padded[i] = np.random.choice(x_data[i], mask.shape[1])
    padded[mask] = np.hstack((x_data[:]))
    return padded

print(pad_adjlist([[0, 1], [2], [3], [5], [4, 6]]))