import numpy as np

from sklearn.preprocessing import normalize


class SlidingWindow:
    def __init__(self, categories):
        self.categories = categories

    # Create 3d array
    def create_time_series(self, filepath, n_timesteps):
        time_series = []
        Y = []
        first_y = True
        first_windows = True
        for category in self.categories:
            print(category)
            data = np.load(filepath + "/" + category + ".npy")
            data = normalize(data, axis=1)
            windows = self.create_sliding_windows(data, n_timesteps)
            y_category = self.create_Y_for_category(category, len(windows))
            if first_y:
                Y = np.array(y_category)
                first_y = False
            else:
                Y = np.vstack((Y, y_category))

            if first_windows:
                time_series = windows
                first_windows = False
            else:
                time_series = np.vstack((time_series, windows))

        return time_series, Y


    # Create windows of n_timesteps BY ADDING PADDING AND THEN SPLITTING THE ARRAY
    def split_into_windows(self, X, n_timesteps):

        windows = []

        # Fill array until it can be split into windows of 10 timesteps (elements)
        while len(X) % n_timesteps != 0:
            X = np.row_stack((X, X[len(X) - 1]))

        windows = np.array_split(X, (len(X) / n_timesteps), axis=0)
        return np.array(windows)

    # Create y data for one category

    def create_Y_for_category(self, category, windows_length):

        indexLabel = self.categories.index(category)
        # Y = np.full((windows_length, 1), indexLabel)
        Y = np.stack([int(category == i) for i in self.categories] for _ in range(windows_length))
        return Y

    # Create sliding windows of n_timesteps
    def create_sliding_windows(self, data, n_timesteps):

        windows = []
        for i in range(data.shape[0] - n_timesteps):
            windows.append(data[i: i + n_timesteps])

        return np.array(windows)

    # Big function to extract features
    def extract_features(self, filepath, n_timesteps):
        X, Y = self.create_time_series(filepath, n_timesteps)

        return X, Y
