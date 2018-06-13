# Input function: instructs tensorflow how to preprocess the data going in to
# the regressor


def preprocess(features, targets, batch_sz=1, shuffle=True, num_epochs=None):
    """
    :param features: pandas df of features
    :param targets: pandas df of targets
    :param batch_sz: Size of batches passed to the model
    :param shuffle: Whether to shuffle the data
    :param num_epochs: Repeat count; None = indefinitely
    :return: tuple of (features, labels) for the next data batch
    """
    import numpy as np
    from tensorflow.python.data import Dataset

    # Convert features to dict of [Column Name] --> [ np.array of values ]
    # In this example, we just have { "total_rooms": [ 1 ... n ] }
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a tf dataset
    ds = Dataset.from_tensor_slices((features, targets))

    # Configure the batch size & number of iterations
    ds = ds.batch(batch_sz).repeat(num_epochs)

    # Shuffle the data if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data in tuple (features, labels)
    return ds.make_one_shot_iterator().get_next()
