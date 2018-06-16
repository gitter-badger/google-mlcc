# Input function: instructs tensorflow how to preprocess the data going in to
# the regressor


def linear_inputfunc(features, targets, batch_sz=1, shuffle=True, num_epochs=None):
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


def preprocess_chd(dataset, california_housing_dataframe):
    """
    :param dataset: str: "features" or "targets"
    :param california_housing_dataframe: pd.Dataframe The California Housing Dataframe
    :return: The processed data subset
    """

    import pandas as pd
    subset = None

    if dataset == "features":
        # Select features
        FEATURES = california_housing_dataframe[
            [
                "latitude",
                "longitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income"
            ]
        ]
        processed_features = FEATURES.copy()

        # Create a synthetic feature
        processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"]
        )

        subset = processed_features

    elif dataset == "targets":
        targets = pd.DataFrame()
        targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0
        )

        subset = targets

    return subset


def construct_feature_cols(input_features):
    """
    :param input_features: Names of numerical input features to use
    :return: A set of corresponding feature columns
    """
    import tensorflow as tf
    return set(
        [tf.feature_column.numeric_column(feat)
            for feat in input_features]
    )
