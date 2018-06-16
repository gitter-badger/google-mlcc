def train_model(
        learning_rate,
        steps,
        batch_sz,
        training_observations,
        training_targets,
        validation_observations,
        validation_targets):

    """
    :param learning_rate: float, the learning rate
    :param steps: int, the total number of training steps (fwd & bckwd pass over
        a single batch)
    :param batch_sz: int, the batch size
    :param training_observations: pd.DataFrame, the input features for training
        from the california_housing_dataframe
    :param training_targets: pd.DataFrame, with one column (the target label)
        to use for training
    :param validation_observations: pd.DataFrame, the input features for
        validation from the california_housing_dataframe
    :param validation_targets: pd.DataFrame, with one column (the target label)
        to use for validation
    :return: A LinearRegressor trained on the input data
    """
    import math
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from matplotlib import pyplot as plt
    from sklearn import metrics
    from . import construct_feature_cols, linear_inputfunc

    periods = 10
    steps_per_period = steps / periods

    # Create the LinearRegressor
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_cols(training_observations),
        optimizer=optimizer
    )

    # Input funcs
    training_input_func = lambda: linear_inputfunc(
        training_observations,
        training_targets["median_house_value"],
        batch_sz
    )

    predict_training_input_func = lambda: linear_inputfunc(
        training_observations,
        training_targets["median_house_value"],
        shuffle=False,
        num_epochs=1
    )

    predict_validation_input_func = lambda: linear_inputfunc(
        validation_observations,
        validation_targets["median_house_value"],
        shuffle=False,
        num_epochs=1
    )

    # Train the model
    print("RMSE (on training data):")
    training_rmses = list()
    validation_rmses = list()

    validation_predictions = None
    training_predictions = None
    for period in range(0, periods):
        # Train the model, beginning with its previous state
        linear_regressor.train(
            input_fn=training_input_func,
            steps=steps_per_period
        )

        # Compute predictions
        training_predictions = linear_regressor.predict(predict_training_input_func)
        training_predictions = np.array([item["predictions"][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(predict_validation_input_func)
        validation_predictions = np.array([item["predictions"][0] for item in validation_predictions])

        # Compute loss
        training_rmse = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets)
        )

        validation_rmse = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets)
        )

        print("Period: {}\tLoss: {:.2f}".format(period, training_rmse))
        training_rmses.append(training_rmse)
        validation_rmses.append(validation_rmse)

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 3, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmses, label="training")
    plt.plot(validation_rmses, label="validation")
    plt.legend()

    calibration_data = pd.DataFrame()
    calibration_data["training_predictions"] = pd.Series(training_predictions)
    calibration_data["validation_predictions"] = pd.Series(validation_predictions)

    return linear_regressor, calibration_data
