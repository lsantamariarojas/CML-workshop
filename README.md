0. If you haven't, please install Github CLI ```winget install --id GitHub.cli```

1. Create a repo and do a first empty commit to main or master
2. Create a new local branch call 'experiment' by ```git checkout -b experiment```
3. Create the following files:

    ```python
    ## ./model.py

    # Import modules and packages
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt


    # Functions and procedures
    def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(6, 5))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend(shadow='True')
    # Set grids
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    # Some text
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    # Show
    plt.savefig('model_results.png', dpi=120)



    def mae(y_test, y_pred):
    """
    Calculuates mean absolute error between y_test and y_preds.
    """
    return tf.metrics.mean_absolute_error(y_test, y_pred)
    

    def mse(y_test, y_pred):
    """
    Calculates mean squared error between y_test and y_preds.
    """
    return tf.metrics.mean_squared_error(y_test, y_pred)


    # Check Tensorflow version
    print(tf.__version__)


    # Create features
    X = np.arange(-100, 100, 4)

    # Create labels
    y = np.arange(-90, 110, 4)


    # Split data into train and test sets
    X_train = X[:40] # first 40 examples (80% of data)
    y_train = y[:40]

    X_test = X[40:] # last 10 examples (20% of data)
    y_test = y[40:]


    # Take a single example of X
    input_shape = X[0].shape 

    # Take a single example of y
    output_shape = y[0].shape


    # Set random seed
    tf.random.set_seed(42)

    # Create a model using the Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1), 
        tf.keras.layers.Dense(1)
        ])

    # Compile the model
    model.compile(loss = tf.keras.losses.mae,
                optimizer = tf.keras.optimizers.SGD(),
                metrics = ['mae'])

    # Fit the model
    model.fit(X_train, y_train, epochs=100)


    # Make and plot predictions for model_1
    y_preds = model.predict(X_test)
    plot_predictions(train_data=X_train, train_labels=y_train,  test_data=X_test, test_labels=y_test,  predictions=y_preds)


    # Calculate model_1 metrics
    mae_1 = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
    mse_1 = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
    print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

    # Write metrics to file
    with open('metrics.txt', 'w') as outfile:
        outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

    ```

    And ```./requirements.txt```:

    ```
    tensorflow
    numpy
    matplotlib
    ```


4. Go into https://cml.dev/doc/start/github and copy the ```.github/workflows/cml.yaml``` file into a ```./.github/workflows/cml.yaml```:

    ```yaml
    # This runs a docker image based on the 'on' action and execute the steps below
    name: My-workflow-name
    on: [push]
    jobs:
    train-and-report:
    # docker image base
        runs-on: ubuntu-latest
        # container name
        container: docker://ghcr.io/iterative/cml:0-dvc2-base1
        steps:
        - uses: actions/checkout@v3
        - name: Train model
        # When needing to pass .env
            env:
            REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            # Steps to run in the container
            run: |
            ## Add any needed steps for your project
            pip install -r requirements.txt
            python train.py  # generate plot.png

            ### Automatic reporting set up
            # Create CML report
            cat metrics.txt >> report.md
            echo '![](./plot.png "Confusion Matrix")' >> report.md
            cml comment create report.md
    ```