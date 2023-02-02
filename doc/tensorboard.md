# How to use tensorboard

1. Delete old log directory
    ```
    $ rm -rf logs
    ```
1. Run training script
    ```
    $ python mnist_keras.py
    ```
1. Run tensorboards
    ```
    $ tensorboard --logdir=logs/
    ```
1. Open web server "http://localhost:6006/"