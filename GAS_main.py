from data_loader import dataLoader
from GAS_model import GAS
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers


if __name__ == "main":
    # declare the super-parameters here
    lr = 0.001
    epochs = 30
    seed = 123
    batch_size = 1000
    momentum = 0.9

    dataLoader = dataLoader()
    adj_list, features, [X_train, X_test], y = dataLoader.obtain_graph()


    # initialize the GAS model
    model = GAS(adj_list, features, [X_train, X_test], y)
    optimizer = optimizers.Adam(lr)

    # train
    for _ in tqdm(range(epochs)):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model(
                [adj_list, r_support, features, r_feature, label, masks[0]], ) # pass parameters to call function in the model
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # test
    test_loss, test_acc = model(
        [adj_list, r_support, features, r_feature, label, masks[1]],) # pass parameters to call function in the model

    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")