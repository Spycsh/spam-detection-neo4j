import tensorflow as tf
from tensorflow import keras

from layers import HiddenStateEdge,HiddenStateUserItem,GraphConvolution,GASConcatenation

class GASModel(keras.Model):

    def __init__(self, kwargs):
        super().__init__()

        self.class_size = 2
        # self.reviews_num =
        self.input_dim_r = kwargs.get("input_dim_r")
        self.input_dim_i = kwargs.get("input_dim_i")
        self.input_dim_u = kwargs.get("input_dim_u")
        self.input_dim_i_x = kwargs.get("input_dim_i_x")
        self.input_dim_u_x = kwargs.get("input_dim_u_x")
        self.input_dim_r_gcn = kwargs.get("input_dim_r_gcn")
        self.output_dim1 = kwargs.get("output_dim1", 64)
        self.output_dim2 = kwargs.get("output_dim2", 64)
        self.output_dim3 = kwargs.get("output_dim3", 64)
        self.output_dim4 = kwargs.get("output_dim4", 64)
        self.output_dim5 = kwargs.get("output_dim5", 64)
        self.num_features_nonzero = kwargs.get("num_features_nonzero")
        self.gcn_dim = kwargs.get("gcn_dim", 5)
        self.h_i_size = kwargs.get("h_i_size")
        self.h_u_size = kwargs.get("h_u_size")

        # GAS layers

        self.review_agg_layer = HiddenStateEdge(
            input_dim=self.input_dim_r + self.input_dim_u + self.input_dim_i,
            output_dim=self.output_dim1)

        # item user aggregator
        self.item_user_agg_layer = HiddenStateUserItem(input_dim_user=self.h_u_size,
                                                input_dim_item=self.h_i_size,
                                                output_dim=self.output_dim3,
                                                hidden_dim=self.output_dim2,
                                                input_dim_u_x=self.input_dim_u_x,
                                                input_dim_i_x=self.input_dim_i_x
                                                # concat=True
                                                )

        # review aggregator
        self.r_gcn_layer = GraphConvolution(input_dim=self.input_dim_r_gcn,
                                            output_dim=self.output_dim5,
                                            num_features_nonzero=self.
                                            num_features_nonzero,
                                            # activation=tf.nn.relu,
                                            dropout=0.5,
                                            # is_sparse_inputs=True,
                                            # norm=True
                                            )

        self.concat_layer = GASConcatenation()

        # logistic weights init
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(
            shape=(
                self.output_dim1 + 2 * self.output_dim2 + self.input_dim_i
                + self.input_dim_u + self.output_dim5,
                self.class_size),
            dtype=tf.float32), trainable=True)

    def call(self, inputs, training: bool):
        # main algorithm here
        support, r_support, features, r_feature, label, idx_mask = inputs

        # forward propagation
        z_e = self.review_agg_layer((support, features))
        z_u, z_i = self.item_user_agg_layer((support, features))
        p_e = self.r_gcn_layer((r_feature, r_support), training=True)
        concat_vecs = [z_e, z_u, z_i, p_e]
        gas_out = self.concat_layer((support, concat_vecs))

        # get masked data
        masked_data = tf.gather(gas_out, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        # output layer
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))


        # loss = -tf.reduce_sum(
        #     tf.math.log(tf.nn.sigmoid(masked_label * logits)))
        # acc = self.accuracy(logits, masked_label)

        # calculate loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = bce(masked_label, logits)

        # calculate accuracy
        m = tf.keras.metrics.BinaryAccuracy()
        m.update_state(masked_label, logits)
        acc = m.result().numpy()

        return loss, acc

    # def accuracy(self, preds: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    #     """
    #     Accuracy.
    #     :param preds: the class prediction probabilities of the input data
    #     :param labels: the labels of the input data
    #     """
    #     correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    #     accuracy_all = tf.cast(correct_prediction, tf.float32)
    #     return tf.reduce_sum(accuracy_all) / preds.shape[0]