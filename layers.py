import tensorflow as tf

from tensorflow.keras import layers


# equation (3)
class AggregatorSubLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, activation, **kwargs):
        super(AggregatorSubLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.agg_weights = self.add_weight('agg_weights', [input_dim, output_dim], dtype=tf.float32, trainable=True)


    def call(self, inputs, *args, **kwargs):
        # split adjcent lists and three embedding lists
        adj_lists, embedding_lists = inputs

        # TODO: dropout?
        review_embedding = embedding_lists[0]
        item_embedding = embedding_lists[1]
        user_embedding = embedding_lists[2]


        # TODO: shuffle(transpose?)
        # the user node and the item node that the review links to
        # adj_lists[5]: review_item 1 * 7
        # item_embedding: 3 * 7
        # review_item: 7 * 7
        review_item = tf.nn.embedding_lookup(item_embedding, tf.cast(adj_lists[5], dtype=tf.int32))
        # review_item = tf.transpose(tf.random.shuffle(tf.transpose(ri)))
        # review_item = tf.transpose(review_item)
        print("review_item", review_item)

        # adj_lists[4] : review_user: 1 * 7
        # user_embedding: 5 * 7
        # review_user: 7 * 7
        review_user = tf.nn.embedding_lookup(user_embedding, tf.cast(adj_lists[4], dtype=tf.int32))
        # review_user = tf.transpose(review_user)
        print("review_user", review_user)

        # equation 4
        agg_vector = tf.concat([review_embedding, review_user, review_item], 1)

        result = self.activation(tf.matmul(agg_vector, self.agg_weights))
        print(result)
        return result


