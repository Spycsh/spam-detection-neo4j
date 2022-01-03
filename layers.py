import tensorflow as tf

from tensorflow.keras import layers



# equation (3)
class HiddenStateEdge(layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(HiddenStateEdge, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = tf.nn.relu
        self.agg_weights = self.add_weight('agg_weights', [input_dim, output_dim], dtype=tf.float32, trainable=True)


    def call(self, inputs, *args, **kwargs):
        # split adjcent lists and three embedding lists
        adj_lists, embedding_lists = inputs

        # TODO: dropout?
        review_embedding = embedding_lists[0]
        user_embedding = embedding_lists[1]
        item_embedding = embedding_lists[2]


        # TODO: shuffle(transpose?)
        # the user node and the item node that the review links to

        # adj_lists[5]: item_by_review 1 * 7
        # item_embedding: 3 * 7
        # item_by_review: 7 * 7
        item_by_review = tf.nn.embedding_lookup(item_embedding, tf.cast(adj_lists[5], dtype=tf.int32))
        # item_by_review = tf.transpose(tf.random.shuffle(tf.transpose(ri)))
        # item_by_review = tf.transpose(item_by_review)
        print("item_by_review", item_by_review)

        # adj_lists[4] : user_by_review: 1 * 7
        # user_embedding: 5 * 7
        # user_by_review: 7 * 7
        user_by_review = tf.nn.embedding_lookup(user_embedding, tf.cast(adj_lists[4], dtype=tf.int32))
        # user_by_review = tf.transpose(user_by_review)
        print("user_by_review", user_by_review)

        # equation 4
        agg_vector = tf.concat([review_embedding, user_by_review, item_by_review], 1)

        result = self.activation(tf.matmul(agg_vector, self.agg_weights))
        print(result)
        return result


# equation (5) (6) (7) (8)
class HiddenStateUserItem(layers.Layer):
    def __init__(self, input_dim_item, input_dim_user, hidden_dim, output_dim, **kwargs):
        super(HiddenStateUserItem, self).__init__(**kwargs)
        self.input_dim_item = input_dim_item
        self.input_dim_user = input_dim_user
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = tf.nn.relu
        # set trainable weights
        self.user_weights = self.add_weight('user_weights', [input_dim_user, hidden_dim], dtype=tf.float32, trainable=True)
        self.item_weights = self.add_weight('item_weights', [input_dim_item, hidden_dim], dtype=tf.float32, trainable=True)
        self.concat_user_weights = self.add_weight('concat_user_weights', [hidden_dim, output_dim], dtype=tf.float32, trainable=True)
        self.concat_item_weights = self.add_weight('concat_item_weights', [hidden_dim, output_dim], dtype=tf.float32, trainable=True)

    def call(self, inputs):
        adj_lists, embedding_lists = inputs
        # TODO: dropout?
        review_embedding = embedding_lists[0]
        user_embedding = embedding_lists[1]
        item_embedding = embedding_lists[2]

        # the review embeddings which correspond to the user
        review_by_user = tf.nn.embedding_lookup(review_embedding, tf.cast(adj_lists[0], dtype=tf.int32))
        # ur = tf.transpose(tf.random.shuffle(tf.transpose(ur)))

        # the items which the user bought
        item_by_user = tf.nn.embedding_lookup(item_embedding, tf.cast(adj_lists[1], dtype=tf.int32))

        review_by_item = tf.nn.embedding_lookup(review_embedding, tf.cast(adj_lists[2], dtype=tf.int32))

        user_by_item = tf.nn.embedding_lookup(user_embedding, tf.cast(adj_lists[3], dtype=tf.int32))


        # equation (6)
        # TODO: need to print to see the shape
        concat_i_e = tf.concat([item_by_user, review_by_user], axis=2)
        concat_u_e = tf.concat([user_by_item, review_by_item], axis=2)

        s_i_e = tf.shape(concat_i_e)
        s_u_e = tf.shape(concat_u_e)
        concat_i_e = tf.reshape(concat_i_e, [s_i_e[0], s_i_e[1] * s_i_e[2]])
        concat_u_e = tf.reshape(concat_u_e, [s_u_e[0], s_u_e[1] * s_u_e[2]])

        # equation (7) attention
        attention_user = self.attention(user_embedding, user_embedding, concat_u_e)
        attention_item = self.attention(item_embedding, item_embedding, concat_i_e)

        # equation(5)
        agg_user_neigh_embedding = self.activation(tf.matmul(attention_user, self.user_weights))
        agg_item_neigh_embedding = self.activation(tf.matmul(attention_item, self.item_weights,))

        # equation (8)
        # TODO: check formula here
        user_output = tf.concat(tf.matmul(self.concat_user_weights, user_embedding), agg_user_neigh_embedding)
        item_output = tf.concat(tf.matmul(self.concat_item_weights, item_embedding), agg_item_neigh_embedding)

        return user_output, item_output

    # scaled dot product attention
    def attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        k_trans = tf.transpose(k)
        product = tf.matmul(q, k_trans)
        dk = tf.cast(tf.shape(k)[-1], tf.float32) # dimension
        scaled_attention = tf.matmul(tf.nn.softmax(product / tf.math.sqrt(dk), axis=-1), v)
        return scaled_attention


class GraphConvolution(layers.Layer):
    def __init__(self, input_dim, output_dim, dropout=0.5, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        # self.dropout = dropout
        self.activation = tf.nn.relu

        self.weights_ = []

        for i in range(1):
            w = self.add_weight('weight' + str(i), [input_dim, output_dim], dtype=tf.float32)
            self.weights_.append(w)

    def call(self, inputs, training=True):
        # forward propagation
        x, support_ = inputs

        # dropout
        if training is True:
            x = tf.nn.dropout(x, self.dropout)

        # convolve
        supports = list()
        for i in range(len(support_)):
            # has features x
            pre_sup = tf.matmul(x, self.weights_[i])

            support = tf.matmul(support_[i], pre_sup)
            supports.append(support)

        output = tf.add_n(supports)
        axis = list(range(len(output.getshape()) - 1))
        mean, variance = tf.nn.moments(output, axis)
        scale = None
        offset = None
        variance_epsilon = 0.001
        output = tf.nn.batch_normalization(output, mean, variance, offset,
                                           scale, variance_epsilon)

        # no bias here

        # normalization
        return tf.nn.l2_normalize(self.activation(output), axis=None, epsilon=1e-12)


class GASConcatenation(layers.Layer):

    def __init__(self, **kwargs):
        super(GASConcatenation, self).__init__(**kwargs)

    def __call__(self, inputs):
        adj_list, concat_vecs = inputs
        # neighbor sample
        ri = tf.nn.embedding_lookup(concat_vecs[2],
                                    tf.cast(adj_list[5], dtype=tf.int32))

        ru = tf.nn.embedding_lookup(concat_vecs[1],
                                    tf.cast(adj_list[4], dtype=tf.int32))

        concate_vecs = tf.concat([ri, concat_vecs[0], ru, concat_vecs[3]],
                                 axis=1)
        return concate_vecs