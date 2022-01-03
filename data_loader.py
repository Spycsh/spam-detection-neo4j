from neo4j import GraphDatabase
import numpy as np

from sklearn.model_selection import train_test_split



class DataLoader:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def obtain_graph(self):
        with self.driver.session() as session:
            # user_review_adj, user_item_adj, item_review_adj, item_user_adj, review_item_adj, review_user_adj
            user_review_adj = session.write_transaction(self.get_user_review_adj)
            user_item_adj = session.write_transaction(self.get_user_item_adj)
            item_review_adj = session.write_transaction(self.get_item_review_adj)
            item_user_adj = session.write_transaction(self.get_item_user_adj)
            review_item_adj = session.write_transaction(self.get_review_item_adj)
            review_user_adj = session.write_transaction(self.get_review_user_adj)


            # padding of the adjacent matrix
            user_review_adj_padding = self.pad_adj_list(user_review_adj)
            user_item_adj_padding = self.pad_adj_list(user_item_adj)
            item_review_adj_padding = self.pad_adj_list(item_review_adj)
            item_user_adj_padding = self.pad_adj_list(item_user_adj)

            print("user review adj")
            print(user_review_adj_padding)
            print("user item adj")
            print(user_item_adj_padding)
            print("item review adj")
            print(item_review_adj_padding)
            print("item user adj")
            print(item_user_adj_padding)
            print("review item adj")
            print(review_item_adj)
            print("review user adj")
            print(review_user_adj)

            # initialize review_vecs
            review_vecs = np.array([[1, 0, 0, 1, 0],
                                    [1, 0, 0, 1, 1],
                                    [1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1],
                                    [1, 1, 0, 1, 1]])

            # initialize user_vecs and item_vecs with user_review_adj and
            # item_review_adj
            # for example, u0 has r1 and r0, then we get the first line of user_vecs:
            # [1, 1, 0, 0, 0, 0, 0]
            # user_vecs = np.array([[1, 1, 0, 0, 0, 0, 0],
            #                       [0, 0, 1, 0, 0, 0, 0],
            #                       [0, 0, 0, 1, 0, 0, 0],
            #                       [0, 0, 0, 0, 0, 1, 0],
            #                       [0, 0, 0, 0, 1, 0, 1]])
            # item_vecs = np.array([[1, 0, 1, 1, 0, 0, 0],
            #                       [0, 1, 0, 0, 1, 0, 0],
            #                       [0, 0, 0, 0, 0, 1, 1]])
            user_vecs = np.zeros((len(user_review_adj), len(review_vecs)))
            for i, x in enumerate(user_review_adj):
                for y in x:
                    user_vecs[i][y] = 1
            print("user vectors:")
            print(user_vecs)

            # user_review_adj
            item_vecs = np.zeros((len(item_review_adj), len(review_vecs)))
            for i, x in enumerate(item_review_adj):
                for y in x:
                    item_vecs[i][y] = 1
            print("item vectors:")
            print(item_vecs)

            features = [review_vecs, user_vecs, item_vecs]

            # initialize the Comment Graph
            # use word2vec or sentence2vec to generate
            # A Simple but Tough-to-Beat
            # Baseline for Sentence Embeddings. (2017)
            homo_adj = [[1, 0, 0, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 1, 0],
                        [1, 0, 1, 0, 0, 1, 0],
                        [0, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0]]


            adjs = [user_review_adj, user_item_adj, item_review_adj, item_user_adj,
                    review_user_adj, review_item_adj, homo_adj]

            # assign spam or not
            y = np.array(
                [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]
            )
            index = range(len(y))

            X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y,
                                                                test_size=0.4,
                                                                random_state=48,
                                                                shuffle=True)
            split_ids = [X_train, X_test]
            return adjs, features, split_ids, y


    def get_user_review_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN user.id, review.id "
                        "ORDER BY user.id")

        return self._get_adj(result)

    def get_user_item_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN user.id, item.id "
                        "ORDER BY user.id")

        return self._get_adj(result)

    def get_item_review_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN item.id, review.id "
                        "ORDER BY item.id")
        return self._get_adj(result)

    def get_item_user_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN item.id, user.id "
                        "ORDER BY item.id")
        return self._get_adj(result)

    def get_review_item_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN review.id, item.id "
                        "ORDER BY review.id")

        return self._get_adj_by_review(result)

    def get_review_user_adj(self, tx):
        result = tx.run("MATCH (user)-[review:Review]->(item) "
                        "RETURN review.id, user.id "
                        "ORDER BY review.id")

        return self._get_adj_by_review(result)

    @staticmethod
    def _get_adj(result):
        d = {}
        for pair in result:
            if pair[0] in d:
                d[pair[0]].append(pair[1])
            else:
                d[pair[0]] = [pair[1]]

        adj = [sorted(i) for i in d.values()]

        return adj

    @staticmethod
    def _get_adj_by_review(result):
        d = {}
        for pair in result:
            d[pair[0]] = pair[1]

        return list(d.values())

    @staticmethod
    def pad_adj_list(x_data):
        # Get lengths of each row of data
        lens = np.array([len(x_data[i]) for i in range(len(x_data))])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        padded = np.zeros(mask.shape)
        for i in range(mask.shape[0]):
            padded[i] = np.random.choice(x_data[i], mask.shape[1])
        padded[mask] = np.hstack((x_data[:]))
        return padded

if __name__ == "__main__":
    dataLoader = DataLoader("bolt://localhost:7687", "neo4j", "admin")
    dataLoader.obtain_graph()
    dataLoader.close()
