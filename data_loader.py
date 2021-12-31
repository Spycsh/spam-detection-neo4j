from neo4j import GraphDatabase


class DataLoader:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def obtain_heterogeneous_graph(self):
        with self.driver.session() as session:
            # user_review_adj, user_item_adj, item_review_adj, item_user_adj, review_item_adj, review_user_adj
            user_review_adj = session.write_transaction(self.get_user_review_adj)
            user_item_adj = session.write_transaction(self.get_user_item_adj)
            item_review_adj = session.write_transaction(self.get_item_review_adj)
            item_user_adj = session.write_transaction(self.get_item_user_adj)
            review_item_adj = session.write_transaction(self.get_review_item_adj)
            review_user_adj = session.write_transaction(self.get_review_user_adj)
            print(user_review_adj)
            print(user_item_adj)
            print(item_review_adj)
            print(item_user_adj)
            print(review_item_adj)
            print(review_user_adj)

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

        return list(d.values())

    @staticmethod
    def _get_adj_by_review(result):
        d = {}
        for pair in result:
            d[pair[0]] = pair[1]

        return list(d.values())


if __name__ == "__main__":
    dataLoader = DataLoader("bolt://localhost:7687", "neo4j", "admin")
    dataLoader.obtain_heterogeneous_graph()
    dataLoader.close()
