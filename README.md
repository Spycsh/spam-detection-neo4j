# spam-detection-neo4j


## notes

create user-review-item graph
```
CREATE (user0:User{name:'user0',id:0})
CREATE (user1:User{name:'user1',id:1})
CREATE (user2:User{name:'user2',id:2})
CREATE (user3:User{name:'user3',id:3})
CREATE (user4:User{name:'user4',id:4})

CREATE (item0:Item {name:'item0',id:0})
CREATE (item1:Item {name:'item1',id:1})
CREATE (item2:Item {name:'item2',id:2})


CREATE (user0)-[review0:Review{name:'review0',id:0}]->(item0)
CREATE (user0)-[review1:Review{name:'review1',id:1}]->(item1)
CREATE (user1)-[review2:Review{name:'review2',id:2}]->(item0)
CREATE (user2)-[review3:Review{name:'review3',id:3}]->(item0)
CREATE (user3)-[review5:Review{name:'review5',id:5}]->(item2)
CREATE (user4)-[review4:Review{name:'review4',id:4}]->(item1)
CREATE (user4)-[review6:Review{name:'review6',id:6}]->(item2)
```

look up

click the node labels or
```
MATCH (n) RETURN n LIMIT 25
```

```
MATCH (user)-[:Review]->(item)
RETURN user.name, item.name
```

```
MATCH (user)-[review0:Review{name:'review0'}]->(item)
RETURN user.name, item.name
```

delete all
```
match (a) -[r] -> () delete a, r
match (a) delete a
```
