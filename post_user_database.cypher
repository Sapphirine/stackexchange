// add post nodes
USING PERIODIC COMMIT

LOAD CSV WITH HEADERS FROM "file:/Users/Zhen/desktop/Courses/Bigdata/stackexchange/data/post2.csv" AS row FIELDTERMINATOR ';'

CREATE (:post {ID: row.ID, CreationDate:row.CreationDate,  Tags:row.Tags, ViewCount:row.ViewCount, FavoriteCount:row.FavoriteCount, Label:row.LABEL });

// add user nodes
USING PERIODIC COMMIT

LOAD CSV WITH HEADERS FROM "file:/Users/Zhen/desktop/Courses/Bigdata/stackexchange/data/user.csv" AS row FIELDTERMINATOR ';'

CREATE (:user {ID: row.Id, Reputation: row. Reputation, CreationDate:row.CreationDate,  Location:row. Location, UpVotes:row. UpVotes, DownVotes:row. DownVotes, Age:row. Age, Label:row.LABEL });


// add post relation
USING PERIODIC COMMIT

LOAD CSV WITH HEADERS FROM "file:/Users/Zhen/desktop/Courses/Bigdata/stackexchange/data/post_relation.csv" AS row2 FIELDTERMINATOR ';'
MATCH (u:post),(p:post)
where  u.ID=row2.START_ID and p.ID=row2.END_ID
MERGE (u) -[t :Answer]-> (p) ;

// add user post relation
LOAD CSV WITH HEADERS FROM "file:/Users/Zhen/desktop/Courses/Bigdata/stackexchange/data/userPoste.csv" AS row2 FIELDTERMINATOR ';'
MATCH (u:user),(p:post)
where  u.ID=row2.OwnerUserId and p.ID=row2.Id and row2.Type='ask'
MERGE (u) -[t :Ask]-> (p)

LOAD CSV WITH HEADERS FROM "file:/Users/Zhen/desktop/Courses/Bigdata/stackexchange/data/userPoste.csv" AS row2 FIELDTERMINATOR ';'
MATCH (u:user),(p:post)
WHERE u.ID=row2.OwnerUserId and p.ID=row2.Id and row2.Type='answer'
MERGE (u) -[t :Answer]-> (p)