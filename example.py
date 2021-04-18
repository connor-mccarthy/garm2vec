from garm2vec import Garm2Vec

image_path = "/Users/connor/workspace/garm2vec/data/images/15970.jpg"
description = "Turtle Check Men Navy Blue Shirt"

image_path2 = "/Users/connor/workspace/garm2vec/data/images/39386.jpg"
description2 = "Peter England Men Party Blue Jeans"

g2v = Garm2Vec()

vector = g2v.get_one([image_path, description])
print(vector)
vectors = g2v.get_many([[image_path, description], [image_path2, description2]])
print(vectors)

fig1 = g2v.plot_one([image_path, description])
fig2 = g2v.plot_many([[image_path, description], [image_path2, description2]])
