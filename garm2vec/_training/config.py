from dataclasses import dataclass
from typing import List, Union


@dataclass
class ClassifierMetadata:
    name: str
    nodes: int
    activation: str
    loss: str
    metrics: Union[List[str], str]


outputs = [
    ("season", 5),
    ("gender", 5),
    ("article_type", 142),
    ("usage", 10),
    ("master_category", 7),
]

classifiers = [
    ClassifierMetadata(
        name,
        nodes=nodes,
        activation="softmax",
        loss="categorical_crossentropy",
        metrics="accuracy",
    )
    for name, nodes in outputs
]
