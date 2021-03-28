from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from self_frame import Placeholder
from self_frame import Linear
from self_frame import Sigmoid
from self_frame import MSE
from self_frame import topological_sort_from_disorder_to_order
from self_frame import forward_and_backward
from self_frame import optimizer
import numpy as np

data = load_boston()
X_, y_ = data["data"], data["target"]

X_rm = X_[: ,5]
w1_, b1_ = np.random.normal(), np.random.normal()
w2_ ,b2_ = np.random.normal(), np.random.normal()

X_node, y_node = Placeholder(name="X",is_trainable=False), Placeholder(name="Y",is_trainable=False)
w1_node, b1_node = Placeholder(name="w1",is_trainable=True), Placeholder(name="b1",is_trainable=True)
w2_node ,b2_node = Placeholder(name="w2",is_trainable=True), Placeholder(name="b2",is_trainable=True)

feed_dict = {
    X_node:X_rm,
    y_node:y_,
    w1_node:w1_,
    b1_node:b1_,
    w2_node:w2_,
    b2_node:b2_,
}

output1 = Linear(X_node,w1_node,b1_node,name="linear1")
output2 = Sigmoid(output1,name="sigmoid")
yhat = Linear(output2,w2_node,b2_node,name="linear2")
cost = MSE(y_node,yhat,name="cost")

losses = []
epoch = 10
batch_size=len(X_rm)

for iteration in range(epoch):
    loss = 0

    for batch in range(batch_size):
        index = np.random.choice(range(len(X_rm)))
        X_node.value = X_rm[index]
        y_node.value = y_[index]

        graph_sort = topological_sort_from_disorder_to_order(feed_dict)
        forward_and_backward(graph_sort)
        optimizer(graph_sort,learning_rate=1e-2)
        # print("w1:{} b1:{} w2:{} b2{}".format(w1_node.gradients,b1_node.gradients,w2_node.gradients,b2_node.gradients))
        feed_dict[X_node] = X_node.value
        feed_dict[y_node] = y_node.value
        feed_dict[w1_node] = w1_node.value
        feed_dict[b1_node] = b1_node.value
        feed_dict[w2_node] = w2_node.value
        feed_dict[b2_node] = b2_node.value
        loss +=cost.value
        print(cost.value)


