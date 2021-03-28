import numpy as np
import random
class Node:
    def __init__(self,name = None,inputs=[],is_trainable = True):
        self.outputs = []
        self.inputs = inputs
        self.name = name
        self.is_trainable = is_trainable

        for node in self.inputs:
            node.outputs.append(self) ##将该节点作为输入节点的的输出节点

        self.value = None

        self.gradients = {}

    def forward(self):#前向传播
        raise NotImplementedError

    def backward(self):#反向传播
        raise NotImplementedError

    def __repr__(self):
        return self.name

class Placeholder(Node):
    def __init__(self,name=None,is_trainable=True):
        Node.__init__(self,name=name,is_trainable=is_trainable)

    def forward(self,value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self:0}
        for node in self.outputs:
            self.gradients[self] = node.gradients[self] * 1

class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))

class Linear(Node):
    def __init__(self,nodes,weights,bias,name=None,is_trainable=False):
        Node.__init__(self,name=name,inputs=[nodes,weights,bias],is_trainable=is_trainable)

    def forward(self):
        inputs_v = self.inputs[0].value
        weights_v = self.inputs[1].value
        bias_v = self.inputs[2].value
        self.value = np.dot(inputs_v,weights_v) + bias_v

    def backward(self):
        ##对每一个节点初始化偏导
        self.gradients={node:np.zeros_like(node.value) for node in self.inputs}

        for node in self.outputs:
            loss_for_linear_partial = node.gradients[self]

            inputs = self.inputs[0]
            weights = self.inputs[1]
            bias = self.inputs[2]
            #形如y = x*w + b 对x，w， b 求偏导
            # self.gradients[inputs] = loss_for_linear_partial * weights.value
            # self.gradients[weights] = inputs.value * loss_for_linear_partial
            self.gradients[inputs] = np.dot(loss_for_linear_partial,weights.value)
            self.gradients[weights] = np.dot(inputs.value,loss_for_linear_partial)
            self.gradients[bias] = np.sum(loss_for_linear_partial,axis=0,keepdims=False)
            '''
            .T 矩阵转置
            np.sum() axis指定轴方向上元素求和；keepdims 举例：如果一个2*3*4的三维矩阵，axis=0，keepdims默认为False，则结果矩阵被降维至3*4（二维矩阵）；
            如果keepdims=True, 则矩阵维度保持不变，还是三维，只是第零个维度由2变为1，即1*3*4的三维矩阵
            '''
class Sigmoid(Node):
    def __init__(self,node,name=None,is_trainable=False):
        Node.__init__(self,inputs=[node],name=name,is_trainable=is_trainable)

    def sigmoid_(self,x):
        return 1./(1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self.sigmoid_(self.x)

    # def partial(self):
    #     return self.sigmoid_(self.x) * (1 - self.sigmoid_(self.x))

    def backward(self):
        self.partial = self.sigmoid_(self.x) * (1 - self.sigmoid_(self.x))
        # y = 1 / (1 + e^-x)
        # y' = 1 / (1 + e^-x) (1 - 1 / (1 + e^-x))
        self.gradients = {node:np.zeros_like(node.value) for node in self.inputs}

        for node in self.outputs:
            loss_for_sigmoid_partial = node.gradients[self]
            self.gradients[self.inputs[0]] = loss_for_sigmoid_partial * self.partial  ## * keep all dimensions same!

class MSE(Node): ##损失函数
    def __init__(self,y,yhat,name=None,is_trainable=False):
        Node.__init__(self,inputs=[y,yhat],name=name,is_trainable=is_trainable)
    def forward(self):
        y = self.inputs[0].value.reshape(-1,1)
        yhat = self.inputs[1].value.reshape(-1,1)
        assert(y.shape == yhat.shape)
        self.data_nums = self.inputs[0].value.shape[0]
        self.diff = y - yhat
        self.value = np.mean(self.diff**2)


    def backward(self):
        self.gradients[self.inputs[0]] = (2/self.data_nums) * self.diff
        self.gradients[self.inputs[1]] = (-2/self.data_nums) * self.diff



from collections import defaultdict
def feed_dict_to_graph(feed_dict):  ##将数据转化为图的形式存储，方便进行拓扑排序
    computing_graph = defaultdict(list)

    nodes = [n for n in feed_dict]

    while nodes:
        node = nodes.pop(0)

        if isinstance(node, Placeholder):
            node.value = feed_dict[node]

        if node in computing_graph:
            continue

        for n in node.outputs:
            computing_graph[node].append(n)
            nodes.append(n)

    return computing_graph

def topological_sorting(graph):##通过拓扑排序得到有序的节点序列 前向和反向传播

    sorted_nodes = []

    while len(graph) > 0:
        all_inputs_nodes = []
        all_outputs_nodes = []

        for node in graph: ##将输出和输入节点分类别存放
            all_inputs_nodes += graph[node]
            all_outputs_nodes.append(node)

        all_inputs_nodes = set (all_inputs_nodes)
        all_outputs_nodes = set (all_outputs_nodes)


        only_input_nodes = all_outputs_nodes - all_inputs_nodes

        if len(only_input_nodes) > 0:
            node_ = random.choice(list(only_input_nodes))

            sort_first_node = [node_]

            if len(graph)==1:
                sort_first_node += graph[node_]

            graph.pop(node_)

            sorted_nodes += sort_first_node

            for _,links_line in graph.items(): ##删除与只是输入节点相关联的边
                if node_ in links_line:
                    links_line.remove(node_)

        else: ##图是环
            break

    return sorted_nodes

def topological_sort_from_disorder_to_order(feed_dict):
    graph = feed_dict_to_graph(feed_dict)

    return topological_sorting(graph)

def forward_and_backward(nodes): ##定义一次epoch 简化代码
    for node in nodes:
        if isinstance(node,Placeholder):
            node.forward(value=node.value)
        else:
            node.forward()

    for node in nodes[::-1]:
        node.backward()


def optimizer(trainable_nodes,learning_rate = 1e-2):
    for node in trainable_nodes:
        if node.is_trainable:
            node.value = node.value - 1 * learning_rate * np.mean(node.gradients[node])


