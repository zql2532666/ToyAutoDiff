import numpy as np


class Variable:
    count = 0
    def __init__(self, value, parents=(), op=''):
        self.value = value if isinstance(value, np.ndarray) else np.array(value, dtype=np.float64)
        self.input_size = self.value.shape
        self.grad = np.zeros(self.input_size)
        self.parents = set(parents)
        self.op = op       # parent operation
        self.backward_step = lambda: None
        self.name = 'x_' + str(Variable.count)
        Variable.count += 1
    
    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, name=\"{self.name}\", op=\"{self.op}\", n_parents={len(self.parents)})"

    def __add__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        out = Variable(self.value + other.value, (self, other), op='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out.backward_step = _backward
        return out

    def __radd__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        out = Variable(self.value + other.value, (self, other), op='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out.backward_step = _backward
        return out

    def __mul__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        out = Variable(self.value * other.value, (self, other), op='*')
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out.backward_step = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, number):
        assert isinstance(number, (int, float)), "only supporting int/float powers"
        out = Variable(np.power(self.value, number), (self,), op=f'^{number}')

        def _backward():
            self.grad += out.grad * (number * self.value ** (number - 1))

        out.backward_step = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        out = Variable(self.value - other.value, (self, other), op='-')
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out.backward_step = _backward
        return out

    def __rsub__(self, other):
        return other - self

    def __truediv__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        return self * other ** -1

    def __rtruediv__(self, other):
        assert isinstance(other, (int, float, Variable)), "invalid data type"
        if not isinstance(other, Variable):
            data = np.empty(self.input_size)
            data.fill(other)
            other = Variable(data)
            
        return other * self ** -1

    def sin(self):
        out = Variable(np.sin(self.value), (self,), op='sin')

        def _backward():
            self.grad += out.grad * np.cos(self.value)

        out.backward_step = _backward
        return out

    def cos(self):
        out = Variable(np.cos(self.value), (self,), op='cos')

        def _backward():
            self.grad -= np.sin(self.value) * out.grad

        out.backward_step = _backward
        return out

    def tan(self):
        return self.sin() / self.cos()

    def log(self):
        out = Variable(np.log(self.value), (self,), op='log')

        def _backward():
            self.grad += (1 / self.value) * out.grad

        out.backward_step = _backward
        return out
    
    def exp(self):
        out = Variable(np.exp(self.value), (self,), op='exp')

        def _backward():
            self.grad += out.grad * np.exp(self.value)

        out.backward_step = _backward
        return out

    def relu(self):
        out = Variable(np.maximum(np.zeros(self.input_size), self.value), (self,), op='relu')

        def _backward():
            self.grad += out.grad * (out.value > 0)

        out.backward_step = _backward
        return out
        
    def topsort(self):
        sorted_nodes = []
        visited = set()
        
        def dfs(v):
            if v in visited:
                return
            visited.add(v)
            for nei in v.parents:
                dfs(nei)
            sorted_nodes.append(v)
            
        dfs(self)
        return sorted_nodes

    def backward(self):
        topo = self.topsort()
        self.grad = np.empty(self.input_size)
        self.grad.fill(1.0)

        for node in reversed(topo):
            print(node)
            node.backward_step()

    def build_graph(self):
        edges, nodes = set(), set()

        def dfs(v):
            if v not in nodes:
                nodes.add(v)
                for nei in v.parents:
                    edges.add((nei, v))
                    dfs(nei)
            
        dfs(self)
        return nodes, edges