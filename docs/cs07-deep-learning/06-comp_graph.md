# Backpropagation in depth

In the [last lesson](http://127.0.0.1:8000/deeplearning/05-rnn/), we learned how to create a recurrent neural network.  We now know how to build several network architectures using components like dense layers, softmax, and recurrent layers.

We've been a bit loose with how we cover backpropagation, so that neural network architecture is easier to understand.  Backpropagation is how a neural network calculates how much to change each parameter in the network (the gradient).  Understanding how it works is important for tuning networks for performance, and writing fused kernels for GPUs.

In this lesson, we'll do a deep dive into how backpropagation works.  We'll do this by building a computational graph to keep track of which changes we make to input data.

A computational graph looks like this:

![comp_graph](images/comp_graph/comp_graph.png)

It shows all the individual operations we performed (like multiplication) to modify the value of `X`, in order.  Keeping track of a computational graph is how we know how to reverse our operations to do backpropagation.

This is the same way frameworks like PyTorch do forward and backward passes.  To create the computational graph, we'll make a miniature version of PyTorch!

## The Softmax function

We'll first build a computational graph of the softmax function, then backpropagate through that graph to get the gradient against our inputs.  We introduced the softmax function in a previous lesson.  It's used to convert the output of a neural network into probabilities that can be used as predictions.  The softmax function is defined as:

$$\zeta=\frac{e^{\hat{y_{i}}}}{\sum_{j=0}e^{\hat{y_{j}}}}$$

For each row of our neural network output, we raise $e$ to the power of our output value, then divide by the sum of $e$ raised to the power of each of the outputs for that row.

The softmax function looks like this in code:

```python
import numpy as np

def softmax_func(normalized):
    raised = np.exp(normalized)
    output = raised / np.sum(raised, axis=1).reshape(-1,1)
    return output
```

We can test the softmax function using some fake data that we generate:

```python
# 5 rows and 3 columns of random numbers
x = np.random.rand(5, 3)

# Generate random correct labels for later
# Exactly one label per row will be correct
y = np.zeros_like(x)
inds = (np.arange(0,y.shape[0]), np.random.randint(0, 3, size=y.shape[0]))
y[inds] = 1
```

`x` is our input to the softmax function.  It has `3` columns.  `y` is our target, where each row is a one-hot encoded vector.  The `1` will correspond to the correct label for each row.

We can then apply the softmax function:

```python
normalized = x - np.max(x, axis=-1).reshape(-1,1)
softmax_func(normalized)
```

    array([[0.38110313, 0.34380373, 0.27509314],
           [0.39553464, 0.41072091, 0.19374445],
           [0.2744683 , 0.39579312, 0.32973858],
           [0.37172812, 0.35272599, 0.27554589],
           [0.46719982, 0.32717758, 0.20562259]])

In the above code, we subtract the maximum from each element in the row before passing the data into the softmax function.  This prevents numerical underflow or overflow.  Each [numeric type](https://numpy.org/doc/stable/user/basics.types.html) (float, integer, etc) can only hold a certain number of digits.  For example, floating point 16 can store 5 exponent bits, and ten digit bits (each bit is only base 2, so this is less than the same number of base-10 digits).  The maximum value we can store in `float16` is `65500`:

```python
# Check the maximum value we can assign to float16
np.finfo('float16').max
```

    65500.0

```python
# This is an example of numeric overflow, where we store more digits than float16 can hold
a = np.array([0], dtype=np.float16)
a[0] = 6.55e5
```

    /var/folders/xz/9z84c__j28g8tg28bmcthjj00000gn/T/ipykernel_61885/2419725567.py:3: RuntimeWarning: overflow encountered in cast
      a[0] = 6.55e5
    

When we raise $e$ to a very large or small number, we can generate a number that is too large to store in our specific data type.  Subtracting the max gives us the same end result, but reduces the risk of overflow.  Feel free to try the softmax out with and without subtracting the max to see how it works!

## Staged Softmax

Instead of computing the softmax derivative, we previously used the fact that the derivative of the softmax and negative log likelihood functions "cancel out", and end up with a derivative of $p-y$.  But what if we want to find the derivative ourselves?

We can approach it analytically, and find the derivative of the entire function.  We can even use SymPy to help us do the derivation, like we did in an earlier lesson.  Another method is to break the softmax function apart into individual operations.  Each operation will make a single modification to the data:

![softmax_steps.svg](images/comp_graph/softmax_steps.svg)

We perform 3 operations on the data:

- Exp - we raise e to the power x.
- Sum - we add up the $e^x$ values for each row.
- Divide - we divide the $e^x$ values by the sums.

Note that the output of `Exp` is passed to both the `Sum` and `Divide` operations.

By breaking up the softmax this way, we can take the derivative of each individual piece instead of the whole function at once.  By the [chain rule](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/a/chain-rule-review), multiplying the derivative of each individual operation will result in the derivative of the whole function.  We used the chain rule in previous lessons to find the partial derivative of the loss with respect to the model weights and biases.

Now we can build the forward pass of our staged softmax.  The derivative of multiplication is easier to calculate than division, so we'll swap some of our operations to remove the division.

Luckily for us, raising a value `x` to the power `-1` is the same as taking `1/x`.  So instead of dividing `Exp/Sum`, we can do `Exp * Sum ^ -1`, leaving us with these operations:

- Exp
- Sum
- Pow - we invert the sum by raising to the power `-1`
- Multiply - we multiply the inverted sum and the exp values

```python
raised = np.exp(x) # step 1
summed = np.sum(raised, axis=-1).reshape(-1,1) # step 2.  reshape so each row has 1 column.
pow = summed ** -1 # step 3
staged_softmax = raised * pow # step 4
```

```python
staged_softmax
```

    array([[0.38110313, 0.34380373, 0.27509314],
           [0.39553464, 0.41072091, 0.19374445],
           [0.2744683 , 0.39579312, 0.32973858],
           [0.37172812, 0.35272599, 0.27554589],
           [0.46719982, 0.32717758, 0.20562259]])

Our staged softmax has the exact same output as our original function.

## Softmax Derivative

To get the derivative of the softmax, we need to reverse the operations we did before:

![softmax_steps_full_bwd.svg](images/comp_graph/softmax_steps_full_bwd.svg)

To compute the full derivative, here are the steps we need to follow:

1. Start at the final step.  Take in the loss gradient as the input, and multiply by the derivative of the final step.
2. Pass the gradient to the previous operation.
3. Continue calculating the derivative of each operation, and multiplying by the gradient.  Note that `Exp` is input to two operations, so it will sum both gradients.
4. Continue until we reach the first operation.

To calculate loss, we'll use negative log likelihood, which is $NLL = - \sum_{i=0} y_{i} \log p_{i}$.  Since $y$ is only non-zero at one position per row, this will only have a single nonzero term ($-y_{i} * \log p_{i}$ where $i$ is the correct label where $y$ equals `1`).

We can solve this derivative by breaking it into steps:

* $\log p_{i}$ - the derivative of a natural log (base e) is $\frac{1}{p_{i}}$
* $-y_{i} * \log p_{i}$ - the derivative wrt p is $-y_{i}$

So the derivative of NLL is $\frac{1}{p} * -y_{i} = \frac{-y_{i}}{p_{i}}$

We'll use the negative log likelihood derivative $\frac{\partial L}{\partial p}$ below:

```python
def nll_grad(y, pred):
    return -1 * y / pred

loss_grad = nll_grad(y, staged_softmax)
```

We can then calculate the softmax derivative by multiplying the derivatives of the individual operations.  Remember that a derivative is the rate of change of a function as we change the input.

- Exp - the derivative of $e^x$ is $e^x$ (this is a very cool property of $e$!)
- Sum - since a sum operation will combine input elements into one, we just distribute the gradient over all input elements.  A change to any of the input elements will have a direct impact on the output.
- Pow - the derivative of $x^{-1}$ is $-1 * x^{-2}$.  More on [this here](https://www.khanacademy.org/math/old-ap-calculus-ab/ab-derivative-rules/ab-diff-negative-fraction-powers/a/power-rule-review).
- Multiply - the derivatives of $x*y$ are $y$ wrt $x$ and $x$ wrt $y$.  This is because any change to $x$ is multiplied by $y$, and vice versa.  Thus the rate of change of $x$ is $y$, and vice versa.

We can now create the backward pass of our staged softmax.  The backward pass will start with the loss gradient.  This will be a matrix showing how much we need to adjust each of the output values from our softmax to reduce our loss.  We can then compute gradients for each operation, ending with the gradient against the input, `x`.  If `x` was the output of a neural network, we would continue backpropagation at that point to adjust the network parameters.

We'll name each gradient according to the step it is a gradient for, not the step it is coming from.  So `raised_grad` is the gradient on `raised`.

```python
# Step 4 derivative
raised_grad = loss_grad * pow
pow_grad = loss_grad * raised
pow_grad = np.sum(pow_grad, axis=-1).reshape(-1,1) # reshape gradient to match input data

# Step 3
summed_grad = (-1 * summed ** -2) * pow_grad

# Step 2
raised_grad_2 = np.ones_like(raised) * summed_grad # distribute gradient across inputs

# Step 1
raised_grad += raised_grad_2 # sum incoming gradients
staged_softmax_grad = raised_grad * np.exp(x)
```

We did two things above that might be confusing.  The first is that we summed 2 gradients on raised.  This is because raised connects to 2 operations, and both have separate gradients.  Whenever this happens, we sum the gradients.

The second is that we reshaped `pow_grad` to have a single column.  This is to match `pow`, which only had `1` column in the forward pass.  Whenever a gradient doesn't match the shape of the input data, we change the size of the gradient to match it.  This is because the gradient represents the partial derivative against the input data to the operation.

We can compare our result to the derivative of the softmax equation to make sure everything worked.  The derivative of the softmax is $S_{i}((i==j) - S_{j})$.  We take each element of a single row in the output of a softmax, like this:

    [0.28, 0.25, 0.47]

We then compare each element against each other element.  So we could start at element `0` (`.28`), and compare it to itself.  Then `i` is `0` and `j` is `0`.  So the equation is `.28 * (1 - .28)`.  When we then keep `i` the same, but move `j` to `1`.  The equation becomes `.28 * (0 - .25)`.  And so on, until we construct a matrix like this:

![softmax_deriv.svg](images/comp_graph/softmax_deriv.svg)

We then sum across the rows and multiply by the incoming gradient to get the partial derivative against the inputs.  We can define the softmax derivative in code:

```python
def softmax_grad_func(softmax, loss_grad):
    output = np.zeros_like(softmax)
    for i in range(softmax.shape[0]):
        sm_row = softmax[i,:]
        sm_grad = (-np.outer(sm_row, sm_row) + np.diag(sm_row.flatten()))
        row_grad = sm_grad * loss_grad[i,:].reshape(1,-1)
        output[i,:] = np.sum(row_grad, -1)
    return output
```

We then compare our derivative by stages with the derivation.  The `np.allclose` function tells us if all the values in an array are close to another array.  We use this instead of `==` because there are small numerical differences in similar computations with numpy.

```python
derived_softmax_grad = softmax_grad_func(staged_softmax, loss_grad)
np.allclose(derived_softmax_grad, staged_softmax_grad)
```

    True

We can also compare the gradient we computed of the loss and softmax together with the derivative of both together, which is `p-y`:

```python
np.allclose(staged_softmax_grad, staged_softmax - y)
```

    True

## Operations

Breaking the softmax into stages helped us understand the basic units of a computational graph.  But what if we don't want to have to type out all the code for the forward and backward pass every time?  It would be nice if we could only define the forward pass, and automatically have the backward pass happen.  This is how it works in frameworks like PyTorch.

We can build our own version of PyTorch by individually defining each operation, then mixing and matching the operations to create a more complex equation.  By doing this, we can create a computational graph like this one:

![comp_graph.png](images/comp_graph/comp_graph.png)

Each node in the graph will be a separate class that knows how to do a forward and backward pass.  So we can just execute the graph to run forward and backward passes.

We can start out by defining the operations.  I've written a class called `Node`, which we can subclass to define each operation.  You can look at the code for `Node` if you want.  It gives us some nice methods for running the operations in a graph in order, both forward and backward:

- `apply_fwd` - runs the forward pass up to the node it is called on.
- `apply_bwd` - runs the backward pass from the node it is called on backwards.
- `zero_grad` - zero out our gradient before running a backward pass.
- `generate_graph` - helps us visualize the computational graph.
- `generate_derivative_chains` - shows us the equation for calculating the partial derivative at a node.

For each operation, we just implement the `forward` and `backward` methods, which take in input data, and pass them through the operation.  The `Node` class takes care of the rest.

We'll use the same formulas for each operation that we used in our staged softmax earlier:

```python
import sys, os
sys.path.append(os.path.abspath("../nnets"))
from graph import Node, Parameter, display_chain

class Exp(Node):
    def forward(self, x):
        return np.exp(x) # raise e to the power x

    def backward(self, grad):
        x = self.cache[0] # Pull the x value used in the forward pass
        return np.exp(x) * grad # multiply the incoming gradient by the derivative

class Sum(Node):
    def forward(self, x):
        return np.sum(x, axis=-1).reshape(-1,1)

    def backward(self, grad):
        x = self.cache[0] # Pull the x value used in the forward pass
        return np.ones_like(x) * grad # distribute the gradient over the input data shape

class Pow(Node):
    def forward(self, x, exponent):
        return x ** exponent

    def backward(self, grad):
        x, exponent = self.cache # Pull the x and exponent values used in the forward pass
        return grad * exponent * x ** (exponent - 1), 1

class Multiply(Node):
    def forward(self, x, y):
        return x * y

    def backward(self, grad):
        x, y = self.cache # Pull the x and y values used in the forward pass
        return grad * y, grad * x
```

As you can see above, we've defined the 4 operations that we need for our softmax in code.

We can now define our whole softmax operation as a computational graph.  When we initialize a `Node`, we pass in the nodes that feed into it.  If we're using the node to feed in data (like inputs), we use a special `Parameter` node:

```python
# The input data for our network.  We set needs_grad=True so the gradient is calculated for this parameter in the backward pass.
# Desc is a short description of what the data in this node is.
X = Parameter(x, desc="X", needs_grad=True)

# Raise e to the power x.  Out is a description of the output of the node.
raised = Exp(X, out="e^X")
# Sum the raised values.
summed = Sum(raised, out="sum(e^X)")

# Define -1 as a parameter, so we can use it as an exponent.
negative_one = Parameter(-1, desc="-1", needs_grad=False)
# Invert our sums
inverted = Pow(summed, negative_one, out="1 / sum(e^X)")
# Multiply the inverted sums by e^X
softmax = Multiply(raised, inverted, out="softmax(X)")
```

We can now use the `generate_graph` method on the `softmax` node to visualize the computational graph.

```python
softmax.generate_graph()
```

![svg](images/comp_graph/comp_graph_32_0.svg)

And use the `apply_fwd` method to calculate the softmax.  We can then verify that it is the same as our `staged_softmax` that we calculated earlier:

```python
operations_softmax = softmax.apply_fwd()

np.allclose(staged_softmax, operations_softmax)
```

    True

Now, for the cool part.  We can use the same computational graph to do the backward pass!  Let's first visualize what the backward pass looks like:

```python
softmax.generate_graph(backward=True)
```

![svg](images/comp_graph/comp_graph_36_0.svg)

We can then run `apply_bwd` to run the backward pass.  We first call `zero_grad` to ensure that the initial gradients on all nodes are properly set to zero.

Any parameter nodes with `needs_grad` set to `True` will now have a property called `grad`, where we can get the gradient.  We can get the gradient from `X`, and verify that it matches our staged gradient.

```python
softmax.zero_grad()
softmax.apply_bwd(loss_grad)


operations_softmax_grad = X.grad
np.allclose(staged_softmax_grad, operations_softmax_grad)
```

    True

As a bonus, we can also check the equations that we multiplied to calculate the partial derivative with respect to `X`.  You can see that we had to add together 2 different gradients, just like we did with the staged version.  The $\partial$ symbol means partial derivative, and $\frac{\partial e^{X}}{\partial X}$ means "the partial derivative of $e^X$ with respect to $X$.  The way to interpret this partial derivative is "as $X$ changes, how does $e^{X}$ change?".  By multiplying the partial derivatives of each operation, we can get the larger partial derivative, which is $\frac{\partial L}{\partial X}$ - how the loss changes as we change $X$.

```python
softmax.generate_derivative_chains()
display_chain(X.display_partial_derivative())
```

$\frac{\partial}{\partial X} = \frac{\partial softmax(X)}{\partial e^X}*\frac{\partial e^X}{\partial X} + \\\frac{\partial softmax(X)}{\partial 1 / sum(e^X)}*\frac{\partial 1 / sum(e^X)}{\partial sum(e^X)}*\frac{\partial sum(e^X)}{\partial e^X}*\frac{\partial e^X}{\partial X}$

We just built a computational graph of the softmax function, then ran a forward and backward pass.  This is exactly how deep learning frameworks like PyTorch and TensorFlow work.  They define common operators, like `torch.dot`, keep track of what operators you called, build a graph, and automatically run the backward pass based on the derivative of each operator.  You could say we built a miniature deep learning framework!

## 2-layer network

Let's extend our framework to work across a 2-layer neural network.  We need to define 3 additional operations:

- `MatMul` - to multiply two matrices.  The forward pass is `x@w`.
- `Add` - add two values up.
- `Relu` - a nonlinear activation function.  Anything below `0` will be set to `0`.

Here are the operations:

```python
class MatMul(Node):
    def forward(self, x, w):
        return x @ w # multiply the two matrices.

    def backward(self, grad):
        x, w = self.cache
        # return the input gradient times the weights as the gradient on x
        # the input x values (from the forward pass) times the input gradient is the gradient on the weights
        return grad @ w.T, x.T @ grad

class Add(Node):
    def forward(self, x, b):
        return x + b

    def backward(self, grad):
        # Any change to x or b will scale the output the same amount
        return grad, grad

class Relu(Node):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad):
        x = self.cache[0]
        new_grad = np.array(grad)
        # The derivative of relu is 0 when the input
        # in the forward pass was below 0
        # 1 otherwise
        new_grad[x < 0] = 0
        return new_grad
```

When you train a neural network using a framework like PyTorch, you will usually be using the GPU.  GPUs enable us to parallelize operations and train neural networks much faster than we could with a CPU.

One way to speed up GPU code is to fuse operators.  This means that we combine operators that run together into a single operator.  This combined operator can then be optimized into a kernel that runs better on the GPU.  We'll discuss this more in a future lesson.

As an example of the process, below you can see the fusion of the `Softmax` operation.  The fused softmax used the softmax functions we defined before to do the forward and backward passes.

We also fuse an entire neural network layer into the `Dense` operator.

```python
class Softmax(Node):
    def forward(self, x):
        return softmax_func(x)

    def backward(self, grad):
        x = self.cache[0]
        softmax = self.forward(x)
        return softmax_grad_func(softmax, grad)

class Dense(Node):
    def forward(self, x, w, b):
        # Multiply by weight, add bias
        return x @ w + b

    def backward(self, grad):
        x, w, b = self.cache
        # Return 3 gradients for x, w, and b
        return grad @ w.T, x.T @ grad, grad
```

We now have the operators we need to define our neural network.  Let's first load in the data.  We'll use the same telescope data from an earlier lesson.  We have observations from a telescope, and we want to classify whether each observation is a star, galaxy, or quasar.

We use a data wrapper that I wrote to load and split the data automatically.  We'll only use the training set in this lesson, but you can experiment with the other 2 sets as well.

```python
sys.path.append(os.path.abspath("../data"))
from csv_data import SkyServerDatasetWrapper

# Load the data
wrapper = SkyServerDatasetWrapper()
[train_x, train_y], [valid_x, valid_y], [test_x, test_y] = wrapper.get_flat_datasets()
```

```python
train_x[:2]
```

    array([[ 0.16745842, -0.58492272,  1.03148637, -0.34855938, -0.83728027,
            -0.94605772, -0.99534154, -0.83806089,  0.21085172, -0.21763043,
            -0.36973112,  1.03148936,  1.30931064],
           [ 0.16886159, -0.58311429,  0.05243046, -0.16653251, -0.15415531,
            -0.08264457, -0.02604308, -0.83806089,  0.21085172, -0.21763043,
            -0.36984929, -0.63621258, -0.87919741]])

We then need to one-hot encode our data so that we can use it for classification.  We create 3-element vectors where only one element is `1`, and the others are `0`.  The position of the `1` corresponds to the target we want to predict.

```python
def encode(target, max_value=3):
    # A matrix with 3 columns
    encoded = np.zeros((target.shape[0], max_value))
    # Setup the indices that we'll set to one
    inds = (np.arange(0,target.shape[0]), target.reshape(-1))
    # Set the target positions to 1
    encoded[inds] = 1
    return encoded

train_y = encode(train_y)
```

```python
train_y[:5]
```

    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])

Now that we have our training data, we can initialize our weights and biases.  We're taking in `13` features, so we'll setup our weights for the first layer accordingly.  We want to output `3` digits, one for each encoding position.

```python
# Set a seed so we can reproduce results
np.random.seed(0)
w1 = np.random.rand(13, 10)
b1 = np.random.rand(1, 10)
w2 = np.random.rand(10, 3)
b2 = np.random.rand(1, 3)
```

We can now define our network in code!  This is very similar to networks we've built in the past.  We have our first layer, with a relu activation, then our second layer, then a softmax to get probabilities:

```python
X = Parameter(train_x, desc="X", needs_grad=False)
Y = Parameter(train_y, desc="y", needs_grad=False)

w1_param = Parameter(w1, desc="W1")
b1_param = Parameter(b1, desc="b1")

matmul1 = MatMul(X, w1_param, out="X @ W1")
add1 = Add(matmul1, b1_param, out="Z1")

layer1 = Relu(add1, out="A1")

w2_param = Parameter(w2, desc="W2")
b2_param = Parameter(b2, desc="b2")
matmul2 = MatMul(layer1, w2_param, out="Z1 @ W2")
add2 = Add(matmul2, b2_param, out="Z2")

softmax = Softmax(add2, out="softmax(Z2)")
```

Just like before, we can create a computation graph.  This one is more complex than before, but should still be readable!

```python
softmax.generate_graph()
```

![svg](images/comp_graph/comp_graph_57_0.svg)

We can create predictions using `apply_fwd`:

```python
predictions = softmax.apply_fwd()

predictions[:2]
```

    array([[0.51408579, 0.29557176, 0.19034245],
           [0.42508635, 0.26655849, 0.30835516]])

Now, we can run the backward pass of the network.  We'll first graph it out:

```python
softmax.generate_graph(backward=True)
```

![svg](images/comp_graph/comp_graph_61_0.svg)

Then we'll run our backward pass.  We first calculate our loss gradient, then pass it into our backward pass as the incoming gradient:

```python
softmax.zero_grad() # zero out the gradients
loss_grad = nll_grad(train_y, predictions) # compute the loss gradient
softmax.apply_bwd(loss_grad) # pass the loss gradient into the softmax, and run the backward pass
softmax.generate_derivative_chains() # generate our derivative equations
```

Each of our parameters should now have a gradient.  We'll only show `w2`, but you can look at the other parameters as well:

```python
w2_param.grad[:5]
```

    array([[ 1078.86623385,  1452.92238635, -2531.7886202 ],
           [  685.34169939,  1438.06439063, -2123.40609002],
           [   11.39607931,  3488.94754329, -3500.3436226 ],
           [ 1990.1147882 ,   471.22083925, -2461.33562744],
           [ -538.48614737,  1989.95610171, -1451.46995434]])

We can also look at the partial derivative chains to see how we calculate the derivative with respect to a specific parameter:

```python
display_chain(w2_param.display_partial_derivative())
```

$\frac{\partial}{\partial W2} = \frac{\partial softmax(Z2)}{\partial Z2}*\frac{\partial Z2}{\partial Z1 @ W2}*\frac{\partial Z1 @ W2}{\partial W2}$

```python
display_chain(w1_param.display_partial_derivative())
```

$\frac{\partial}{\partial W1} = \frac{\partial softmax(Z2)}{\partial Z2}*\frac{\partial Z2}{\partial Z1 @ W2}*\frac{\partial Z1 @ W2}{\partial A1}*\frac{\partial A1}{\partial Z1}*\frac{\partial Z1}{\partial X @ W1}*\frac{\partial X @ W1}{\partial W1}$

We just built a computational graph, and used it to do the full forward and backward pass for a neural network!  If you want, you can extend this to update the parameters and train the network.  You would just need to set a learning rate, then subtract the gradient from each parameter.  You would have to set a batch size, and iterate through the data as well.

This has hopefully given you a good look at how backpropagation, works, and how we compute the partial derivatives of each operation, then multiply them out.

Let's do a quick verification to make sure that we did everything correctly.  We can implement the network forward and backward pass like we did in an earlier lesson:

```python
# Forward pass
l1 = train_x @ w1 + b1
l1_activated = np.maximum(l1, 0)
l2 = l1_activated @ w2 + b2
probs = softmax_func(l2)

# Loss
loss_grad = nll_grad(train_y, probs)

# L2 gradients
sm_grad = softmax_grad_func(probs, loss_grad)
l2_w_grad = l1_activated.T @ sm_grad
l2_b_grad = sm_grad.sum(axis=0)

# L1 gradients
l1_grad = sm_grad @ w2.T
l1_grad[l1 < 0] = 0
l1_w_grad = train_x.T @ l1_grad
l1_b_grad = l1_grad.sum(axis=0)
```

Then we can verify that our computational graph matches the manual results:

```python
np.allclose(l1_w_grad, w1_param.grad)
```

    True

## Wrap-up

We did a lot in this lesson!  We learned how to break apart a derivative into steps, then compute each step separately.  Then, we constructed a computational graph and ran the forward and backward passes.

I recommend doing some experimentation with the graph, and making sure you really understand how everything is working.  In the [next lesson](http://127.0.0.1:8000/deeplearning/07-optimizers/), we'll use PyTorch to automatically construct the graph for us.
