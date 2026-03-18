## PyTorch

In the last few lessons, we learned how to build and optimize neural network architectures.  This gave us a grounding in how data flows between layers, how parameters get adjusted, and how loss decreases.  So far, we've been using NumPy to build and optimize our networks.  In this lesson, we'll learn about PyTorch, a framework that makes building and applying neural networks much simpler.

We'll start off by taking a look at how PyTorch represents data, and we'll move to building a complete neural network in PyTorch.

## Tensors

We'll first load in the same house prices dataset from the last lesson.  Each row in this dataset represents a single house.  The predictor columns are:

- `interest`: The interest rate
- `vacancy`: The vacancy rate
- `cpi`: The consumer price index
- `price`: The price of a house
- `value`: The value of a house
- `adj_price`: The price of a house, adjusted for inflation
- `adj_value`: The value of a house, adjusted for inflation

The predictor columns have all been scaled using the scikit-learn `StandardScaler`.  This gives each column a mean of 0 and a standard deviation of 1.  This makes it easier to activate our nonlinearities.

The target column is `next_quarter`, which is the price of the house in three months.  `next_quarter` has been scaled so the minimum value is `0`, and it has been divided by `1000` and rounded to the nearest integer.  This makes the prediction task simpler for our network.

```python
import sys, os
sys.path.append(os.path.abspath('../data'))
from csv_data import HousePricesDatasetWrapper

# Load in data from csv file
wrapper = HousePricesDatasetWrapper()
train_data, valid_data, test_data = wrapper.get_flat_datasets()
```

The data is currently loaded into NumPy arrays.  We can instead load the data into torch tensors.  Tensors are n-dimensional data structures similar to NumPy arrays.  The primary difference is that torch tensors can be loaded onto different devices, like GPUs.  We'll discuss this more later.

For now, we'll load our training set predictors and targets into torch tensors:

```python
# Import torch.  You can install it with pip install torch.
import torch

# Convert the numpy arrays to torch tensors
train_x = torch.from_numpy(train_data[0])
train_y = torch.from_numpy(train_data[1])
```

```python
train_x
```

    tensor([[ 1.9451,  1.3964, -1.5228,  ..., -0.1168, -0.1389,  0.8226],
            [ 1.9325,  1.3964, -1.4935,  ..., -0.1168, -0.1560,  0.8022],
            [ 1.9955,  1.3964, -1.4935,  ..., -0.1168, -0.0446,  0.8022],
            ...,
            [-0.2595, -0.6860,  0.5061,  ...,  0.3840,  0.4345,  0.3539],
            [-0.2469, -0.6860,  0.5061,  ...,  0.3840,  0.4217,  0.3539],
            [-0.1839, -0.6860,  0.5061,  ...,  0.3840,  0.6257,  0.3539]],
           dtype=torch.float64)

Tensors work very similarly to NumPy arrays.  For example, you can do operations using scalars:

```python
train_x + 1
```

    tensor([[ 2.9451,  2.3964, -0.5228,  ...,  0.8832,  0.8611,  1.8226],
            [ 2.9325,  2.3964, -0.4935,  ...,  0.8832,  0.8440,  1.8022],
            [ 2.9955,  2.3964, -0.4935,  ...,  0.8832,  0.9554,  1.8022],
            ...,
            [ 0.7405,  0.3140,  1.5061,  ...,  1.3840,  1.4345,  1.3539],
            [ 0.7531,  0.3140,  1.5061,  ...,  1.3840,  1.4217,  1.3539],
            [ 0.8161,  0.3140,  1.5061,  ...,  1.3840,  1.6257,  1.3539]],
           dtype=torch.float64)

One important difference is that you want to make sure to use torch functions instead of NumPy methods.  This ensures that the operation is done on the appropriate device.  There are torch equivalents for most NumPy functions:

```python
# Take the square root of each value in the array.  Negative values have an undefined square root.
torch.sqrt(train_x)
```

    tensor([[1.3947, 1.1817,    nan,  ...,    nan,    nan, 0.9070],
            [1.3901, 1.1817,    nan,  ...,    nan,    nan, 0.8957],
            [1.4126, 1.1817,    nan,  ...,    nan,    nan, 0.8957],
            ...,
            [   nan,    nan, 0.7114,  ..., 0.6196, 0.6591, 0.5949],
            [   nan,    nan, 0.7114,  ..., 0.6196, 0.6494, 0.5949],
            [   nan,    nan, 0.7114,  ..., 0.6196, 0.7910, 0.5949]],
           dtype=torch.float64)

## Autograd

One big advantage that Torch has over NumPy for deep learning is autograd.  Autograd will automatically calculate the gradient, without you having to write a backward pass!

To do this, we first need to define that parameter that we want a gradient for, then set `requires_grad` to `True`:

```python
# Define a matrix of weights
# Torch.rand generates random numbers
weights = torch.rand(train_x.shape[1], 1)
# Set requires_grad to True so that autograd can work
weights.requires_grad = True
```

Then we can load in our training data and multiply it by the weights.  You may have noticed above that our `train_x` tensor is in `float64`.  This is because the NumPy arrays were in `float64`.  `float64` means that each number is stored using `64` bits of data.  In PyTorch, the default tends to be `float32`, which uses `32` bits to store each number.

The main difference is the range of possible values that the number can store.

```python
import numpy as np

# Display the maximum value of float64
np.finfo("float64").max
```

    1.7976931348623157e+308

```python
# Display the maximum value of float32
np.finfo("float32").max
```

    3.4028235e+38

`float32` can store large enough numbers that we rarely have issues when training deep learning models.  Thus, it's much more common to work with `float32` in PyTorch.  There are also times when you'll work with `float16` or `int8`, and we'll cover those in a later lesson.

We'll convert our array to `float32`, which is just `torch.float`, since it's the default.  We can then multiply the weights and the `train_x` values:

```python
train_x = train_x.to(torch.float)
predictions = train_x @ weights
```

We can find the gradient by finding the loss (mean squared error derivative), then calling `loss.backward()`.  This will automatically backpropagate from `loss` to `weights`:

```python
loss = (predictions - train_y).mean()
loss.backward()
```

Then we can display the weight gradient:

```python
weights.grad
```

    tensor([[ 0.2605],
            [ 0.4172],
            [-0.5335],
            [-0.5425],
            [-0.5502],
            [-0.5209],
            [-0.5167]])

And make the gradient update with a `1e-5` learning rate:

```python
weights = weights - 1e-5 * weights.grad
```

## nn.Module

We can use the `nn.Module` class to organize our code and make it easier to keep track of parameters.  We can only write the forward pass of the network, and torch will automatically run backpropagation:

```python
from torch import nn
import math

class DenseLayer(nn.Module):
    def __init__(self, input_units, output_units):
        super().__init__()

        # Initialize our weights and biases
        # Scale by k to improve convergence
        k = math.sqrt(1/input_units)
        # Putting a tensor inside nn.Parameter will mark that tensor as needing a gradient
        self.weight = nn.Parameter(torch.rand(input_units, output_units) * 2 * k - k)
        self.bias = nn.Parameter(torch.rand(1, output_units) * 2 * k - k)

    def forward(self, x):
        # Simple forward pass!
        return x @ self.weight + self.bias
```

Above, we write a complete neural network layer.  We initialize the weights and biases, then we write a forward pass.  We use `nn.Parameter` to mark parameters that need gradients.

We can then initialize a multilayer network as another `nn.Module`:

```python
class DenseNetwork(nn.Module):
    def __init__(self, input_units, hidden_units, output_units, layers):
        super().__init__()

        torch.manual_seed(0)
        modules = []
        # Define multiple network layers
        for i in range(layers):
            in_size = out_size = hidden_units
            if i == 0:
                # The first layer has the same number of rows in the weight matrix as columns in the input data
                in_size = input_units
            elif i == layers - 1:
                # The last layer has the same number of columns in the weight matrix as the target
                out_size = output_units
            modules.append(DenseLayer(in_size, out_size))

        # A modulelist holds a list of parameters
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        # Loop through each module and apply it to the data sequentially
        for module in self.module_list:
            x = module(x)
        return x
```

Above, we use `nn.ModuleList` to store a list of parameters/modules.  Most of the work is just setting the correct number of inputs and outputs in the layers.

## DataLoader

Before we can train our network, we have to get our data into the right format for PyTorch.  To do that, we need to setup a `Dataset` and a `DataLoader`.  A `Dataset` is a wrapper around our data.  It behaves like a list, and returns a single training example when indexed:

```python
from torch.utils.data import DataLoader, Dataset

class PriceData(Dataset):
    def __init__(self, x, y):
        # Take in our x and y tensors (predictor, target)
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        # Return how many examples are in the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single training example
        x = self.x[idx]
        y = self.y[idx]
        return x, y

# Initialize the dataset
train_ds = PriceData(train_x, train_y)
```

We then have to wrap the dataset in a `DataLoader`.  A `DataLoader` makes it easy to work with batches of data, or distributed data across multiple devices.  We first set a batch size, and then initialize our `DataLoader` by passing in our `Dataset`.

By default, a `DataLoader` will shuffle the input data every epoch.  We set `shuffle` to `False` since our data is time series, and we want to preserve temporal relationships:

```python
batch_size = 16
train = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
```

## Training Loop

We can now write and run a full training loop using our network and DataLoader:

```python
from statistics import mean

# Define our hyperparameters
epochs = 50
layers = 5
hidden_size = 25
lr = 5e-4

# Initialize our network
net = DenseNetwork(train_x.shape[1], hidden_size, 1, layers)
```

PyTorch makes it easy to create optimizers.  You can define your own, or import premade optimizers.  We'll use the existing torch implementation of `SGD`:

```python
# Create the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```

Whenever we want to update our parameters, we call `optimizer.step()`.  We call `optimizer.zero_grad()` to initialize our gradients to zero, just like we did in the [backpropagation lesson](https://github.com/VikParuchuri/zero_to_gpt/blob/master/explanations/comp_graph.ipynb).

We can now write and run a training loop:

```python
def train_loop(net, optimizer, epochs):
    # Use a predefined loss function
    loss_fn = nn.MSELoss()

    train_losses = []
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train):
            # zero_grad will set all the gradients to zero
            # We need this because gradients will accumulate in the backward pass
            optimizer.zero_grad()
            # Make a prediction using the network
            pred = net(x)
            # Calculate the loss
            loss = loss_fn(pred, y)
            # Call loss.backward to run backpropagation
            loss.backward()
            # Step the optimizer to update the parameters
            optimizer.step()
            train_losses.append(loss.item())

        # Display loss information every few epochs
        if (epoch + 1) % 10 == 0:
            print(mean(train_losses))

train_loop(net, optimizer, epochs)
```

    49.68326367922127
    30.341548889130355
    23.66963948508104
    20.26992240929976
    18.193296501636507
    

## Prebuilt modules

We used prebuilt components for our optimizer and loss functions.  This is another one of the advantages of PyTorch - you don't have to code everything from scratch.  We can also use prebuilt components for our neural network.

Here's an example of swapping our manual `DenseLayer` implementation for `nn.Linear`, which works very similarly:

```python
class DenseNetwork(nn.Module):
    def __init__(self, input_units, hidden_units, output_units, layers):
        super().__init__()

        torch.manual_seed(0)
        modules = []
        for i in range(layers):
            in_size = out_size = hidden_units
            if i == 0:
                in_size = input_units
            elif i == layers - 1:
                out_size = output_units
            # Use nn.Linear instead of our own implementation
            modules.append(nn.Linear(in_size, out_size))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
```

```python
net = DenseNetwork(train_x.shape[1], hidden_size, 1, layers)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train_loop(net, optimizer, epochs)
```

    51.05724552236497
    30.849216412380336
    23.951016113410393
    20.450236316770315
    18.319719779416918

PyTorch makes it easy to swap components in and out to make a more complex network.  You can pick from:

- Layer types
- Complete networks
- Optimizers
- Schedulers
- Loss functions

## Portability

PyTorch also makes your code portable across devices.  So far, we've run on CPU.  Running on the CPU is convenient, but it's also much slower than running on the GPU.  We'll dive into why in a later lesson.

If you want to run on a different device, it's usually hard - you have to use an interface to the specific device.  Luckily, PyTorch makes it simple to swap between devices.  You just use the `.to()` method to send tensors to different devices.  If you call `.to()` on an `nn.Module` instance, all of the parameters used by the model will be sent to the device.

We can first set our device appropriately, depending on our system:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
```

Then, we need to make a small modification to our `Dataset`.  We'll now send our predictors and target to the device.  If you're going to use a device other than the cpu for computation, you need to make sure that all the tensors that you're using (the model, the inputs, etc) are on the same device.  Otherwise, you'll get torch errors.

In this case, we'll send the data to the device when we need it, instead of all upfront.  This saves GPU RAM.

```python
class PriceData(Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # Send x and y to the device
        return x.to(device), y.to(device)
```

We can then initialize our new dataset and train the network.  We'll also need to send the network to the same device that the inputs are on:

```python
train_ds = PriceData(train_x, train_y)
train = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

net = DenseNetwork(train_x.shape[1], hidden_size, 1, layers).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

train_loop(net, optimizer, epochs)
```

    51.057245976105335
    30.84921663403511
    23.95101627384623
    20.450236380659042
    18.319719846621155
    

If you want to pull a value back from the device you sent it to (for example, if you're logging errors), you'll need to use `detach()`, and `cpu()`.  This will remove the tensor from the computational graph, so autograd doesn't try to use it in backpropagation.  It will also pull the tensor back to system RAM so the CPU can access it.

```python
x = torch.rand(5,5).to(device)
y = x.detach().cpu()
y
```

    tensor([[0.3907, 0.2057, 0.6909, 0.6334, 0.6904],
            [0.4445, 0.4336, 0.4603, 0.6318, 0.1163],
            [0.0340, 0.6871, 0.2262, 0.4579, 0.6386],
            [0.5701, 0.8223, 0.5655, 0.6238, 0.4552],
            [0.5738, 0.6833, 0.8411, 0.0262, 0.2917]])

## Disable autograd

There are times when you'll need to disable autograd.  For example, when you're calculating error across a validation set.  In cases like this, you can use the `torch.no_grad` context manager:

```python
x = torch.rand(5,5).to(device)
x.requires_grad = True

with torch.no_grad():
    y = torch.mean(x * 2)

y.backward()
```

    --------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[57], line 7
          4 with torch.no_grad():
          5     y = torch.mean(x * 2)
    ----> 7 y.backward()
    

    File ~/.virtualenvs/nnets/lib/python3.10/site-packages/torch/_tensor.py:488, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        478 if has_torch_function_unary(self):
        479     return handle_torch_function(
        480         Tensor.backward,
        481         (self,),
       (...)
        486         inputs=inputs,
        487     )
    --> 488 torch.autograd.backward(
        489     self, gradient, retain_graph, create_graph, inputs=inputs
        490 )
    

    File ~/.virtualenvs/nnets/lib/python3.10/site-packages/torch/autograd/__init__.py:197, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        192     retain_graph = create_graph
        194 # The reason we repeat same the comment below is that
        195 # some Python versions print out the first line of a multi-line function
        196 # calls in the traceback and some print out the last line
    --> 197 Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        198     tensors, grad_tensors_, retain_graph, create_graph, inputs,
        199     allow_unreachable=True, accumulate_grad=True)
    

    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

As we can see, torch will throw an error about not requiring a gradient.  Disabling the gradient calculation can save a lot of time and memory when doing inference and validation.  We don't need a gradient in these cases, since we're not updating the parameters.

## Wrap-up

In this lesson, we learned about PyTorch, a deep learning framework.  PyTorch has prebuilt components, autograd, and support for different devices.  This makes it much simpler to build neural networks.  We'll learn more about PyTorch in subsequent lessons, but this should cover the main features.

We're very close to implementing a transformer.  In the next lesson, we'll learn how to work with text data, then the lesson after that will be implementing transformers.
