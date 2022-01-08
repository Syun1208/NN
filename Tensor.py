import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype= torch.float32, device=device, requires_grad=True)
print(my_tensor.device)
print(my_tensor.dtype)
print(my_tensor.requires_grad)
print(my_tensor.shape)

#Other common initialization methods
x = torch.empty(size = (3,3))
x = torch.zeros(3,3)
x = torch.ones(3,3)
x = torch.eye(3,3)
x = torch.rand(4,5)
x = torch.arange(start=1, end=10, step=1)
x = torch.linspace(start=1, end=15, steps=20)
x = torch.empty(size=(1,4)).normal_(mean = 0, std = 1)
x = torch.empty(size=(1,4)).uniform_(0, 1)
x = torch.diag(torch.ones(1, 4))
print(x)

#How to initialize and convert tensors to other types (int, float, double)

tensor = torch.eye(4, 4)
print(tensor.bool()) #Boolean TRUE/FALSE
print(tensor.short()) #int16
print(tensor.long()) #int64(important)
print(tensor.int()) #int32
print(tensor.float()) #float32(important)
print(tensor.double()) #float64
print(tensor.half()) #float16

#Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5), dtype=int)
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)

#Tensor Math & Comparision Operations
x = torch.tensor([5, 2, 4])
y= torch.tensor([6, 8, 2])

#Addition
z1 = torch.empty(3)
print(torch.add(x,y,out=z1))
z2 = x+y
print(z2)

#Substraction
z3 = torch.subtract(x,y,out=z1)
#z3 = x - y
print(z3)

#Division
z4 = torch.true_divide(x, y)
print(z4)

#Inplace operation
t = torch.zeros(3)
t.add_(x)
t += x

#Exponentiation
z5 = torch.pow(x, 2)
z6 = x**2

# Simple comparison
z7 = x < 4
z8 = x > 4
print(z7)
print(z8)

#Matrix Multiplication
x1 = torch.rand((2, 6))
x2 = torch.rand((6, 8))
x3 = torch.mm(x1, x2)
print(x3)

#Matrix Exponentiation
matrix_exp = torch.rand(8,8)
print(matrix_exp.matrix_power(3))
