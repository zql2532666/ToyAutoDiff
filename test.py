from autodiff.autodiff import Variable


a = Variable([[2., 3.], [9., 4.]])
b = Variable([[6., 4.], [7., 1.]])

Q = 3*a**3 - b**2
Q.name = 'Q'

print(Q)
Q.backward()