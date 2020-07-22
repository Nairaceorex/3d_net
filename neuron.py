import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)

    def forward(self, x):
        x = self.fc1(x)
        return x


x = torch.randn([1, 100, 3])
target = torch.randn([1, 100, 64])

model = Net()
criterian = torch.nn.MSELoss()
optimazer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    output = model(x)
    loss = criterian(output, target)
    print("Epoch:", epoch, "Loss:", loss)
    optimazer.zero_grad()
    loss.backward()
    optimazer.step()
