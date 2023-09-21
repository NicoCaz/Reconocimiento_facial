import Red
from torchsummary import summary




net = Red.ResNet(block = Red.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

net.eval()

summary(net, (3, 224, 224))
