import torch

from torch_geometric.nn import EdgeConv
import torch_cluster

class ØBlock(torch.nn.Module):
    def __init__(self,l1,l2,l3,k):
        super(ØBlock, self).__init__()
        
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(l2,l3),
                                            torch.nn.LeakyReLU())
        self.conv_max = EdgeConv(self.nn_conv1,aggr = 'max')
        self.conv_mean = EdgeConv(self.nn_conv1,aggr = 'mean')
        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')
        self.relu = torch.nn.LeakyReLU()
        self.knn_graph = torch_cluster.knn_graph
        self.k = k
    def forward(self, x,edge_index,batch):
        pos = x[:,0:3]
        edge_index = self.knn_graph(x=pos,k=self.k,batch=batch)
        x_max = self.conv_max(x,edge_index)
        x_mean = self.conv_mean(x,edge_index)
        x_add = self.conv_add(x,edge_index)

        x = self.relu(torch.cat((x_max,x_mean,x_add),dim=1))
        return x,edge_index