# https://github.com/tonyngjichun/SOLAR/tree/master


import torch.nn as nn
import torch


# MODULES

class SOLARLoss(nn.Module):
    # Loss = L FOS + (lambda * sos_loss)
    # L FOS is Triplet or Contrastive

    def __init__(self, margin = 1.25, eps=1e-6, _lambda = 10, criterion1='triplet'):
        super().__init__()

        # 1st order
        self.margin = margin
        self.eps = eps

        if criterion1 == 'triplet':
            self.L_FOS = TripletLoss(self.margin)
        else:
            self.L_FOS = ContrastiveLoss(self.margin,self.eps)

        # 2nd order
        self._lambda = _lambda
        self.L_SOS = SOSLoss()

    def forward(self, x, label):
        return self.L_FOS(x, label) + self._lambda * self.L_SOS(x, label)



class ContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)
    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7
    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class SOSLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, label):
        return sos_loss(x, label)

class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'



# FUNCTIONAL
def contrastive_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def triplet_loss(x, label, margin=0.1):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1).item() # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0)) / nq

def sos_loss(x, label):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1).item() # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0) # D * (B * num_neg)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_an = torch.sum(torch.pow(xa - xn, 2), dim=0)
    dist_pn = torch.sum(torch.pow(xp - xn, 2), dim=0)

    return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq
