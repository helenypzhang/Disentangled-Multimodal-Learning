import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable      

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        """
        Compute the knowledge distillation loss
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            temperature: Temperature for softening probability distributions
        """
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        return distillation_loss
 
class PathBatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(PathBatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, att10, att20):
        # assert omic.size() == vgrid.size()
        # att10.shape [B, H, L1, L2], att20.shape [B, H, L1, L2]
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            att10 = torch.cat(GatherLayer.apply(att10), dim=0)
            att20 = torch.cat(GatherLayer.apply(att20), dim=0)
        # reshape as N*C
        # att10 = att10.view(8, N, -1)
        # att20 = att20.view(8, N, -1)
        att10 = att10.view(N, 8, -1).transpose(0,1) #[8,N,-1]
        att20 = att20.view(N, 8, -1).transpose(0,1) #[8,N,-1]

        # form N*N similarity matrix
        att10_similarities = []
        for item in att10:
            att10_similarity = item.mm(item.t())
            att10_norm = torch.norm(att10_similarity, 2, 1).view(-1, 1)
            att10_similarity = att10_similarity / att10_norm
            att10_similarities.append(att10_similarity)
        mean_att10_sim = torch.mean(torch.stack(att10_similarities), dim=0)

        att20_similarities = []
        for item in att20:
            att20_similarity = item.mm(item.t())
            att20_norm = torch.norm(att20_similarity, 2, 1).view(-1, 1)
            att20_similarity = att20_similarity / att20_norm
            att20_similarities.append(att20_similarity)
        mean_att20_sim = torch.mean(torch.stack(att20_similarities), dim=0)

        batch_loss = (mean_att10_sim - mean_att20_sim) ** 2 / N
        
        return batch_loss            


def low_rank_loss(tensor):
    # Perform Singular Value Decomposition
    # u, s, v are the singular value decomposition results
    u, s, v = torch.svd(tensor)
    # The loss is the sum of all singular values except the largest one
    # We want to minimize this sum to push the rank towards 1
    loss = torch.sum(s[1:])
    return loss

# # Example usage
# B, L = 5, 3  # Example dimensions
# output = torch.rand(B, L)  # Random tensor for illustration
# loss = low_rank_loss(output)
# print("Low-rank loss:", loss.item())

def diag_variance_loss(x, weight=1.0):
    diag_elements = x.diagonal()  # 取对角线元素
    diag_var = torch.var(diag_elements)  # 计算方差
    return weight * diag_var  # 作为正则项加总到损失

# # example
# loss = main_loss + diag_variance_loss(your_tensor, weight=0.1)

class OmicDomainScaleLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(OmicDomainScaleLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, att1_tea10, att1_tea20, att2_tea10, att2_tea20):
        # assert omic.size() == vgrid.size()
        # omic.shape [B, 128], vgrid.shape[8B, 2, 12, 12]
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            att1_tea10 = torch.cat(GatherLayer.apply(att1_tea10), dim=0)
            att1_tea20 = torch.cat(GatherLayer.apply(att1_tea20), dim=0)
            att2_tea10 = torch.cat(GatherLayer.apply(att2_tea10), dim=0)
            att2_tea20 = torch.cat(GatherLayer.apply(att2_tea20), dim=0)

        # reshape as N*C
        att1_tea10 = att1_tea10.view(N, -1) #[B,8,2500,144]-->[N,-1]
        att1_tea20 = att1_tea20.view(N, -1) #[B,8,2500,144]-->[N,-1]

        att2_tea10 = att2_tea10.view(N, -1)
        att2_tea20 = att2_tea20.view(N, -1)
        '''
        # form N*N similarity matrix
        similarity = activations.mm(activations.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.mm(ema_activations.t())
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        batch_loss = (similarity - ema_similarity) ** 2 / N
        '''

        # form N*N similarity matrix for att1
        # cal->similarity_att1(att1_tea10, att1_tea20)->diag(att1)equal elements
        similarity_att1 = att1_tea10.mm(att1_tea20.t())
        norm_att1 = torch.norm(similarity_att1, 2, 1).view(-1, 1)
        similarity_att1 = similarity_att1 / norm_att1

        # form N*N similarity matrix for att2
        # cal->similarity_att2(att2_tea10, att2_tea20)->diag(att2)equal elements
        similarity_att2 = att2_tea10.mm(att2_tea20.t())
        norm_att2 = torch.norm(similarity_att2, 2, 1).view(-1, 1)
        similarity_att2 = similarity_att2 / norm_att2

        # diag_variance_loss(your_tensor, weight=0.1)
        loss1 = diag_variance_loss(similarity_att1, weight=10000)
        loss2 = diag_variance_loss(similarity_att2, weight=10000)
        loss = loss1 + loss2
        
        return loss



def directional_consistency_loss(M, eps=1e-6):
    """
    Args:
        M: tensor of shape [2, N]
        eps: small value to avoid numerical issues when checking equality
    Returns:
        loss: scalar tensor using MSE
    """
    # Get differences between first and second row
    differences = M[0] - M[1]  # shape: [N]
    
    # Create mask for non-equal elements (handling numerical precision)
    nonzero_mask = torch.abs(differences) > eps
    
    # Count number of non-equal elements
    N_nonzero = torch.sum(nonzero_mask)
    
    # Convert differences to signs (+1 for positive, -1 for negative)
    signs = torch.sign(differences)  # shape: [N]
    
    # Calculate normalized value
    if N_nonzero > 0:
        print('n_nonzero > 0')
        x_normalized = torch.sum(signs) / N_nonzero
    else:
        print('n_nonzero = 0:', N_nonzero)
        x_normalized = torch.tensor(0.0, device=M.device)
    
    # Calculate MSE between |x_normalized| and 1
    # loss = torch.nn.functional.mse_loss(torch.abs(x_normalized), torch.tensor(1.0))
    loss = (torch.abs(x_normalized) - 1.0) ** 2
    # Or alternatively: loss = (torch.abs(x_normalized) - 1.0) ** 2
    
    return loss

class OmicDomainScaleLoss_wrong(nn.Module):
    def __init__(self, batch_size, world_size):
        super(OmicDomainScaleLoss_wrong, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, att1_tea10, att1_tea20, att2_tea10, att2_tea20):
        # assert omic.size() == vgrid.size()
        # omic.shape [B, 128], vgrid.shape[8B, 2, 12, 12]
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            att1_tea10 = torch.cat(GatherLayer.apply(att1_tea10), dim=0)
            att1_tea20 = torch.cat(GatherLayer.apply(att1_tea20), dim=0)
            att2_tea10 = torch.cat(GatherLayer.apply(att2_tea10), dim=0)
            att2_tea20 = torch.cat(GatherLayer.apply(att2_tea20), dim=0)

        print('att1_tea10.sum:', att1_tea10.sum()) #[B,8,2500,144]-->[N,8,-1]->[N]
        print('att1_tea20.sum:', att1_tea20.sum()) #[B,8,2500,144]-->[N,8,-1]->[N]
        print('att2_tea10.sum:', att2_tea10.sum()) #[B,8,2500,144]-->[N,8,-1]->[N]
        print('att2_tea20.sum:', att2_tea20.sum()) #[B,8,2500,144]-->[N,8,-1]->[N]

        # reshape as N*C
        avg_att1_tea10 = att1_tea10.view(N, 8, -1).mean(dim=(1,2)) #[B,8,2500,144]-->[N,8,-1]->[N]
        avg_att1_tea20 = att1_tea20.view(N, 8, -1).mean(dim=(1,2)) #[B,8,2500,144]-->
        avg_att2_tea10 = att2_tea10.view(N, 8, -1).mean(dim=(1,2))
        avg_att2_tea20 = att2_tea20.view(N, 8, -1).mean(dim=(1,2))

        att1_tea = torch.cat((avg_att1_tea10.unsqueeze(1), avg_att1_tea20.unsqueeze(1)), dim=1).t() #[N]-->[N,1]-->[N,2]-->[2,N]
        att2_tea = torch.cat((avg_att2_tea10.unsqueeze(1), avg_att2_tea20.unsqueeze(1)), dim=1).t() #[N]-->[N,1]-->[N,2]-->[2,N]

        loss1 = directional_consistency_loss(att1_tea)
        loss2 = directional_consistency_loss(att2_tea)
        loss = loss1 + loss2
        
        return loss


class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, omic, vgrid):
        # assert omic.size() == vgrid.size()
        # omic.shape [B, 128], vgrid.shape[8B, 2, 12, 12]
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            omic = torch.cat(GatherLayer.apply(omic), dim=0)
            vgrid = torch.cat(GatherLayer.apply(vgrid), dim=0)
        # reshape as N*C
        omic = omic.view(N, -1)
        vgrid = vgrid.view(8, N, -1)

        # form N*N similarity matrix
        similarity = omic.mm(omic.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        vgrid_similarities = []
        for item in vgrid:
            vgrid_similarity = item.mm(item.t())
            vgrid_norm = torch.norm(vgrid_similarity, 2, 1).view(-1, 1)
            vgrid_similarity = vgrid_similarity / vgrid_norm
            vgrid_similarities.append(vgrid_similarity)
        mean_vgrid_sim = torch.mean(torch.stack(vgrid_similarities), dim=0)

        batch_loss = (similarity - mean_vgrid_sim) ** 2 / N
        
        return batch_loss
