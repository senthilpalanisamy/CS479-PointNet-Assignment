import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)
        self.linear_transform1 = nn.Sequential(nn.Conv1d(3, 64, 1),
                                               nn.BatchNorm1d(64),
                                               nn.ReLU(64),
                                               nn.Conv1d(64, 64, 1),
                                               nn.BatchNorm1d(64),
                                               nn.ReLU(),
                                              )
        self.linear_transform2 = nn.Sequential(nn.Conv1d(64, 64, 1),
                                               nn.BatchNorm1d(64),
                                               nn.ReLU(),
                                               nn.Conv1d(64, 128, 1),
                                               nn.BatchNorm1d(128),
                                               nn.ReLU(),
                                               nn.Conv1d(128, 1024, 1),
                                               nn.BatchNorm1d(1024),
                                               nn.ReLU()
                                               )


        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - local feature: [B, N, 64]
        """                                                                    
        point_transformation = self.stn3(pointcloud.permute(0,2,1))
        x = torch.matmul(pointcloud, point_transformation.T)
        local_features = self.linear_transform1(x.permute(0,2,1)) # x -> (B, 64, n)
        feature_transformation = self.stn64(x)
        local_features_aligned = torch.matmul(local_features.permute(0,2,1), feature_transform.T) # x -> (B, n, 64)
        x = self.linear_transform2(local_features_aligned.permute(0,2,1)) # x -> (B, 1024, n)
        global_feature = torch.max(x, dim=2)[0] # x -> (B, 1024)
        return global_feature, local_features_aligned


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        self.class_scores = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
                nn.BatchNorm1d(num_classes),
                nn.ReLU()
                )
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        self.features, _  = self.pointnet_feat(pointcloud) # (B, 1024)
        self.class_scores = self.class_scores(self.features) # (B, k)
        return self.class_scores


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()
        self.pointnet_features = PointNetFeat(input_transform = True,
                                              feature_transform = True)
        self.linear_transformation1 = nn.Sequential(
                nn.Conv1d(1088, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, m),
                nn.BatchNorm1d(m),
                nn.ReLU()
                )


    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        B, N, _ = pointcloud.shape
        global_features, local_features = self.pointnet_features(pointcloud) # (B, 1024), (B, N,  64)
        global_features_expanded = global_features_expanded.repeat(1, 1, N) # (B, 1024, N)
        concat_features = torch.concatenate([global_features_expanded, local_features.permute(0,2,1)], dim=1)
        segmentation = self.linear_transformation1(concat_features) # (B, M, N)
        return segmentation # (B, 50, N)



class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()
        self.layer1 = nn.Sequential(
                nn.Linear(1024, num_points // 4),
                nn.BatchNorm1d(num_points // 4),
                nn.ReLU()
                )
        self.layer2 = nn.Sequential(
                nn.Linear(num_points // 4, num_points // 2),
                nn.BatchNorm1d(num_points // 2),
                nn.ReLU()
                )
        self.layer3 = nn.Sequential(
                nn.Linear(num_points // 2, num_points),
                nn.Dropout1d(num_points),
                nn.BatchNorm1d(num_points),
                nn.ReLU()
                )
        self.fc = nn.Linear(num_points, num_points * 3)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        self.global_features, _ = self.pointnet_feat(pointcloud) # (B, 1024)
        x = self.layer1(self.global_features) # (B, n / 2)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x.reshape(-1, self.num_points, 3)


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
