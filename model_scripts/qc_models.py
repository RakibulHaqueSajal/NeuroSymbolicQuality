import torch
from monai.networks import nets
from monai.networks.nets import ResNetBlock
from torch import nn
import timm


def get_inplanes():
    return [64, 128, 256, 512]

class ResNet3D(nets.resnet.ResNet):
    def __init__(self, block, layers, block_inplanes, spatial_dims, n_input_channels):
        super().__init__(block=block, layers=layers, block_inplanes=block_inplanes, \
                         spatial_dims=spatial_dims, n_input_channels = n_input_channels, feed_forward=False, act='prelu')

        # self.layer1 = nn.Identity()
        # self.layer2 = nn.Identity()
        # self.layer3 = nn.Identity()
        # self.layer4 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x) # Shape: (batch_size, 64, 8, 8, 8)
        x = self.layer2(x) # Shape: (batch_size, 128, 1, 1, 1)
        x = self.layer3(x) # Shape: (batch_size, 256, 1, 1, 1)
        x = self.layer4(x) # Shape: (batch_size, 512, 1, 1, 1)

        x = self.avgpool(x) # Shape: (batch_size, 512, 1, 1, 1)

        x = x.view(x.size(0), -1) # Shape: (batch_size, 512)

        return x


class QCModel3D(nn.Module):
    def __init__(
        self,
        encoder_name,
        spatial_dims=None,
        n_input_channels=None,
        num_classes=2,
    ):
        super(QCModel3D, self).__init__()

        self.encoder_3d = None
        in_features = 64
        self.L = 64  # feature space dimension

        if encoder_name == "resnet":
            self.encoder_3d = ResNet3D(
                block=ResNetBlock,
                layers=[1, 1, 1, 1],
                block_inplanes=get_inplanes(),
                spatial_dims=spatial_dims,
                n_input_channels=n_input_channels,
            )  # ResNet34
            in_features = 64
        else:
            msg = "Only ResNet3D is implemented"
            raise NotImplementedError(msg)

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.BatchNorm1d(self.L),
            nn.PReLU(),
        )

        self.classifier = nn.Linear(self.L, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.encoder_3d(x)

        latent_space = self.feature_extractor(x)
        return self.classifier(latent_space)


class Attention(nn.Module):
    """
    Attention module for performing attention mechanism on input data.

    Args:
        L (int): The input size of the attention module.
        D (int): The hidden size of the attention module.
        K (int): The output size of the attention module.

    Methods:
        forward(x, isNorm=True): Performs forward pass of the attention module.

    """

    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),  # matrix V x h
            nn.Tanh(),
            nn.Linear(self.D, self.K)  # matrix W x tanh(V x h)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, isNorm=True):
        """
        Performs forward pass of the attention module.

        Args:
            x (torch.Tensor): The input tensor of shape N x L.
            isNorm (bool): Flag indicating whether to apply softmax normalization (default: True).

        Returns:
            torch.Tensor: The output tensor of shape K x N.

        """
        A = self.attention(x)  # N x K

        if len(A.shape) == 4:
            A = torch.transpose(A, 2, 3)
        elif len(A.shape) == 3:
            A = torch.transpose(A, 1, 2) # (bs, K, N)
        else:
            A = torch.transpose(A, 1, 0) # K x N

        if isNorm:
            if len(A.shape) == 4:
                A = torch.softmax(A, dim=3)
            elif len(A.shape) == 3:
                A = torch.softmax(A, dim=2)  # softmax over N,
            else:
                A = self.softmax(A)  # softmax over N,

        return A  # K x N or (bs, K, N)


class QCModel2D(nn.Module):
    def __init__(self, encoder_name, block=None, layers=None, block_inplanes=None, spatial_dims=None, n_input_channels=None, num_classes=2):
        super(QCModel2D, self).__init__()
        
        self.encoder_2d = None
        self.D = 32 # hidden layer dimension for attention
        self.K = 1 # number of attention heads
        
        if encoder_name == 'resnet':
            self.encoder_2d = timm.create_model('resnet10t', pretrained=True, num_classes=0, in_chans=n_input_channels)

            # Freeze all layers
            for param in self.encoder_2d.parameters():
                param.requires_grad = False

            # Unfreeze only layer4
            for param in self.encoder_2d.layer4.parameters():
                param.requires_grad = True

            # self.encoder_2d = ResNet3D(block=ResNetBlock, 
            #                            layers=[1, 1, 1, 1], 
            #                            block_inplanes=get_inplanes(), 
            #                            spatial_dims=spatial_dims, 
            #                            n_input_channels=n_input_channels) # ResNet34
            in_features = 512
        else:
            raise NotImplementedError('Only ResNet3D is implemented')
        
        self.L = in_features # feature space dimension
        # Feature projection layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.BatchNorm1d(self.L),
            nn.PReLU(),
        )
        
        # Attention mechanism for slice-level aggregation
        self.attention = Attention(L=self.L, D=self.D, K=self.K)
        
        # Classifier on aggregated features
        self.classifier = nn.Linear(self.L * self.K, num_classes-1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.classifier:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)

    def forward(self, x, hog_feats=None, return_attention=False):
        bs, n_slices, ch, h, w = x.shape
        x = x.reshape(bs * n_slices, ch, h, w) # Shape: (bs * n_slices, ch=1, h=512, w=512)
        
        # Extract features from encoder
        x = self.encoder_2d(x)  # Shape: (bs * n_slices, 512)
        
        # Project to lower dimensional feature space
        # x = self.feature_extractor(x)  # Shape: (bs * n_slices, L=128)
        
        # Reshape to separate batch and slices
        feature_space = x.view(bs, n_slices, -1)  # Shape: (bs, n_slices, L=128)
        
        # Apply attention mechanism across slices
        attention_weights = self.attention(feature_space)  # Shape: (bs, K=1, n_slices)
        
        # Weighted aggregation of slice features
        if len(attention_weights.shape) == 3:
            aggregated_features = torch.bmm(attention_weights, feature_space)  # Shape: (bs, K=1, L=128)
        else:
            aggregated_features = torch.mm(attention_weights, feature_space)  # Shape: (K=1, L=128)
        
        # Flatten for classification
        aggregated_features = aggregated_features.view(bs, -1)  # Shape: (bs, L*K=128)
        
        # Classification
        logits = self.classifier(aggregated_features)  # Shape: (bs, num_classes-1)
        
        if return_attention:
            return logits, attention_weights

        #print(logits.shape)
        
        return logits

class QCModel2DWithConcept(nn.Module):
    """
    2D QC model that also predicts three concept scores (each with 5 classes).
    Outputs: concatenation of main classifier logits and three concept logits.
    Final output dim = num_classes + 3*concept_classes.
    """
    def __init__(self, encoder_name, block=None, layers=None, block_inplanes=None,
                 spatial_dims=None, n_input_channels=None, num_classes=4, concept_classes=4):
        super(QCModel2DWithConcept, self).__init__()
        
        self.encoder_2d = None
        self.D = 32  # hidden layer dimension for attention
        self.K = 1   # number of attention heads
        self.concept_classes = concept_classes
        
        if encoder_name == 'resnet':
            self.encoder_2d = timm.create_model('resnet10t', pretrained=True, num_classes=0, in_chans=n_input_channels)

            # Freeze all layers
            for param in self.encoder_2d.parameters():
                param.requires_grad = False

            # Unfreeze only layer4
            for param in self.encoder_2d.layer4.parameters():
                param.requires_grad = True
            # self.encoder_2d = ResNet3D(block=ResNetBlock, 
            #                            layers=[1, 1, 1, 1], 
            #                            block_inplanes=get_inplanes(), 
            #                            spatial_dims=spatial_dims, 
            #                            n_input_channels=n_input_channels)
            in_features = 512
        else:
            raise NotImplementedError('Only ResNet3D is implemented')
        
        self.L = in_features  # feature space dimension
        
        # Feature projection (optional, kept similar to QCModel2D)
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.BatchNorm1d(self.L),
            nn.PReLU(),
        )
        
        # Attention mechanism
        self.attention = Attention(L=self.L, D=self.D, K=self.K)
        
        # Main classifier head (kept as before)
        self.classifier = nn.Linear(self.L * self.K, num_classes-1)
        
        # Concept MLP heads: two-layer MLP with LeakyReLU
        hidden_dim = max(64, (self.L * self.K) // 2)
        self.concept_head_1 = nn.Sequential(
            nn.Linear(self.L * self.K, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, self.concept_classes-1)
        )
        self.concept_head_2 = nn.Sequential(
            nn.Linear(self.L * self.K, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, self.concept_classes-1)
        )
        self.concept_head_3 = nn.Sequential(
            nn.Linear(self.L * self.K, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, self.concept_classes-1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # default: kaiming for all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        # override classifier and concept MLP linears with Xavier (and zero bias)
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_normal_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        # initialize MLP layers
        for seq in (self.concept_head_1, self.concept_head_2, self.concept_head_3):
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, hog_feats=None, return_attention=False):
        bs, n_slices, ch, h, w = x.shape
        x = x.reshape(bs * n_slices, ch, h, w)
        
        # Extract features from encoder
        x = self.encoder_2d(x)  # (bs * n_slices, in_features)
        
        # Project features (optional)
        # x = self.feature_extractor(x)
        
        feature_space = x.view(bs, n_slices, -1)  # (bs, n_slices, L)
        
        attention_weights = self.attention(feature_space)  # (bs, K, n_slices)
        
        # Weighted aggregation
        if len(attention_weights.shape) == 3:
            aggregated = torch.bmm(attention_weights, feature_space)  # (bs, K, L)
        else:
            aggregated = torch.mm(attention_weights, feature_space)  # (K, L)
        
        aggregated = aggregated.view(bs, -1)  # (bs, L*K)
        
        # Heads
        main_logits = self.classifier(aggregated)                     # (bs, num_classes-1)
        c1 = self.concept_head_1(aggregated)                          # (bs, concept_classes)
        c2 = self.concept_head_2(aggregated)                          # (bs, concept_classes)
        c3 = self.concept_head_3(aggregated)                          # (bs, concept_classes)
        
        # Return as before
        if return_attention:
            return main_logits, c1, c2, c3, attention_weights
        
        return main_logits, c1, c2, c3



