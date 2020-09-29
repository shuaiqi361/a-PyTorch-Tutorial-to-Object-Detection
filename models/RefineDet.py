import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from dataset.transforms import *
from Loss import SmoothL1Loss
from utils import find_jaccard_overlap


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps
    Feel free to substitute with other pre-trained backbones
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 512, 512)
        out = F.relu(self.conv1_2(out))  # (N, 64, 512, 512)
        out = self.pool1(out)  # (N, 64, 256, 256)

        out = F.relu(self.conv2_1(out))  # (N, 128, 256, 256)
        out = F.relu(self.conv2_2(out))  # (N, 128, 256, 256)
        out = self.pool2(out)  # (N, 128, 128, 128)

        out = F.relu(self.conv3_1(out))  # (N, 256, 128, 128)
        out = F.relu(self.conv3_2(out))  # (N, 256, 128, 128)
        out = F.relu(self.conv3_3(out))  # (N, 256, 128, 128)
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))  # (N, 512, 64, 64)
        out = F.relu(self.conv4_2(out))  # (N, 512, 64, 64)
        out = F.relu(self.conv4_3(out))  # (N, 512, 64, 64)
        conv4_3_feats = out
        out = self.pool4(out)  # (N, 512, 32, 32)

        out = F.relu(self.conv5_1(out))  # (N, 512, 32, 32)
        out = F.relu(self.conv5_2(out))  # (N, 512, 32, 32)
        out = F.relu(self.conv5_3(out))  # (N, 512, 32, 32)
        out = self.pool5(out)  # (N, 512, 32, 132), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 32, 32)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 32, 32)

        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)

        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        self.load_state_dict(state_dict)

        print("Loading pre-trained VGG16 base model.")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 16 x 16

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8 x 8

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight.data)
                if c.bias is not None:
                    nn.init.constant_(c.bias.data, 0)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 32, 32)
        :return: higher-level feature maps conv8_2, conv9_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 16, 16)
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))  # (N, 128, 8, 8)
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out

        return conv8_2_feats, conv9_2_feats


class TCBConvolutions(nn.Module):
    """
    Anchor Refinement Module:
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, internal_channels=256):
        """
        :param n_classes: number of different types of objects
        """
        super(TCBConvolutions, self).__init__()

        self.feat_channels = {'conv4_3': 512,
                              'conv7': 1024,
                              'conv8_2': 512,
                              'conv9_2': 256}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.tcb_conv4_3 = TCB(self.feat_channels['conv4_3'], self.feat_channels['conv7'], internal_channels)
        self.tcb_conv7 = TCB(self.feat_channels['conv7'], self.feat_channels['conv8_2'], internal_channels)
        self.tcb_conv8_2 = TCB(self.feat_channels['conv8_2'], self.feat_channels['conv9_2'], internal_channels)
        self.tcb_conv9_2 = TCBTail(self.feat_channels['conv9_2'], internal_channels)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats):
        """
        Forward propagation. To ge initial offsets w.r.t. anchors anc binary labels

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 64, 64)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 32, 32)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 16, 16)
        :param conv9_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 8, 8)
        :return: 16320 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # batch_size = conv4_3_feats.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.tcb_conv4_3(conv4_3_feats, conv7_feats)  # (N, 256, 64, 64)
        l_conv7 = self.tcb_conv7(conv7_feats, conv8_2_feats)  # (N, 256, 32, 32)
        l_conv8_2 = self.tcb_conv8_2(conv8_2_feats, conv9_2_feats)  # (N, 256, 16, 16)
        l_conv9_2 = self.tcb_conv9_2(conv9_2_feats)  # (N, 256, 8, 8)

        return l_conv4_3, l_conv7, l_conv8_2, l_conv9_2


class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    To link between the ARM and ODM, we introduce the TCBs to convert features of different layers from the ARM,
    into the form required by the ODM, so that the ODM can share features from the ARM.
    """

    def __init__(self, lateral_channels, channels, internal_channels=256, is_batchnorm=False):
        """
        :param lateral_channels: forward feature channels
        :param channels: pyramidal feature channels
        :param internal_channels: internal conv channels fix to 256
        :param is_batchnorm: adding batch norm
        """
        super(TCB, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.use_bias = not self.is_batchnorm
        self.out_channels = internal_channels

        self.conv1 = nn.Conv2d(lateral_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv3 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)

        self.deconv = nn.ConvTranspose2d(channels, internal_channels, kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=self.use_bias)

        self.relu = nn.ReLU(inplace=True)

        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.deconv_bn = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, lateral, higher_level):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            out = self.relu(self.bn2(self.conv2(lateral_out)) + self.deconv_bn(self.deconv(higher_level)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            lateral_out = self.relu(self.conv1(lateral))
            out = self.relu(self.conv2(lateral_out) + self.deconv(higher_level))
            out = self.relu(self.conv3(out))

        return out


class TCBTail(nn.Module):
    """
    Transfer Connection Block Architecture
    To link between the ARM and ODM, we introduce the TCBs to convert features of different layers from the ARM,
    into the form required by the ODM, so that the ODM can share features from the ARM.
    """

    def __init__(self, lateral_channels, internal_channels=256, is_batchnorm=False):
        """
        :param lateral_channels: forward feature channels
        :param channels: pyramidal feature channels
        :param internal_channels: internal conv channels fix to 256
        :param is_batchnorm: adding batch norm
        """
        super(TCBTail, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.use_bias = not self.is_batchnorm
        self.out_channels = internal_channels

        self.conv1 = nn.Conv2d(lateral_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)
        self.conv3 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=self.use_bias)

        self.relu = nn.ReLU(inplace=True)

        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, lateral):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            out = self.relu(self.bn2(self.conv2(lateral_out)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            lateral_out = self.relu(self.conv1(lateral))
            out = self.relu(self.conv2(lateral_out))
            out = self.relu(self.conv3(out))

        return out


class ARMConvolutions(nn.Module):
    """
    Anchor Refinement Module:
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self):
        """
        :param n_classes: number of different types of objects
        """
        super(ARMConvolutions, self).__init__()

        self.n_classes = 2  # foreground and background
        self.feat_channels = {'conv4_3': 512,
                              'conv7': 1024,
                              'conv8_2': 512,
                              'conv9_2': 256}

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 3,
                   'conv7': 3,
                   'conv8_2': 3,
                   'conv9_2': 3}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(self.feat_channels['conv4_3'], n_boxes['conv4_3'] * 4,
                                     kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(self.feat_channels['conv7'], n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(self.feat_channels['conv8_2'], n_boxes['conv8_2'] * 4,
                                     kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(self.feat_channels['conv9_2'], n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(self.feat_channels['conv4_3'], n_boxes['conv4_3'] * self.n_classes,
                                    kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(self.feat_channels['conv7'], n_boxes['conv7'] * self.n_classes,
                                  kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(self.feat_channels['conv8_2'], n_boxes['conv8_2'] * self.n_classes,
                                    kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(self.feat_channels['conv9_2'], n_boxes['conv9_2'] * self.n_classes,
                                    kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight.data)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats):
        """
        Forward propagation. To ge initial offsets w.r.t. anchors anc binary labels

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 64, 64)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 32, 32)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 16, 16)
        :param conv9_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 8, 8)
        :return: 16320 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        # A total of 16320 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2], dim=1).contiguous()
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2], dim=1).contiguous()

        return locs, classes_scores


class ODMConvolutions(nn.Module):
    """
    Object Detection Module.
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, n_classes, internal_channels=256):
        """
        :param n_classes: number of different types of objects
        """
        super(ODMConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 3,
                   'conv7': 3,
                   'conv8_2': 3,
                   'conv9_2': 3}

        self.feat_channels = {'conv4_3': 512,
                              'conv7': 1024,
                              'conv8_2': 512,
                              'conv9_2': 256}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(internal_channels, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(internal_channels, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(internal_channels, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(internal_channels, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(internal_channels, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(internal_channels, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(internal_channels, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(internal_channels, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight.data)
                if c.bias is not None:
                    nn.init.constant_(c.bias.data, 0)

    def forward(self, conv4_3_tcb, conv7_tcb, conv8_2_tcb, conv9_2_tcb):
        """
        Forward propagation.
        input from TCB modules
        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 64, 64)
        :param conv5_3_feats: conv5_3 feature map, a tensor of dimensions (N, 512, 32, 32)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 16, 16)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 8, 8)

        :return: 16320 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_tcb.size(0)

        # # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_tcb)  # (N, 16, 64, 64)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_tcb)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_tcb)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_tcb)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        # # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_tcb)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)

        c_conv7 = self.cl_conv7(conv7_tcb)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_tcb)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_tcb)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        # A total of 16320 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2], dim=1).contiguous()
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2], dim=1).contiguous()

        return locs, classes_scores


class RefineDet512(nn.Module):
    """
    The RefineDet512 network - encapsulates the base VGG network, auxiliary, ARM and ODM
    """

    def __init__(self, n_classes, config):
        super(RefineDet512, self).__init__()
        self.device = config.device
        self.n_classes = n_classes
        self.base = VGGBase()
        self.theta = 0.01
        # self.disable_parameter_requires_grad(self.base)
        self.aux_convs = AuxiliaryConvolutions()
        self.arm_convs = ARMConvolutions()
        self.odm_convs = ODMConvolutions(self.n_classes)
        self.tcb_convs = TCBConvolutions()

        self.rescale_factors_conv4_3 = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors_conv4_3, 20.)

        self.rescale_factors_conv7 = nn.Parameter(torch.FloatTensor(1, 1024, 1, 1))
        nn.init.constant_(self.rescale_factors_conv7, 10.)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 512, 512)
        :return: 22536 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 64, 64), (N, 1024, 32, 32)

        # # Rescale conv4_3 after L2 norm
        norm4 = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 64, 64)
        conv4_3_feats = conv4_3_feats / norm4  # (N, 512, 64, 64)
        conv4_3_feats = conv4_3_feats * self.rescale_factors_conv4_3  # (N, 512, 64, 64)

        # # Rescale conv7 after L2 norm
        norm7 = conv7_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 64, 64)
        conv7_feats = conv7_feats / norm7  # (N, 1024, 64, 64)
        conv7_feats = conv7_feats * self.rescale_factors_conv7  # (N, 1024, 64, 64)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats = \
            self.aux_convs(conv7_feats)

        arm_locs, arm_scores = self.arm_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats)
        tcb_conv4_3, tcb_conv7, tcb_conv8_2, tcb_conv9_2 = \
            self.tcb_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        odm_locs, odm_scores = self.odm_convs(tcb_conv4_3, tcb_conv7, tcb_conv8_2, tcb_conv9_2)

        raw_locs = self.offset2bbox(arm_locs.data.detach(), odm_locs.data.detach())

        prior_positive_idx = (arm_scores[:, :, 1] > self.theta)  # (batchsize, n_priors)

        return arm_locs, arm_scores, odm_locs, odm_scores, raw_locs, odm_scores, prior_positive_idx

    def offset2bbox(self, arm_locs, odm_locs):
        batch_size = arm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)

        for i in range(batch_size):
            init_bbox_cxcy = gcxgcy_to_cxcy(arm_locs[i], self.priors_cxcy)
            true_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(odm_locs[i], init_bbox_cxcy))

        return true_locs

    def create_prior_boxes(self):

        fmap_dims = {'conv4_3': 64,
                     'conv7': 32,
                     'conv8_2': 16,
                     'conv9_2': 8}

        obj_scales = {'conv4_3': 0.0625,
                      'conv7': 0.125,
                      'conv8_2': 0.25,
                      'conv9_2': 0.5}
        scale_factor = [1.]
        # scale_factor = [2. ** 0, 2. ** (1 / 3.), 2. ** (2 / 3.)]
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 0.5],
                         'conv8_2': [1., 2., 0.5],
                         'conv9_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        for fac in scale_factor:
                            prior_boxes.append([cx, cy, obj_scales[fmap] * fac * sqrt(ratio),
                                                obj_scales[fmap] * fac / sqrt(ratio)])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device).contiguous()
        prior_boxes.clamp_(0, 1)

        return prior_boxes


class RefineDetLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, threshold=0.5, neg_pos_ratio=3, theta=0.01):
        super(RefineDetLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes
        self.config = config
        self.theta = theta

        self.arm_loss = SmoothL1Loss(reduction='mean')
        self.odm_loss = SmoothL1Loss(reduction='mean')

        self.arm_cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.odm_cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def compute_arm_loss(self, arm_locs, arm_scores, boxes, labels):
        """
        :param arm_locs: offset prediction from Anchor Refinement Modules
        :param arm_scores: binary classification scores from Anchor Refinement Modules
        :param boxes: gt bbox
        :param labels: gt labels
        :return:
        """
        batch_size = arm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = arm_scores.size(2)  # should be 2

        true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # initial overlap

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            prior_for_each_object = prior_for_each_object[overlap_for_each_object > 0]

            if len(prior_for_each_object) > 0:
                overlap_for_each_prior.index_fill_(0, prior_for_each_object, 1.0)

            for j in range(prior_for_each_object.size(0)):
                object_for_each_prior[prior_for_each_object[j]] = j

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            # Converted labels to 0, 1's
            label_for_each_prior = (label_for_each_prior > 0).long()
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        # Identify priors that are positive (non-background, binary)
        positive_priors = true_classes > 0
        n_positives = positive_priors.sum(dim=1)  # (N)
        # LOCALIZATION LOSS
        loc_loss = self.arm_loss(arm_locs[positive_priors].view(-1, 4),
                                 true_locs_encoded[positive_priors].view(-1, 4))

        # CONFIDENCE LOSS
        # Number of positive and hard-negative priors per image
        n_hard_negatives = self.neg_pos_ratio * n_positives

        # First, find the loss for all priors
        conf_loss_all = self.arm_cross_entropy(arm_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, -1)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=-1,
                                              descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
            self.device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss

    def compute_odm_loss(self, arm_locs, arm_scores, odm_locs, odm_scores, boxes, labels):
        """
        :param arm_locs: serve as "anchor boxes"
        :param arm_scores:
        :param odm_locs:
        :param odm_scores:
        :param boxes:
        :param labels:
        :return:
        """
        # print(arm_scores.size(), arm_locs.size(), odm_scores.size(), odm_locs.size())
        batch_size = odm_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = odm_scores.size(2)

        # print(n_priors, predicted_locs.size(), predicted_scores.size())
        assert n_priors == odm_locs.size(1) == odm_scores.size(1)

        # Calculate ARM loss: offset smoothl1 + binary classification loss
        decoded_arm_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        # decoded_odm_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_locs_encoded = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            decoded_arm_locs[i] = cxcy_to_xy(gcxgcy_to_cxcy(arm_locs[i], self.priors_cxcy))
            overlap = find_jaccard_overlap(boxes[i], decoded_arm_locs[i])

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (22536)

            overlap_for_each_object, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            prior_for_each_object = prior_for_each_object[overlap_for_each_object > 0]
            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            if len(prior_for_each_object) > 0:
                overlap_for_each_prior.index_fill_(0, prior_for_each_object, 1.0)

            for j in range(prior_for_each_object.size(0)):
                object_for_each_prior[prior_for_each_object[j]] = j

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs_encoded[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]),
                                                  xy_to_cxcy(decoded_arm_locs[i]))

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0
        # Eliminate easy background bboxes from ARM
        arm_scores_prob = F.softmax(arm_scores, dim=2)
        easy_negative_idx = arm_scores_prob[:, :, 1] < self.theta

        positive_priors = positive_priors & ~easy_negative_idx

        # LOCALIZATION LOSS
        loc_loss = self.odm_loss(odm_locs[positive_priors].view(-1, 4),
                                 true_locs_encoded[positive_priors].view(-1, 4))

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.odm_cross_entropy(odm_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, -1)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg[easy_negative_idx] = 0.

        conf_loss_neg, _ = conf_loss_neg.sort(dim=-1,
                                              descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(
            self.device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss

    def forward(self, arm_locs, arm_scores, odm_locs, odm_scores, boxes, labels):
        """
        :param arm_locs: offset prediction and binary classification scores from Anchor Refinement Modules
        :param arm_scores:10
        :param odm_locs: offset refinement prediction and multi-class classification scores from ODM
        :param odm_scores:
        :param boxes: gt bbox and labels
        :param labels:
        :return:
        """
        arm_loss = self.compute_arm_loss(arm_locs, arm_scores, boxes, labels)
        odm_loss = self.compute_odm_loss(arm_locs.data.detach(), arm_scores.data.detach(), odm_locs, odm_scores, boxes,
                                         labels)

        # TOTAL LOSS
        return arm_loss + odm_loss
