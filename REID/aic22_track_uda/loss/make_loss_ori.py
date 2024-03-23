# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import numpy as np
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import matplotlib.pyplot as plt
def focal_loss(labels, logits, alpha, gamma):
    #print('using focal loss')
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss

    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss
    
def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights_2 = torch.tensor(weights).float()
    weights_2 = weights_2.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #plt.figure('Draw')
    #plt.plot(weights)
    #plt.savefig("easyplot02.jpg")
    
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    elif loss_type == "classbalance":
        cb_loss = F.cross_entropy(input = logits, target = labels, weight = weights_2)
    return cb_loss

def make_loss(cfg, num_classes, class_weights=None):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))

    else:
        print('not supported METRIC_LOSS_TYPE: {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target, cam_label=None): # score: classifier's logits, feat: feature embedding, target: cls label
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return xent(score, target)
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
 
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #print('using right sampler and loss')
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                              cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        # ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                        # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
        
                    else:

                        ID_LOSS = F.cross_entropy(score, target)
        
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                              cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet_focal':
 
        def loss_func(score, feat, target, target_cam, class_weights=class_weights):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #print('using right sampler and loss') 
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                              cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        # ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                        # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
        
                    else:
                        no_of_classes = num_classes
                        beta = 0.9999
                        gamma = 2.0
                        samples_per_cls = class_weights
                        for i in class_weights:
                            assert i >= 0
                            assert i <2028
                        loss_type = "classbalance"
                        #ID_LOSS = F.cross_entropy(score, target)
                        
                        #plt.figure('Draw')
                        #plt.plot(class_weights)
                        #plt.savefig("easyplot01.jpg")
                        ID_LOSS = CB_loss(target, score, samples_per_cls, no_of_classes, loss_type, beta, gamma)
                        #print(ID_LOSS)
                        #print(ID_LOSS.size())
                        #exit()
        
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                    #print(f'ID loss: {ID_LOSS}',f'TRI_LOSS: {TRI_LOSS}')
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                              cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))               
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet_center':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion

