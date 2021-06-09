# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!::::"+cfg.DATALOADER.SAMPLER)
    if sampler == 'softmax':
        def loss_func(score, feat, q_feat,target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, q_feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(scorcls_score_soft,cls_score_harde, feat, q_feat_soft,q_feat_hard, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #print("fuckL::::::"+cfg.MODEL.IF_LABELSMOOTH)
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    #return xent(score.detach(), target) + triplet(feat, target)[0] + F.mse_loss(q_feat, feat.detach())
                    return xent(scorcls_score_soft, target) + \
                        triplet(feat, target)[0] + \
                        F.mse_loss(q_feat_hard, feat.detach()) + \
                        F.mse_loss(q_feat_hard,q_feat_soft) + \
                        xent(scorcls_score_soft, target) + \
                        F.mse_loss(scorcls_score_soft,scorcls_score_soft)
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    # def loss_func(scorcls_score_soft,cls_score_harde, feat, q_feat_soft,q_feat_hard, target):
    #     if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
    #         if cfg.MODEL.IF_LABELSMOOTH == 'on':
    #             return xent(score, target) + \
    #                     cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
    #         else:
    #             return F.cross_entropy(score, target) + \
    #                     cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

    #     elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
    #         if cfg.MODEL.IF_LABELSMOOTH == 'on':
    #             #print("on")
    #             # return xent(score, target) + \
    #             #         triplet(feat, target)[0] + \
    #             #         cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
    #             #         F.mse_loss(q_feat_soft, feat.detach()) + \
    #             #         F.mse_loss(q_feat_hard,q_feat_soft)
    #             return xent(scorcls_score_soft, target) + \
    #                     triplet(feat, target)[0] + \
    #                     cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
    #                     F.mse_loss(q_feat_hard, feat.detach()) + \
    #                     F.mse_loss(q_feat_hard,q_feat_soft) + \
    #                     xent(scorcls_score_soft, target) + \
    #                     F.mse_loss(scorcls_score_soft,scorcls_score_soft)
                        
    #         else:
    #             return F.cross_entropy(score, target) + \
    #                     triplet(feat, target)[0] + \
    #                     cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

    #     else:
    #         print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
    #               'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    # return loss_func, center_criterion

    #bu jia joint loss
    def loss_func(scorcls_score_soft,cls_score_harde, feat, q_feat_soft,q_feat_hard, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(cls_score_harde, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(cls_score_harde, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(cls_score_harde, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(cls_score_harde, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion