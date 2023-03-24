import os
import torch
import pickle as pkl
import numpy as np

_qtypes_map = {"1_action_1st": 0,
               "2_action_3rd": 1,
               "3_what_obj_1st": 2,
               "3_what_obj_3rd": 3,
               "4_who_1st": 4,
               "4_who_3rd": 5,
               "5_count": 6,
               "6_what_color": 7,
               "7_other": 8}

_inv_qtypes_map = {0: "1_action_1st",
                   1: "2_action_3rd",
                   2: "3_what_obj_1st",
                   3: "3_what_obj_3rd",
                   4: "4_who_1st",
                   5: "4_who_3rd",
                   6: "5_count",
                   7: "6_what_color",
                   8: "7_other"}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, path_feats,
                 aug_tech=-1, app_ft_vcs=True, mot_ft_vcs=True, app_layer="pool5", mot_layer="pool5"):
        self.labels = labels  
        self.list_IDs = list_IDs  
        if aug_tech >= 0:
            augs_to_apply = _augs[aug_tech]
            for _f in augs_to_apply:
                self.list_IDs = _f(self.list_IDs, labels)
        self.vgg_dict = {}
        self.c3d_dict = {}

        self.app_ft_vcs = app_ft_vcs
        self.app_layer = app_layer
        self.mot_ft_vcs = mot_ft_vcs
        self.mot_layer = mot_layer
        self.path_feats = path_feats

        if app_ft_vcs:
            for vc in labels:
                with open(os.path.join(path_feats, 'vgg_%s.pkl' % (vc,)), 'rb') as f:
                    feat1 = pkl.load(f, encoding='latin1')
                    feat1 = feat1['feat']
                    self.vgg_dict[vc] = feat1

        if mot_ft_vcs:
            for vc in labels:
                with open(os.path.join(path_feats, 'c3d_%s.pkl' % (vc,)), 'rb') as f:
                    feat1 = pkl.load(f, encoding='latin1')
                    feat1 = feat1['feat']
                    self.c3d_dict[vc] = feat1

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs.loc[index]

        if self.app_ft_vcs:
            if 'hf' in ID['video_cam'] or 'vf' in ID['video_cam']:
                if 'h' in ID['video_cam']:
                    vgg_feat = self.vgg_dict_hflipped[ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
                else:
                    assert False, "not implemented"
                    vgg_feat = self.video_feature['vgg_vflipped'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            elif 'r' in ID['video_cam']:
                assert False, "not implemented"
                vgg_feat = self.video_feature['vgg_reversed'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            elif 'b' in ID['video_cam']:
                assert False, "not implemented"
                vgg_feat = self.video_feature['vgg_blurred'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            else:
                vgg_feat = self.vgg_dict[ID['video_cam']][ID['start']:ID['end']+1]
            vgg_feat = torch.from_numpy(vgg_feat).unsqueeze(0)
        else:
            vgg_feat = []
            for idx in range(ID['start'], ID['end'] + 1):
                with open(os.path.join(self.path_feats,
                                       ID['video_cam'],
                                       'vgg_{}_{}_{}.pkl'.format(ID['video_cam'], self.app_layer, idx)
                                       ), 'rb') as f:
                    feat1 = pkl.load(f, encoding="latin1")
                    feat1 = feat1['feat']
                    vgg_feat.append(torch.Tensor(feat1))
            vgg_feat = torch.stack(vgg_feat).unsqueeze(0)  # (1, seq_len, H=7, W=7, C=512)

        if self.mot_ft_vcs:
            if 'hf' in ID['video_cam'] or 'vf' in ID['video_cam']:
                if 'h' in ID['video_cam']:
                    c3d_feat = self.c3d_dict_hflipped[ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
                else:
                    assert False, "not implemented"
                    c3d_feat = self.video_feature['c3d_vflipped'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            elif 'r' in ID['video_cam']:
                assert False, "not implemented"
                c3d_feat = self.video_feature['c3d_reversed'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            elif 'b' in ID['video_cam']:
                assert False, "not implemented"
                c3d_feat = self.video_feature['c3d_blurred'][ID['video_cam'][:3]][ID['start']:ID['end'] + 1]
            else:
                c3d_feat = self.c3d_dict[ID['video_cam']][ID['start']:ID['end']+1]
            c3d_feat = torch.from_numpy(c3d_feat).unsqueeze(0)
        else:
            c3d_feat = []
            for idx in range(ID['start'], ID['end'] + 1):
                with open(os.path.join(self.path_feats,
                                       ID['video_cam'],
                                       'c3d_{}_{}_{}.pkl'.format(ID['video_cam'], self.mot_layer, idx)
                                       ), 'rb') as f:
                    feat1 = pkl.load(f, encoding="latin1")
                    feat1 = feat1['feat']
                    c3d_feat.append(torch.Tensor(feat1))
            c3d_feat = torch.stack(c3d_feat).unsqueeze(0)  # (1, seq_len, C=512, T=1, H=4, W=4)

        question_length = len(ID['question_encode'].split(","))
        _texts = {"qtext": ID["question"],
                  "a1": ID["a1"],
                  "a2": ID["a2"],
                  "a3": ID["a3"],
                  "a4": ID["a4"],
                  "a5": ID["a5"]
                  }

        return ID['question_encode'], ID['label'], ID['a1_encoder'], ID['a2_encoder'], ID['a3_encoder'], ID[
            'a4_encoder'], ID['a5_encoder'], vgg_feat, c3d_feat, question_length, ID['question'], ID['answer'], _texts
