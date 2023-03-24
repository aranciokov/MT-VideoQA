# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_rand import *
from memory_module import *
# from data.util.GST.models import TemporalModel
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import XLMTokenizer, XLMModel, XLMConfig
import  numpy as np

class SpatialAttentionModule(nn.Module):

    def __init__(self, input_size=3072, feat_dim=7, hidden_size=512, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(SpatialAttentionModule, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feat_dim = feat_dim
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.drop_keep_prob_image_embed_t = nn.Dropout(dropout)
         
        self.init_weights()
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)



    def forward(self, hidden_frames, hidden_text):
        
        # hidden_text:  1 x 1024 (tgif-qa paper Section 4.2, use a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x 7 x 7 x 3072 (from C3D and resnet, 1024+2048 = 3072)
        #print(hidden_frames.size(),'spatial 1---') # (1, 1, 1, 2048) or (1, 7, 7, 3072)
        assert self.feat_dim==hidden_frames.size(2)
        hidden_frames = hidden_frames.view(hidden_frames.size(0), hidden_frames.size(1) * hidden_frames.size(2), hidden_frames.size(3))
        #hidden_frames = hidden_frames.permute([0,2,1])
        #print('Spatial hidden_frames transferred size', hidden_frames.size())  # should be 1 x 49 x 3072 or (1, 1, 2048)
        
        # precompute Uahj, see Appendix A.1.2 Page12, last sentence, 
        # NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)
        #Uh = Uh.repeat(1,hidden_frames.size(1),1) 
        #print('Spatial Uh size', Uh.size())  # (1, 1, 512)

        
        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        # W is 512x512, s_i-1 is 
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,49,512)
        #print('Spatial Ws size',Ws.size())   # (1, 1 or 49, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        #print('Spatial att_vec size',att_vec.size()) # should be 1x49, as weights for each encoder output ht
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1x49 to 1x49x1
        #print('Spatial expanded att_vec size',att_vec.size())   # (1, 1 or 49, 1)
                
        
        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        #print('Spatial ht_weighted size', ht_weighted.size()) # should be (1,49,input_size)  (1, 1(or 49), 2048)
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum = self.drop_keep_prob_image_embed_t(ht_sum)
        #print('Spatial ht_sum size', ht_sum.size())  # should be (1,input_size)  (1, 2048)
        
        return ht_sum

class TemporalAttentionModule(nn.Module):

    def __init__(self, input_size, hidden_size=512):
        """Set the hyper-parameters and build the layers."""
        super(TemporalAttentionModule, self).__init__()
        self.input_size = input_size   # in most cases, 2*hidden_size
        self.hidden_size = hidden_size
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        #self.Wa = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        #self.Ua = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.init_weights()
        
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)


    def forward(self, hidden_frames, hidden_text, inv_attention=False):
        
        # hidden_text:  1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T x 1024 (from video encoder output, 1024 is similar from above)

        #print('Temporal hidden_text transferred size', hidden_text.size())  # should be 1 x 1024
        #print('Temporal hidden_frames transferred size', hidden_frames.size())  # should be 1 x T x 3072
        
        # precompute Uahj, see Appendix A.1.2 Page12, last sentence, 
        # NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)
        #print('Temporal Uh size', Uh.size())  # (1, 1, 512)

        
        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,T,512)
        #print('Temporal Ws size',Ws.size())       # (1, T, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        
        if inv_attention:
            att_vec = - att_vec

        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        #print('Temporal att_vec size',att_vec.size()) # should be 1xT, as weights for each encoder output ht  (1,T,1)
        
#         if inv_attention==True:
#             att_vec = 1.0 - att_vec
        
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1xT to 1xTx1 
        #print('Temporal expanded att_vec size',att_vec.size())
                
        
        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        #print('Temporal ht_weighted size', ht_weighted.size()) # should be (1,T,input_size)  # (1, T, 1024)
        ht_sum = torch.sum(ht_weighted, dim=1)
        #print('Temporal ht_sum size', ht_sum.size())  # should be (1,input_size)  (1, 1024)

        return ht_sum

# attention-based multimodal
class MultiModalNaiveModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalNaiveModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.init_weights()
        
    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.01)
        self.Wat.data.normal_(0.0, 0.01)
        self.Uav.data.normal_(0.0, 0.01)
        self.Uat.data.normal_(0.0, 0.01)
        self.Vav.data.normal_(0.0, 0.01)
        self.Vat.data.normal_(0.0, 0.01)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.01)
        self.Wvh.data.normal_(0.0, 0.01)
        self.Wth.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

    def forward(self, h, hidden_frames, hidden_text, inv_attention=False):
        
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )
        

        
        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)
        #print att_vec_v.size(),att_vec_t.size()
        
        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        
        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        
        
        output = torch.tanh( torch.matmul(h,self.Whh) + torch.matmul(hv_sum,self.Wvh) + 
                             torch.matmul(ht_sum, self.Wth) + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output
        
class MultiModalAttentionModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        

        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)

        
        self.init_weights()
        
    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.01)
        self.Wat.data.normal_(0.0, 0.01)
        self.Uav.data.normal_(0.0, 0.01)
        self.Uat.data.normal_(0.0, 0.01)
        self.Vav.data.normal_(0.0, 0.01)
        self.Vat.data.normal_(0.0, 0.01)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.01)
        self.Wvh.data.normal_(0.0, 0.01)
        self.Wth.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

        self.video_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.video_sum_encoder.bias.data.fill_(0)
        self.question_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.question_sum_encoder.bias.data.fill_(0)


        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)
        
        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)

        
    def forward(self, h, hidden_frames, hidden_text):
        
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )

        
        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)
        #print att_vec_v.size(),att_vec_t.size()
        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        hv_sum2 = self.video_sum_encoder(hv_sum)

        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum2 = self.question_sum_encoder(ht_sum)        
        
        
        Wbs = torch.matmul(h, self.Wb)
        mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
        mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
        mtv =  torch.tanh(torch.cat([mv1,mt1],dim=0))
        mtv2 = torch.matmul(mtv, self.wb)
        beta = torch.softmax(mtv2,dim=0)
        #print beta.size(),beta        
        
        output = torch.tanh( torch.matmul(h,self.Whh) + beta[0] * hv_sum2 + 
                             beta[1] * ht_sum2 + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output

class MultiModalAttentionModule_batch(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule_batch, self).__init__()

        self.hidden_size = hidden_size
        self.simple = simple

        # alignment model
        # see appendices A.1.2 of neural machine translation

        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size)
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size)


        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)


        self.init_weights()

    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.01)
        self.Wat.data.normal_(0.0, 0.01)
        self.Uav.data.normal_(0.0, 0.01)
        self.Uat.data.normal_(0.0, 0.01)
        self.Vav.data.normal_(0.0, 0.01)
        self.Vat.data.normal_(0.0, 0.01)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.01)
        self.Wvh.data.normal_(0.0, 0.01)
        self.Wth.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

        self.video_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.video_sum_encoder.bias.data.fill_(0)
        self.question_sum_encoder.weight.data.normal_(0.0, 0.02)
        self.question_sum_encoder.bias.data.fill_(0)


        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)

        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)


    def forward(self, h, hidden_frames, hidden_text, video_lens, qa_lens, batch_size, num_choices):
        _output = []
        # h: 1 x B x 256
        # hidden_text:  B*Choices x T1 x 256 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: B*Choices x T2 x 256 (from video encoder output, 1024 is similar from above)

        for _b in range(batch_size):
            _vl = video_lens[_b]
            _qal = max(qa_lens[_b*num_choices:(_b+1)*num_choices])
            _vids = hidden_frames[_b*num_choices:(_b+1)*num_choices, :_vl, :]
            _txt = hidden_text[_b*num_choices:(_b+1)*num_choices, :_qal, :]
            _h = h[:, _b*num_choices:(_b+1)*num_choices,:]

            #print hidden_frames.size(),hidden_text.size()
            Uhv = torch.matmul(_h, self.Uav)  # (1,512)
            Uhv = Uhv.squeeze(0)
            Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

            Uht = torch.matmul(_h, self.Uat)  # (1,512)
            Uht = Uht.squeeze(0)
            Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
            #print Uhv.size(),Uht.size()

            Wsv = torch.matmul(_vids, self.Wav) # (1,T,512)
            #print Wsv.size()
            att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )

            Wst = torch.matmul(_txt, self.Wat) # (1,T,512)
            att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )


            att_vec_v = torch.softmax(att_vec_v, dim=1)
            att_vec_t = torch.softmax(att_vec_t, dim=1)
            #print att_vec_v.size(),att_vec_t.size()

            att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1
            att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1

            hv_weighted = att_vec_v * _vids
            hv_sum = torch.sum(hv_weighted, dim=1)
            hv_sum2 = self.video_sum_encoder(hv_sum)

            ht_weighted = att_vec_t * _txt
            ht_sum = torch.sum(ht_weighted, dim=1)
            ht_sum2 = self.question_sum_encoder(ht_sum)


            Wbs = torch.matmul(_h, self.Wb)
            mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
            mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
            mtv = torch.tanh(torch.cat([mv1,mt1],dim=0))
            mtv2 = torch.matmul(mtv, self.wb)
            beta = torch.softmax(mtv2,dim=0)
            #print beta.size(),beta

            output = torch.tanh(torch.matmul(_h, self.Whh) +
                                beta[0].unsqueeze(1) * hv_sum2 +
                                beta[1].unsqueeze(1) * ht_sum2 + self.bh)
            _output.append(output.view(output.size(1),output.size(2)))

        _output = torch.stack(_output).cuda()
        _output = torch.reshape(_output, (-1, _output.shape[-1]))
        return _output


class LSTMEncDec(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                 answer_vocab_size=None, max_len=20, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(LSTMEncDec, self).__init__()

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)

        self.embed = nn.Embedding(vocab_size, text_embed_size)

        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1 = nn.LSTMCell(feat_channel, hidden_size)
        self.lstm_video_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.video_encoder = nn.Linear(feat_channel, hidden_size * 2)

        self.linear_decoder_count_1 = nn.Linear(hidden_size * 2, hidden_size * 2)

        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2, 1)  # Count is regression problem

        self.max_len = max_len

        self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t, s_t2, c_t, c_t2

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):

        video_features = data_dict['video_features']
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']

        outputs = []
        predictions = []
        batch_size = len(questions)  # batch size has to be 1

        for j in range(batch_size):
            features_question_j = self.embed(questions[j])
            vgg, c3d = video_features[0][j], video_features[1][j]
            feature = torch.cat([vgg, c3d], dim=1)
            #print feature.size()

            nImg = vgg.shape[0]
            outputs_j = []

            for n_cand in range(num_mult_choices):

                nQuestionWords = question_lengths[j]

                #############################
                # run text encoder first time
                #############################
                s1_t1, s1_t2, c1_t1, c1_t2 = self.init_hiddens()

                for i in range(nQuestionWords):
                    input_question = features_question_j[n_cand, i:i+1]
                    s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                    s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))

                ###########################################
                # run video encoder with spatial attention
                ###########################################
                sV_t1, sV_t2, cV_t1, cV_t2 = self.init_hiddens()

                # record each time t, hidden states, for later temporal attention after text encoding
                hidden_array_1 = []
                hidden_array_2 = []

                for i in range(nImg):
                    # lstm
                    sV_t1, cV_t1 = self.lstm_video_1(feature[i:i + 1], (sV_t1, cV_t1))
                    sV_t2, cV_t2 = self.lstm_video_2(sV_t1, (sV_t2, cV_t2))

                    sV_t1_vec = sV_t1.view(sV_t1.size(0), 1, sV_t1.size(1))
                    sV_t2_vec = sV_t2.view(sV_t2.size(0), 1, sV_t2.size(1))

                    hidden_array_1.append(sV_t1_vec)
                    hidden_array_2.append(sV_t2_vec)

                # assume sV_t1 is of size (1,1,hidden)
                sV_l1 = torch.cat(hidden_array_1, dim=1)
                sV_l2 = torch.cat(hidden_array_2, dim=1)

                sV_ll = torch.cat((sV_l1, sV_l2), dim=2)

                #############################
                # run text encoder second time
                #############################
                # here sT_t1, sT_t2 are the last hiddens from video, input to text encoder againa
                input_question = features_question_j[n_cand,0:1]
                # print('input_question size', input_question.size())
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sV_t1, cV_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sV_t2, cV_t2))

                for i in range(1, nQuestionWords):
                    input_question = features_question_j[n_cand,i:i+1]
                    sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                    sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                # print('Text encoding One size', sT_t1.size(), sT_t2.size())

                # here sT_t1, sT_t2 is the last hidden
                sT_t = torch.cat((sT_t1, sT_t2), dim=1)  # should be of size (1,1024)

                final_embed = self.linear_decoder_count_1(sT_t)
                output = self.linear_decoder_count_2(final_embed)
                outputs_j.append(output)

            # output is the score of each multiple choice
            outputs_j = torch.cat(outputs_j, 1)
            outputs.append(outputs_j)

            # for evaluate accuracy, find the max one
            _, mx_idx = torch.max(outputs_j, 1)
            predictions.append(mx_idx)
            # print(outputs_j,mx_idx)

        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, predictions

    def accuracy(self, logits, targets):
        correct = torch.sum(logits.eq(targets)).float()
        return correct * 100.0 / targets.size(0)


# _input_dd: dict with keys "qtexts" and "atexts"
# _embed_tech: str in ["bert", "xlm", "glove", "elmo", "glove_frozen", ...]
# _tokenizer: tokenizer used to tokenize the input text
# _answers_lengths: answer lengths
# _dev: str for the device (e.g. "cuda:0")
def tokenize_text(_input_dd, _embed_tech, _tokenizer, _questions, _answers_lengths, _dev):
    if _embed_tech in ["bert", "xlm", "bert_ft", "bert_jointft", "xlm_ft"]:
        qtexts = _input_dd["qtexts"]
        atexts = _input_dd["atexts"]
        _rep_qts = []
        _qlens = []
        for qt in qtexts:
            _rep_qts += [qt] * 5
            _qlens += [len(qt.split())] * 5
        _atexts = []
        _alens = []
        for i in range(len(qtexts)):
            for _k in atexts.keys():
                _atexts.append(atexts[_k][i])
        _alens = [len(at.split()) for at in _atexts]
        qa_lens = [ql + al + 3 for (ql, al) in zip(_qlens, _alens)]  # +3 to take into account [CLS], [SEP]
        _tokenized_text = _tokenizer(_rep_qts, _atexts, padding=True, add_special_tokens=True, return_tensors="pt").to(
            _dev)

    elif _embed_tech in ["glove", "glove_frozen", "glove_ft"]:
        max_m_questions = max(question.shape[1] for question in _questions)
        questions_tensor = []

        for question in _questions:
            npad = ((0, 0), (0, max_m_questions - question.shape[1]))
            question_single = np.pad(question.cpu(), pad_width=npad, mode='constant')
            questions_tensor.append(question_single)

        questions_tensor = torch.tensor(questions_tensor).to(_dev)
        _tokenized_text = questions_tensor.reshape(questions_tensor.shape[0] * questions_tensor.shape[1],
                                                    questions_tensor.shape[2])
        qtexts = _input_dd["qtexts"]
        _qlens = []
        for qt in qtexts:
            _qlens += [len(qt.split())] * 5
        _alens = [al.item() for al in torch.flatten(_answers_lengths).cpu()]
        qa_lens = [ql + al for (ql, al) in zip(_qlens, _alens)]

    elif _embed_tech in ["elmo", "elmo_ft"]:
        qtexts = _input_dd["qtexts"]
        atexts = _input_dd["atexts"]
        _rep_qts = []
        _qlens = []
        for qt in qtexts:
            _rep_qts += [qt] * 5
            _qlens += [len(qt.split())] * 5
        _atexts = []
        _alens = []
        for i in range(len(qtexts)):
            for _k in atexts.keys():
                _atexts.append(atexts[_k][i])
        _alens = [len(at.split()) for at in _atexts]
        qa_lens = [ql + al for (ql, al) in zip(_qlens, _alens)]
        qas = [" ".join([q, a]).split() for q, a in zip(_rep_qts, _atexts)]
        _tokenized_text = _tokenizer(qas)
        _tokenized_text = _tokenized_text.to(_dev)
        # ^ Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
        #   (len(batch), max sentence length, max word length)

    else:
        assert False, "{} not implemented".format(_embed_tech)

    return _tokenized_text, qa_lens


class TheNetwork(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix, activation_function, embed_tech="glove",
                 answer_vocab_size=None, max_len=20, dropout=0.2, embed_layer=-1, additional_tasks="", architecture="_stvqa",
                 use_int_mem=False, device="cuda:0", dataset="egovqa", return_features=False):
        """Set the hyper-parameters and build the layers."""
        super(TheNetwork, self).__init__()

        self.return_features = return_features
        self.device = device

        assert architecture in ["_stvqa", "_enc_dec", "_mrm2s", "_co_mem", "_mrm2s_convLSTM", "_mrm2s_FTFCs"], \
            "{} not implemented yet".format(architecture)
        self.architecture = architecture

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_int_mem = use_int_mem

        if architecture in ["_mrm2s", "_mrm2s_FTFCs"]:  # TpAtt will receive the features from both the stacked LSTM layers
            self.TpAtt_a = TemporalAttentionModule(hidden_size * 2, hidden_size)
            self.TpAtt_m = TemporalAttentionModule(hidden_size * 2, hidden_size)
        else:
            self.TpAtt = TemporalAttentionModule(hidden_size * 2, hidden_size)

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)

        assert embed_tech in ["glove", "glove_frozen", "bert", "xlm", "elmo",
                              "elmo_ft", "bert_ft", "bert_jointft", "xlm_ft", "glove_ft"], \
            "{} not implemented yet".format(architecture)
        self.embed_tech = embed_tech

        if embed_tech in ["glove", "glove_frozen", "glove_ft"]:
          self.embed = nn.Embedding(vocab_size, text_embed_size)
          if embed_tech == "glove_frozen":
              self.embed = self.embed.eval()
        elif embed_tech in ["bert", "xlm", "bert_ft", "bert_jointft", "xlm_ft"]:
            if embed_tech in ["bert", "bert_ft", "bert_jointft"]:
                model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
                config = BertConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
                self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
            else:
                model_class, tokenizer_class, pretrained_weights = (XLMModel, XLMTokenizer, 'xlm-mlm-en-2048')
                config = XLMConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
                self.tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
            self.embed = model_class.from_pretrained(pretrained_weights, config=config)
            if "ft" not in embed_tech:
                self.embed = self.embed.eval()
            self.embed_layer = embed_layer
        elif embed_tech in ["elmo", "elmo_ft"]:
            if device == "cuda:1":
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = "1"
            from allennlp.modules.elmo import Elmo, batch_to_ids
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            if embed_tech == "elmo":
                self.embed = Elmo(options_file, weight_file, 1, requires_grad=False, dropout=0)
                self.embed = self.embed.eval()
            else:
                self.embed = Elmo(options_file, weight_file, 1, requires_grad=True, dropout=0)
            self.batch_to_ids = batch_to_ids

        # ---v additional tasks v---
        #for _add_tsk in additional_tasks.split(","):
        #    assert _add_tsk in ["debug", "qtclassif"], "{} (in {}) not implemented yet".format(_add_tsk, additional_tasks)
        self.additional_tasks = additional_tasks.split(",") if additional_tasks != "" else []
        if len(self.additional_tasks) > 0:
            if "qtclassif" in self.additional_tasks or "qtclassif_unsup" in self.additional_tasks:
                if dataset == "egovqa":
                    from dataset_pytorch import _qtypes_map
                else:
                    from dataset_pororo import _qtypes_map
                self.qt_classifier = nn.Linear(hidden_size*2, len(_qtypes_map))  # video -> question type
                self.softmax_qtc = nn.Softmax(dim=-1)
        # ---^ additional tasks ^---

        if architecture in ["_mrm2s_FTFCs"]:
            """self.vggfc_1 = nn.Linear(512 * 7 * 7, 4096)
            self.vggfc_2 = nn.Linear(4096, 4096)
            self.c3dfc_1 = nn.Linear(512 * 4 * 4, 4096)
            self.c3dfc_2 = nn.Linear(4096, 4096)"""
            self.fc_relu = nn.ReLU()
            self.fc_drop = nn.Dropout()
            self.vgg_ap = nn.AvgPool2d((7, 7))
            self.vggfc_1 = nn.Linear(512, 4096)
            self.c3d_ap = nn.AvgPool2d((4, 4))
            self.c3dfc_1 = nn.Linear(512, 4096)

        if architecture in ["_mrm2s", "_mrm2s_FTFCs"]:
            self.lstm_mm_2 = nn.LSTM(hidden_size, hidden_size)
            self.lstm_mm_1 = nn.LSTM(hidden_size, hidden_size)
            self.linear_mem = nn.Linear(hidden_size*2, hidden_size)
            self.linear_att_a = nn.Linear(hidden_size*2, hidden_size)
            self.linear_att_m = nn.Linear(hidden_size*2, hidden_size)
            self.mm_att = MultiModalAttentionModule_batch(hidden_size)
            self.hidden_enc_1 = nn.Linear(hidden_size * 2, hidden_size)
            self.hidden_enc_2 = nn.Linear(hidden_size * 2, hidden_size)
            self.mrm_vid = MemoryRamTwoStreamModule_batch(hidden_size, hidden_size, max_len)
            self.mrm_txt = MemoryRamModule_batch(hidden_size, hidden_size, max_len)
        elif architecture == "_co_mem":
            self.ma_decoder = nn.Linear(hidden_size * 2 * 3, hidden_size * 2)
            self.mb_decoder = nn.Linear(hidden_size * 2 * 3, hidden_size * 2)
            self.epm1 = EpisodicMemory(hidden_size * 2)
            self.epm2 = EpisodicMemory(hidden_size * 2)
            self.linear = nn.Linear(hidden_size*2*2, hidden_size*2)
        else:
            self.linear = nn.Linear(hidden_size*2, hidden_size*2)
        self.activation_functions = activation_function()
        if architecture in ["_mrm2s", "_mrm2s_FTFCs"]:
            self.linear2 = nn.Linear(hidden_size*2 + hidden_size, 1)
        else:
            self.linear2 = nn.Linear(hidden_size*2, 1)

        if architecture in ["_mrm2s", "_mrm2s_FTFCs", "_co_mem"]:
            self.lstm_video_1_a = nn.LSTM(feat_channel, hidden_size, batch_first=True)
            self.lstm_video_2_a = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.lstm_video_1_m = nn.LSTM(feat_channel, hidden_size, batch_first=True)
            self.lstm_video_2_m = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.lstm_video_1 = nn.LSTM(feat_channel, hidden_size, batch_first=True)
            self.lstm_video_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.lstm_text_1 = nn.LSTM(text_embed_size, hidden_size, batch_first=True)
        self.lstm_text_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.max_len = max_len

        if embed_tech in ["glove", "glove_frozen"]:
            self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).to(self.device)
        s_t2 = torch.zeros(1, self.hidden_size).to(self.device)
        c_t = torch.zeros(1, self.hidden_size).to(self.device)
        c_t2 = torch.zeros(1, self.hidden_size).to(self.device)
        return s_t, s_t2, c_t, c_t2

    def init_hidden(self, batch_size):
        s_t = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return s_t

    def init_hidden_t(self, batch_size, C, H, W):
        s_t = torch.rand(1, batch_size, C, H, W).to(self.device)
        return s_t

    def forward(self, data_dict):
        device = self.device
        batch_size = len(data_dict['question_words'])
        num_choices = data_dict["num_mult_choices"]
        video_features = data_dict['video_features']

        max_n_vgg = max(video.shape[1] for video in video_features[0])
        max_n_c3d = max(video.shape[1] for video in video_features[1])
        assert max_n_c3d == max_n_vgg, "different video lengths"
        video_lengths = [video.shape[1] for video in video_features[0]]

        app_feats = torch.nn.utils.rnn.pad_sequence([v.squeeze(0) for v in video_features[0]], batch_first=True)
        mot_feats = torch.nn.utils.rnn.pad_sequence([v.squeeze(0) for v in video_features[1]], batch_first=True)
        tokenized_txt, qa_lens = self.tokenize_txt(data_dict, device)
        #print(tokenized_txt)
        embed_txt = self.embed_txt(tokenized_txt, batch_size, num_choices, qa_lens)
        #embed_txt.masked_fill_(torch.stack([torch.arange(max(qa_lens)) > q for q in qa_lens]).unsqueeze(-1).to(self.device), 0.)
        #print("embed_txt {}".format(torch.sum(embed_txt, (2, 1))))
        # embed_txt [B*Choices, n_words, E] -> full sequence of word embeddings
        #print("----- video & qa lens ", video_lengths, qa_lens)
        if self.architecture in ["_stvqa", "_enc_dec"]:
            features = torch.cat((app_feats, mot_feats), -1)
            vid_fts, h1, c1, h2, c2 = self.forward_vid(features, batch_size,  video_lengths)
            txt_fts, _, h1, c1, h2, c2 = self.forward_txt(embed_txt, qa_lens, batch_size, num_choices, h1, h2, c1, c2)

            # ---- additional tasks ----
            if len(self.additional_tasks) > 0 and ("qtclassif" in self.additional_tasks
                                                   or "qtclassif_unsup" in self.additional_tasks):
                qt_classes = self.predict_QT(txt_fts, batch_size, num_choices)
            # ---- additional tasks ----

            if self.architecture == "_stvqa":
                outputs, predictions, out_features = self.forward_stvqa(batch_size, vid_fts, txt_fts, self.TpAtt)
            else:
                outputs, predictions = self.forward_encdec(batch_size, txt_fts)

        elif self.architecture in ["_mrm2s", "_co_mem"]:
            feats_a, fts_1a, fts_2a, feats_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m, lhs_txt = self.forward_dualstream(batch_size, qa_lens, video_lengths, num_choices, embed_txt, app_feats, mot_feats)

            # ---- additional tasks ----
            if len(self.additional_tasks) > 0 and ("qtclassif" in self.additional_tasks
                                                   or "qtclassif_unsup" in self.additional_tasks):
                qt_classes = self.predict_QT(lhs_txt, batch_size, num_choices)
            # ---- additional tasks ----

            #print("h1a", torch.sum(h1_a, 2).shape,torch.sum(h1_a, 2))
            #print("h1m", torch.sum(h1_m, 2).shape,torch.sum(h1_m, 2))
            # feats_{a,m} [B*Choices, seq_len, H+H] -> concat of both LSTM layers' hidden states
            # h{1,2}_{a,m} [1, B*Choices, H] -> last hidden state of {1st,2nd} LSTM layer for {appearance,motion} feats
            if self.architecture == "_mrm2s":
                outputs, predictions = self.forward_mrm2s(batch_size, video_lengths, qa_lens, num_choices, embed_txt,
                                                          feats_a, fts_1a, fts_2a, feats_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m)
            else:
                outputs, predictions = self.forward_comem(batch_size, fts_1a, fts_2a, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m, lhs_txt)

        elif self.architecture in ["_mrm2s_FTFCs"]:
            # tensors->feat vectors
            # -- à-la VGG, worse results
            #_vgg = app_feats.view(app_feats.shape[0], app_feats.shape[1], -1)
            #_vgg = self.fc_drop(self.fc_relu(self.vggfc_1(_vgg)))
            #_vgg = self.fc_drop(self.fc_relu(self.vggfc_2(_vgg)))
            #_c3d = mot_feats.view(mot_feats.shape[0], mot_feats.shape[1], -1)
            #_c3d = self.fc_drop(self.fc_relu(self.c3dfc_1(_c3d)))
            #_c3d = self.fc_drop(self.fc_relu(self.c3dfc_2(_c3d)))

            # better results
            _vgg = app_feats.view(app_feats.shape[0]*app_feats.shape[1], app_feats.shape[-1], app_feats.shape[-2], -1)
            _vgg = self.vgg_ap(_vgg)
            _vgg = _vgg.view(app_feats.shape[0], app_feats.shape[1], app_feats.shape[-1])
            _vgg = self.fc_drop(self.fc_relu(self.vggfc_1(_vgg)))
            _c3d = mot_feats.squeeze(3)
            _c3d = _c3d.view(_c3d.shape[0]*_c3d.shape[1], _c3d.shape[2], _c3d.shape[3], -1)
            _c3d = self.c3d_ap(_c3d)
            _c3d = _c3d.view(mot_feats.shape[0], mot_feats.shape[1], mot_feats.shape[2])
            _c3d = self.fc_drop(self.fc_relu(self.c3dfc_1(_c3d)))
            # _vgg, _c3d feature vectors

            feats_a, fts_1a, fts_2a, feats_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m, lhs_txt = self.forward_dualstream(batch_size, qa_lens, video_lengths, num_choices, embed_txt, _vgg, _c3d)

            # ---- additional tasks ----
            if len(self.additional_tasks) > 0 and ("qtclassif" in self.additional_tasks
                                                   or "qtclassif_unsup" in self.additional_tasks):
                qt_classes = self.predict_QT(lhs_txt, batch_size, num_choices)
            # ---- additional tasks ----

            outputs, predictions = self.forward_mrm2s(batch_size, video_lengths, qa_lens, num_choices, embed_txt,
                                                      feats_a, fts_1a, fts_2a, feats_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m)

        _finaloutput = {"outputs": outputs,
                        "predictions": predictions}
        if self.return_features:
            if self.architecture == "_stvqa":
                _finaloutput["features"] = out_features

        # ---- additional tasks ----
        if len(self.additional_tasks) > 0 and ("qtclassif" in self.additional_tasks
                                               or "qtclassif_unsup" in self.additional_tasks):
            _finaloutput["qt_classes"] = qt_classes
        # ---- additional tasks ----

        return _finaloutput

    def predict_QT(self, txt_fts, batch_size, num_choices):
        assert txt_fts.shape == (1, batch_size*num_choices, self.hidden_size*2), \
            "predict_QT: expected {}, got {}".format(
                (1, batch_size * num_choices, self.hidden_size * 2), txt_fts.shape
            )
        logits = self.qt_classifier(txt_fts)
        qt_classes = self.softmax_qtc(logits)
        return qt_classes

    def compress_app_fts(self, app_fts):
        B = app_fts.shape[1]
        N = app_fts.shape[0]
        C = app_fts.shape[2]
        H = app_fts.shape[3]
        W = app_fts.shape[4]
        app_fts = app_fts.view(B * N, C, H, W)
        # (B', C, H, W)

        app_fts = self.lin_app_fts_1(app_fts)
        app_fts = app_fts.view(B, N, -1)
        app_fts = self.lin_app_fts_2(app_fts)
        #app_fts = self.comp_relu(self.drop(self.lin_app_fts_2(app_fts)))
        return app_fts

    def compress_mot_fts(self, mot_fts):
        B = mot_fts.shape[1]
        N = mot_fts.shape[0]
        C = mot_fts.shape[2]
        H = mot_fts.shape[3]
        W = mot_fts.shape[4]
        mot_fts = mot_fts.view(B * N, C, H, W)
        # (B', C, H, W)

        mot_fts = self.lin_mot_fts_1(mot_fts)
        mot_fts = mot_fts.view(B, N, -1)
        mot_fts = self.lin_mot_fts_2(mot_fts)
        #app_fts = self.comp_relu(self.drop(self.lin_mot_fts_2(mot_fts)))
        return mot_fts

    def forward_encdec(self, batch_size, txt_fts):
        output1 = self.linear(txt_fts)
        output3 = self.linear2(output1)

        outputs = output3.reshape(batch_size, 5)
        _, predictions = torch.max(outputs, 1)
        return outputs, predictions

    def forward_stvqa(self, batch_size, vid_fts, txt_fts, att_module):
        vid_att = self.TpAtt(vid_fts, txt_fts.squeeze(0))
        output1 = self.linear(vid_att)
        output2 = self.activation_functions(output1) * txt_fts
        output3 = self.linear2(output2)

        outputs = output3.reshape(batch_size, 5)
        _, predictions = torch.max(outputs, 1)
        return outputs, predictions, vid_att

    def forward_dualstream(self, batch_size, qa_lens, video_lengths, num_choices, embed_txt, app_feats, mot_feats):
        _, _, h1, c1, h2, c2 = self.forward_txt(embed_txt, qa_lens, batch_size, num_choices)

        #print("txt h1", torch.sum(h1, 2))
        #print("txt h2", torch.sum(h2, 2))
        _last_hs = torch.cat((h1, h2), -1)
        """for _bs in range(2):
            for _ncand in range(5):
                print("txt last hidden states batch{} cand{} {} {}".format(
                    _bs, _ncand, _last_hs[:,(_bs*_ncand)+_ncand,:].shape, torch.sum(_last_hs[:,(_bs*_ncand)+_ncand,:])
                ))"""

        feats_a, fts_1a, fts_2a, h1_a, c1_a, h2_a, c2_a = self.forward_vid_mrm(mot_feats, batch_size, video_lengths,
                                                                               self.lstm_video_1_a, self.lstm_video_2_a,
                                                                               h1, h2, c1, c2)
        """for _bs in range(2):
            for _ncand in range(5):
                print("vid seq APP hidden states batch{} cand{} {} {}".format(
                    _bs, _ncand, feats_a[(_bs * _ncand) + _ncand, :, :].shape,
                    torch.sum(feats_a[(_bs * _ncand) + _ncand, :, :])
                ))"""

        feats_m, fts_1m, fts_2m, h1_m, c1_m, h2_m, c2_m = self.forward_vid_mrm(app_feats, batch_size, video_lengths,
                                                                               self.lstm_video_1_m, self.lstm_video_2_m,
                                                                               h1, h2, c1, c2)
        """for _bs in range(2):
            for _ncand in range(5):
                print("vid seq MOT hidden states batch{} cand{} {} {}".format(
                    _bs, _ncand, feats_m[(_bs * _ncand) + _ncand, :, :].shape,
                    torch.sum(feats_m[(_bs * _ncand) + _ncand, :, :])
                ))"""

        _out = feats_a, fts_1a, fts_2a, feats_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m, _last_hs
        return _out

    def forward_dec_comem(self, M):
        out = torch.tanh(self.linear(M))
        return out

    def forward_comem(self, batch_size, fts_1a, fts_2a, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m, lhs_txt):
        M1 = torch.cat((h1_a, h2_a), -1).squeeze(0)  # (B*Choices, H+H)
        M2 = torch.cat((h1_m, h2_m), -1).squeeze(0)

        ma = M1.detach()
        mb = M2.detach()

        for hop in range(3):
            M1 = M1.view(M1.size(0), 1, M1.size(1))
            M2 = M2.view(M2.size(0), 1, M2.size(1))
            mm = ma + mb
            M1 = self.epm1(torch.cat((fts_1a, fts_2a), -1), mm, M1)
            M2 = self.epm2(torch.cat((fts_1m, fts_2m), -1), mm, M2)
            M1 = M1.view(M1.size(0), M1.size(2))
            M2 = M2.view(M2.size(0), M2.size(2))

            maq = torch.cat([ma, M1, lhs_txt.squeeze(0)], dim=1)
            mbq = torch.cat([mb, M2, lhs_txt.squeeze(0)], dim=1)

            ma = torch.tanh(self.ma_decoder(maq))
            mb = torch.tanh(self.mb_decoder(mbq))

        M = torch.cat((ma, mb), dim=1)
        final_embed = self.forward_dec_comem(M)
        outputs, predictions = self.forward_dec_2(final_embed, batch_size)
        return outputs, predictions

    def forward_mrm2s(self, batch_size, video_lens, qa_lens, num_choices, embed_txt,
                      fts_a, fts_1a, fts_2a, fts_m, fts_1m, fts_2m, h1_a, h2_a, h1_m, h2_m):
        _, fullseq_h2_txt, h1, c1, h2, c2 = self.forward_txt(embed_txt, qa_lens, batch_size, num_choices, _hn_1=h1_a + h1_m, _hn_2=h2_a + h2_m)
        if self.use_int_mem:
            feats_txt = torch.cat((c1, c2), -1).squeeze(0)
        else:
            feats_txt = torch.cat((h1, h2), -1).squeeze(0)

        """for _bs in range(2):
            for _ncand in range(5):
                print("2nd txt last hidden states batch{} cand{} {} {}".format(
                    _bs, _ncand, feats_txt[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(feats_txt[(_bs * _ncand) + _ncand, :])
                ))"""

        assert len(fts_m.shape) == 3, "fts_m wrong shape: {} expecting (B*Choices, seq_len, H+H)".format(fts_m.shape)
        assert len(fts_a.shape) == 3, "fts_a wrong shape: {} expecting (B*Choices, seq_len, H+H)".format(fts_a.shape)
        assert len(feats_txt.shape) == 2, "feats_txt wrong shape: {} expecting (B*Choices, H+H)".format(feats_txt.shape)
        vid_att_a = self.TpAtt_a(fts_a, feats_txt)
        vid_att_m = self.TpAtt_m(fts_m, feats_txt)
        # print(vid_att_a.shape,fts_a.shape,feats_txt.shape)
        # > torch.Size([10, 512]) torch.Size([10, 10, 512]) torch.Size([10, 512])
        """for _bs in range(2):
            for _ncand in range(5):
                print("att APP batch{} cand{} {} {}".format(
                    _bs, _ncand, vid_att_a[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(vid_att_a[(_bs * _ncand) + _ncand, :])
                ))
        for _bs in range(2):
            for _ncand in range(5):
                print("att MOT batch{} cand{} {} {}".format(
                    _bs, _ncand, vid_att_m[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(vid_att_m[(_bs * _ncand) + _ncand, :])
                ))"""

        assert len(fts_a.shape) == 3, "expected shape (B, seq_len, H) for fts_a, got {}".format(fts_a.shape)
        assert len(fullseq_h2_txt.shape) == 3, "expected shape (B, seq_len, H+H) for fts_a, got {}".format(fullseq_h2_txt.shape)
        nImg = fts_a.shape[1]
        nQws = fullseq_h2_txt.shape[1]
        mem_fts = self.forward_mem(video_lens, qa_lens, batch_size, num_choices, self.mrm_vid, self.mrm_txt,
                                   fts_2a, fts_2m, h2_a, h2_m, fullseq_h2_txt, nImg, nQws)
        #print(mem_fts.shape, torch.sum(mem_fts, (2)))
        final_embed = self.forward_dec_mrm(vid_att_a, vid_att_m, mem_fts)
        outputs, predictions = self.forward_dec_2(final_embed, batch_size)
        return outputs, predictions

    def tokenize_txt(self, data_dict, device):
        video_features = data_dict['video_features']
        questions = data_dict['question_words']
        question_lengths = data_dict['question_lengths']
        num_mult_choices = data_dict['num_mult_choices']
        answers = data_dict['answers']
        answers_lengths = data_dict['answers_lengths']

        if self.embed_tech in ["xlm", "bert", "bert_ft", "bert_jointft", "xlm_ft"]:
            _tok = self.tokenizer
        elif self.embed_tech in ["elmo", "elmo_ft"]:
            _tok = self.batch_to_ids
        else:
            _tok = None

        tokenized_text, qa_lens = tokenize_text(data_dict, self.embed_tech, _tok, questions, answers_lengths, device)
        return tokenized_text, qa_lens

    def embed_txt(self, tokenized_text, batch_size, num_choices, qa_lens):
        if self.embed_tech in ["glove", "glove_ft"]:
            _embeds = []
            for _i in range(batch_size):
                qa_len = max(qa_lens[_i * num_choices:(_i + 1) * num_choices])
                _embed = self.embed(tokenized_text[_i * num_choices:(_i + 1) * num_choices, :qa_len])
                _embeds += [_embed[_r, :, :] for _r in range(_embed.shape[0])]
            features_questions = torch.nn.utils.rnn.pad_sequence(_embeds, batch_first=True)
        elif self.embed_tech == "glove_frozen":
            with torch.no_grad():
                _embeds = []
                for _i in range(batch_size):
                    qa_len = max(qa_lens[_i * num_choices:(_i + 1) * num_choices])
                    _embed = self.embed(tokenized_text[_i * num_choices:(_i + 1) * num_choices, :qa_len])
                    _embeds += [_embed[_r, :, :] for _r in range(_embed.shape[0])]
                features_questions = torch.nn.utils.rnn.pad_sequence(_embeds, batch_first=True)
        elif self.embed_tech in ["bert", "xlm"]:
            with torch.no_grad():
                bert_outputs = self.embed(**tokenized_text)
            features_questions = bert_outputs[0]  # [B*Choices,seq_len,768]
        elif self.embed_tech in ["bert_ft", "bert_jointft", "xlm_ft"]:
            bert_outputs = self.embed(**tokenized_text)
            features_questions = bert_outputs[0]  # [B*Choices,seq_len,768]
        elif self.embed_tech == "elmo":
            with torch.no_grad():
                features_questions = self.embed(tokenized_text)  # dict w/ keys "elmo_representations" AND "mask"
                features_questions = features_questions["elmo_representations"][0]
        elif self.embed_tech == "elmo_ft":
            features_questions = self.embed(tokenized_text)  # dict w/ keys "elmo_representations" AND "mask"
            features_questions = features_questions["elmo_representations"][0]
        else:
            assert False, "embed_txt: {} not found in the available methods".format(self.embed_tech)

        return features_questions

    def forward_txt(self, embed_txt, qa_lens, batch_size, num_choices, _hn_1=None, _hn_2=None, _cn_1=None, _cn_2=None):
        if _hn_1 is None:
            _hn_1 = self.init_hidden(batch_size * num_choices)
        if _hn_2 is None:
            _hn_2 = self.init_hidden(batch_size * num_choices)
        if _cn_1 is None:
            _cn_1 = self.init_hidden(batch_size * num_choices)
        if _cn_2 is None:
            _cn_2 = self.init_hidden(batch_size * num_choices)

        if _hn_1.shape[1] == batch_size:
            _rep_hn1 = []
            for _i in range(batch_size):
                for _c in range(5):
                    # also repeating the same hidden/cell state of the video lstm
                    _rep_hn1.append(_hn_1[:, _i, :])
            _hn_1 = torch.cat(_rep_hn1, 0).unsqueeze(0)
        if _cn_1.shape[1] == batch_size:
            _rep_cn1 = []
            for _i in range(batch_size):
                for _c in range(5):
                    # also repeating the same hidden/cell state of the video lstm
                    _rep_cn1.append(_cn_1[:, _i, :])
            _cn_1 = torch.cat(_rep_cn1, 0).unsqueeze(0)
        if _hn_2.shape[1] == batch_size:
            _rep_hn2 = []
            for _i in range(batch_size):
                for _c in range(5):
                    # also repeating the same hidden/cell state of the video lstm
                    _rep_hn2.append(_hn_2[:, _i, :])
            _hn_2 = torch.cat(_rep_hn2, 0).unsqueeze(0)
        if _cn_2.shape[1] == batch_size:
            _rep_cn2 = []
            for _i in range(batch_size):
                for _c in range(5):
                    # also repeating the same hidden/cell state of the video lstm
                    _rep_cn2.append(_cn_2[:, _i, :])
            _cn_2 = torch.cat(_rep_cn2, 0).unsqueeze(0)

        assert embed_txt.shape[0] == batch_size * num_choices and embed_txt.shape[0] == _hn_1.shape[1] and _hn_2.shape[1] == _hn_1.shape[1] and _cn_2.shape[1] == _cn_1.shape[1], \
            "wrong shapes embed_txt {} _hn_1 {} _cn_1 {} _hn_2 {} _cn_2 {} ".format(
                embed_txt.shape, _hn_1.shape, _cn_1.shape, _hn_2.shape, _cn_2.shape
            )
        _packed = torch.nn.utils.rnn.pack_padded_sequence(embed_txt, qa_lens,
                                                          batch_first=True, enforce_sorted=False)
        _hs_1, (_hn_txt1, _cn_txt1) = self.lstm_text_1(_packed, (_hn_1, _cn_1))
        _hs_2, (_hn_txt2, _cn_txt2) = self.lstm_text_2(_hs_1, (_hn_2, _cn_2))
        """_tfeats = []
        for _i, _l in enumerate(qa_lens):
                _hs1_last = _hs_1[_i][_l - 1].unsqueeze(0)
                _hs2_last = _hs_2[_i][_l - 1].unsqueeze(0)
                _tfeats.append(torch.cat((_hs1_last, _hs2_last), dim=-1))"""
        if self.use_int_mem:
            decoder_text_rnn = torch.cat((_cn_txt1, _cn_txt2), -1)
        else:
            decoder_text_rnn = torch.cat((_hn_txt1, _hn_txt2), -1)
        _hs_2, _ = torch.nn.utils.rnn.pad_packed_sequence(_hs_2, batch_first=True)
        return decoder_text_rnn, _hs_2, _hn_txt1, _cn_txt1, _hn_txt2, _cn_txt2

    def forward_vid(self, features, batch_size, video_lengths, _hn_1=None, _hn_2=None, _cn_1=None, _cn_2=None):
        if _hn_1 is None:
            _hn_1 = self.init_hidden(batch_size)
        if _hn_2 is None:
            _hn_2 = self.init_hidden(batch_size)
        if _cn_1 is None:
            _cn_1 = self.init_hidden(batch_size)
        if _cn_2 is None:
            _cn_2 = self.init_hidden(batch_size)

        assert features.shape[0] == batch_size and _hn_1.shape[1] == batch_size, \
            "forward_vid: both video fts. and hiddens need to have same batch size ({}), got {} and {}".format(
                batch_size, features.shape[0], _hn_1.shape[1]
            )
        _packed = torch.nn.utils.rnn.pack_padded_sequence(features, video_lengths,
                                                          batch_first=True, enforce_sorted=False)
        _hs_1, (_hn_vid1, _cn_vid1) = self.lstm_video_1(_packed, (_hn_1, _cn_1))
        _hs_2, (_hn_vid2, _cn_vid2) = self.lstm_video_2(_hs_1, (_hn_2, _cn_2))
        _vfeats = []
        _hs_1, _ = torch.nn.utils.rnn.pad_packed_sequence(_hs_1, batch_first=True)
        _hs_2, _ = torch.nn.utils.rnn.pad_packed_sequence(_hs_2, batch_first=True)
        for _i in range(batch_size):
            for _c in range(5):  # repeat same video feats 5 (=num_choices) times
                _vfeats.append(torch.cat((_hs_1[_i, :, :], _hs_2[_i, :, :]), dim=-1))
        decoder_video_rnn = torch.stack(_vfeats)  # B*Choices, seq_len, 512

        return decoder_video_rnn, _hn_vid1, _cn_vid1, _hn_vid2, _cn_vid2

    def forward_vid_mrm(self, features, batch_size, video_lengths, vid_model1, vid_model2,
                        _hn_1=None, _hn_2=None, _cn_1=None, _cn_2=None):
        if _hn_1 is None:
            _hn_1 = self.init_hidden(batch_size)
        if _hn_2 is None:
            _hn_2 = self.init_hidden(batch_size)
        if _cn_1 is None:
            _cn_1 = self.init_hidden(batch_size)
        if _cn_2 is None:
            _cn_2 = self.init_hidden(batch_size)

        assert _hn_1.shape[1] == _cn_1.shape[1] and _hn_1.shape[1] == _hn_2.shape[1] and _hn_2.shape[1] == _cn_2.shape[1], \
            "wrong batch sizes hn1 {} cn1 {} hn2 {} cn2 {} ".format(
                _hn_1.shape, _cn_1.shape, _hn_2.shape, _cn_2.shape
            )
        # each video is tested with each of the h_i, c_i computed with the different q+a_i
        _vfeats = []
        for _i in range(batch_size):
            for _c in range(5):  # repeat same video feats 5 (=num_choices) times
                _vfeats.append(features[_i, :, :])
        _vfeats = torch.stack(_vfeats)

        assert _hn_1.shape[1] == _vfeats.shape[0] and features.shape[0] * 5 == _vfeats.shape[0], \
            "wrong batch size hn1 {} feats {} vfeats {}".format(_hn_1.shape, features.shape, _vfeats.shape)
        if len(video_lengths) == batch_size and _vfeats.shape[0] == batch_size*5:
            _video_lengths = []
            for _j in range(batch_size):
                for _c in range(5):  # repeat same video feats 5 (=num_choices) times
                    _video_lengths.append(video_lengths[_j])

        assert len(_video_lengths) == _vfeats.shape[0], "wrong video lengrhs size"
        _packed = torch.nn.utils.rnn.pack_padded_sequence(_vfeats, _video_lengths,
                                                          batch_first=True, enforce_sorted=False)
        _hs_1, (_hn_vid1, _cn_vid1) = vid_model1(_packed, (_hn_1, _cn_1))
        _hs_2, (_hn_vid2, _cn_vid2) = vid_model2(_hs_1, (_hn_2, _cn_2))
        """_vfeats = []
        for _i in range(batch_size):
            for _c in range(5):  # repeat same video feats 5 (=num_choices) times
                _vfeats.append(torch.cat((_hs_1[_i, :, :], _hs_2[_i, :, :]), dim=-1))
        decoder_video_rnn = torch.stack(_vfeats)  # B*Choices, seq_len, 512"""
        _hs_1, _ = torch.nn.utils.rnn.pad_packed_sequence(_hs_1, batch_first=True)
        _hs_2, _ = torch.nn.utils.rnn.pad_packed_sequence(_hs_2, batch_first=True)
        decoder_video_rnn = torch.cat((_hs_1, _hs_2), -1)  # B*Choices, seq_len, 512
        return decoder_video_rnn, _hs_1, _hs_2, _hn_vid1, _cn_vid1, _hn_vid2, _cn_vid2

    def mm_module(self, video_lens, qa_lens, batch_size, num_choices, tmp, mem_vid, mem_txt, loop=3):
        # mem_{vid,txt} [B*Choices, {seq_len,n_words}, H]
        batch_size = mem_vid.shape[0]
        sm_q1 = self.init_hidden(batch_size)
        cm_q1 = self.init_hidden(batch_size)
        sm_q2 = self.init_hidden(batch_size)
        cm_q2 = self.init_hidden(batch_size)
        mm_oo = self.drop_keep_prob_final_att_vec(self.activation_functions(self.hidden_enc_1(tmp)))
        for _ in range(loop):
            _, (sm_q1, cm_q1) = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            _, (sm_q2, cm_q2) = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))
            # {s,c}m_q{1,2} [1, B*Choices, H]
            mm_o1 = self.mm_att(sm_q2, mem_vid, mem_txt, video_lens, qa_lens, batch_size // num_choices, num_choices)
            mm_o2 = torch.cat((sm_q2, mm_o1.unsqueeze(0)), dim=-1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_enc_2(mm_o2)))

        smq = torch.cat((sm_q1, sm_q2), dim=-1)
        return smq

    def mm_module_v1(self, svt_tmp, memory_ram_vid, memory_ram_txt, loop=3):

        sm_q1, sm_q2, cm_q1, cm_q2 = self.init_hiddens()
        mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_enc_1(svt_tmp)))

        for _ in range(loop):
            sm_q1, cm_q1 = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            sm_q2, cm_q2 = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))

            mm_o1 = self.mm_att(sm_q2, memory_ram_vid, memory_ram_txt)
            mm_o2 = torch.cat((sm_q2, mm_o1), dim=1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_enc_2(mm_o2)))

        smq = torch.cat((sm_q1, sm_q2), dim=1)
        return smq

    def forward_mem(self, video_lens, qa_lens, batch_size, num_choices, mrm_vid, mrm_txt, fts_2a, fts_2m, h2_a, h2_m, h2_txt, nImg, nQws):
        """mem_vid = []
        # fts2a torch.Size([10, 10, 256])
        #    h2_txt torch.Size([10, 12, 256])
        for _b in range(batch_size):
            for _c in range(num_choices):
                mem_vid.append(mrm_vid(fts_2a[_b*num_choices+_c, :video_lens[_b], :],
                                   fts_2m[_b*num_choices+_c, :video_lens[_b], :],
                                   video_lens[_b]))"""
        #mem_vid = torch.stack(_mv).squeeze(1).unsqueeze(0)
        mem_vid = mrm_vid(fts_2a, fts_2m, nImg, self.device)

        """for _bs in range(2):
            for _ncand in range(5):
                print("mem_vid batch{} cand{} {} {}".format(
                    _bs, _ncand, mem_vid[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(mem_vid[(_bs * _ncand) + _ncand, :])
                ))"""

        """mem_txt = []
        for _b in range(batch_size):
            for _c in range(num_choices):
                mem_txt.append(mrm_txt(h2_txt[_b*num_choices+_c, :qa_lens[_b*num_choices+_c], :],
                                   qa_lens[_b * num_choices + _c]))
        #mem_txt = torch.stack(mem_txt).squeeze(1).unsqueeze(0)"""
        mem_txt = mrm_txt(h2_txt, nQws, self.device)

        """for _bs in range(2):
            for _ncand in range(5):
                print("mem_txt batch{} cand{} {} {}".format(
                    _bs, _ncand, mem_txt[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(mem_txt[(_bs * _ncand) + _ncand, :])
                ))"""

        _tmp = torch.cat((h2_a, h2_m), -1)  # (B, 1, 512) last hidden states of the 2nd layer lstms(app&mot)
        #print("_tmp", _tmp.shape, torch.sum(_tmp, 2))
        out = self.mm_module(video_lens, qa_lens, batch_size, num_choices, _tmp, mem_vid, mem_txt)
        """out = []
        for _b in range(batch_size):
            for _c in range(num_choices):
                out.append(self.mm_module_v1(_tmp[:, _b*num_choices+_c, :],
                                             mem_vid[_b*num_choices+_c][:video_lens[_b], :],
                                             mem_txt[_b*num_choices+_c][:qa_lens[_b*num_choices+_c], :]))
        out = torch.stack(out).squeeze(1).unsqueeze(0)"""

        """for _bs in range(2):
            for _ncand in range(5):
                print("mem out batch{} cand{} {} {}".format(
                    _bs, _ncand, out[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(out[(_bs * _ncand) + _ncand, :])
                ))"""

        return out

    def forward_dec_mrm(self, vid_att_a, vid_att_m, mem_fts):
        f_va = self.activation_functions(self.linear_att_a(vid_att_a))
        f_vm = self.activation_functions(self.linear_att_m(vid_att_m))
        f_mm = self.activation_functions(self.linear_mem(mem_fts))

        """for _bs in range(2):
            for _ncand in range(5):
                print("f_va batch{} cand{} {} {}".format(
                    _bs, _ncand, f_va[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(f_va[(_bs * _ncand) + _ncand, :])
                ))

        for _bs in range(2):
            for _ncand in range(5):
                print("f_vm batch{} cand{} {} {}".format(
                    _bs, _ncand, f_vm[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(f_vm[(_bs * _ncand) + _ncand, :])
                ))"""

        """for _bs in range(2):
            for _ncand in range(5):
                print("f_mm batch{} cand{} {} {}".format(
                    _bs, _ncand, f_mm[(_bs * _ncand) + _ncand, :].shape,
                    torch.sum(f_mm[(_bs * _ncand) + _ncand, :])
                ))"""

        final = torch.cat((f_va.unsqueeze(0), f_vm.unsqueeze(0), f_mm), -1)
        return final

    def forward_dec_2(self, final_embed, batch_size):
        output3 = self.linear2(final_embed)
        #print("otputs", output3)
        outputs = output3.reshape(batch_size, 5)
        _, predictions = torch.max(outputs, 1)
        return outputs, predictions

    def accuracy(self, logits, targets):
        # print(logits.size(), targets.size(), logits.type(), targets.type())
        # print(logits,targets)
        # targets = targets.int()
        correct = torch.sum(logits.eq(targets)).float()
        # print(correct,targets.size(0))
        # print(correct, targets.size(0), correct * 100.0 / targets.size(0))
        return correct * 100.0 / targets.size(0)


class TGIFBenchmark(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix, activation_function, embed_tech="glove",
                 answer_vocab_size=None, max_len=20, dropout=0.2, embed_layer=-1, additional_tasks=""):
        """Set the hyper-parameters and build the layers."""
        super(TGIFBenchmark, self).__init__()

        # text input size
        self.text_embed_size = text_embed_size  # should be 300

        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim  # should be 7

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.TpAtt = TemporalAttentionModule(hidden_size * 2, hidden_size)

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)
        self.embed_tech = embed_tech

        if embed_tech in ["glove", "glove_frozen"]:
          self.embed = nn.Embedding(vocab_size, text_embed_size)
          if embed_tech == "glove_frozen":
              self.embed = self.embed.eval()
        elif embed_tech in ["bert", "xlm"]:
            if embed_tech == "bert":
                model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
                config = BertConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
                self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
            else:
                model_class, tokenizer_class, pretrained_weights = (XLMModel, XLMTokenizer, 'xlm-mlm-en-2048')
                config = XLMConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
                self.tokenizer = XLMTokenizer.from_pretrained(pretrained_weights)
            self.embed = model_class.from_pretrained(pretrained_weights, config=config)
            self.embed = self.embed.eval()
            self.embed_layer = embed_layer

        # ---v additional tasks v---
        self.additional_tasks = additional_tasks.split(",") if additional_tasks != "" else []
        if len(self.additional_tasks) > 0:
            if "qtclassif" in self.additional_tasks:
                from dataset_pytorch import _qtypes_map
                self.qt_classifier = nn.Linear(hidden_size*2, len(_qtypes_map))  # video -> question type
                self.softmax_qtc = nn.Softmax()
        # ---^ additional tasks ^---

        self.linear = nn.Linear(hidden_size*2, hidden_size*2)
        self.activation_functions = activation_function()
        self.linear2 = nn.Linear(hidden_size*2, 1)

        self.lstm_video_1 = nn.LSTM(feat_channel, hidden_size, batch_first=True)
        self.lstm_video_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_text_1 = nn.LSTM(text_embed_size, hidden_size, batch_first=True)
        self.lstm_text_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.max_len = max_len

        if embed_tech == "glove":
            self.init_weights(word_matrix)

    def init_weights(self, word_matrix):
        """Initialize weights."""
        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t, s_t2, c_t, c_t2

    def forward(self, data_dict):
        ret = self.forward_action(data_dict)
        return ret

    def forward_action(self, data_dict):
        video_features, questions, question_lengths, num_mult_choices, answers, answers_lengths = data_dict['video_features'],\
                                                                                                  data_dict['question_words'], data_dict['question_lengths'],\
                                                                                                  data_dict['num_mult_choices'], data_dict['answers'], \
                                                                                                  data_dict['answers_lengths']

        device = self.device  #'cuda:0'
        batch_size = len(questions)

        if self.embed_tech in ["bert", "xlm"]:
            qtexts = data_dict["qtexts"]
            atexts = data_dict["atexts"]
            _rep_qts = []
            _qlens = []
            for qt in qtexts:
                _rep_qts += [qt] * 5
                _qlens += [len(qt.split())] * 5
            _atexts = []
            _alens = []
            for i in range(len(qtexts)):
                for _k in atexts.keys():
                    _atexts.append(atexts[_k][i])
            _alens = [len(at.split()) for at in _atexts]
            qa_lens = [ql + al + 2 for (ql, al) in zip(_qlens, _alens)]  # +3 to take into account [CLS], [SEP]
            questions = self.tokenizer(_rep_qts, _atexts, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
        elif self.embed_tech in ["glove", "glove_frozen"]:
            max_m_questions = max(question.shape[1] for question in questions)
            questions_tensor = []

            for question in questions:
                npad = ((0, 0), (0, max_m_questions - question.shape[1]))
                question_single = np.pad(question.cpu(), pad_width=npad, mode='constant')
                questions_tensor.append(question_single)

            questions_tensor = torch.tensor(questions_tensor).to(device)
            questions_tensor = questions_tensor.reshape(questions_tensor.shape[0] * questions_tensor.shape[1],
                                                        questions_tensor.shape[2])
            qtexts = data_dict["qtexts"]
            _qlens = []
            for qt in qtexts:
                _qlens += [len(qt.split())] * 5
            _alens = [al.item() for al in torch.flatten(answers_lengths).cpu()]
            qa_lens = [ql + al for (ql, al) in zip(_qlens, _alens)]

        max_n_vgg = max(video.shape[1] for video in video_features[0])
        max_n_c3d = max(video.shape[1] for video in video_features[1])

        assert max_n_c3d == max_n_vgg

        #v_lens = [v.shape[1] for v in video_features[0]]
        app_feats = torch.nn.utils.rnn.pad_sequence([v.squeeze(0) for v in video_features[0]], batch_first=True)
        mot_feats = torch.nn.utils.rnn.pad_sequence([v.squeeze(0) for v in video_features[1]], batch_first=True)
        """for video in video_features[0]:
            npad = ((0, 0), (0, max_n - video.shape[1]), (0,0))
            vgg_single = np.pad(video.cpu(), pad_width=npad, mode='constant')
            vgg.append(vgg_single)
            mask_single = video.shape[1]*[1] + (max_n - video.shape[1]) * [0]
            mask_video.append(mask_single)

        for video in video_features[1]:
            npad = ((0, 0), (0, max_n - video.shape[1]), (0,0))
            c3d_single = np.pad(video.cpu(), pad_width=npad, mode='constant')
            c3d.append(c3d_single)

        vgg = torch.tensor(vgg).to(device)
        c3d = torch.tensor(c3d).to(device)"""

        features = torch.cat([app_feats,mot_feats], dim=-1)  # shape B,N,8192

        if self.embed_tech == "glove":
            features_questions = self.embed(questions_tensor)
        elif self.embed_tech == "glove_frozen":
            with torch.no_grad():
                features_questions = self.embed(questions_tensor)
        elif self.embed_tech in ["bert", "xlm"]:
            with torch.no_grad():
                bert_outputs = self.embed(**questions)
            features_questions = bert_outputs[0]  # [B*Choices,seq_len,768]
        else:
            assert False, "forgot to compute text feats"

        _hs_1, (_hn_1, _cn_1) = self.lstm_video_1(features)
        _hs_2, (_hn_2, _cn_2) = self.lstm_video_2(_hs_1)
        _vfeats = []
        _rep_hn1, _rep_cn1 = [], []
        _rep_hn2, _rep_cn2 = [], []
        for _i in range(batch_size):
            for _c in range(5):  # repeat same video feats 5 (=num_choices) times
                _vfeats.append(torch.cat((_hs_1[_i, :, :], _hs_2[_i, :, :]), dim=-1))
        decoder_video_rnn = torch.stack(_vfeats)  # B*Choices, seq_len, 512
        # decoder_video_rnn = torch.cat((_hs_1, _hs_2), -1)  # B*Choices, seq_len, 512
        for _i in range(batch_size):
            for _c in range(5):
                # also repeating the same hidden/cell state of the video lstm
                _rep_hn1.append(_hn_1[:, _i, :])
                _rep_hn2.append(_hn_2[:, _i, :])
                _rep_cn1.append(_cn_1[:, _i, :])
                _rep_cn2.append(_cn_2[:, _i, :])
        _rep_hn1 = torch.cat(_rep_hn1, 0).unsqueeze(0)
        _rep_cn1 = torch.cat(_rep_cn1, 0).unsqueeze(0)
        _rep_hn2 = torch.cat(_rep_hn2, 0).unsqueeze(0)
        _rep_cn2 = torch.cat(_rep_cn2, 0).unsqueeze(0)  # 1, B*Choices, 256

        # using the hidden and cell state computed from the video part
        _hs_1, (_hn, _cn) = self.lstm_text_1(features_questions, (_rep_hn1, _rep_cn1))
        _hs_2, (_hn, _cn) = self.lstm_text_2(_hs_1, (_rep_hn2, _rep_cn2))
        """_tfeats = []
        for _i, _l in enumerate(qa_lens):
                _hs1_last = _hs_1[_i][_l - 1].unsqueeze(0)
                _hs2_last = _hs_2[_i][_l - 1].unsqueeze(0)
                _tfeats.append(torch.cat((_hs1_last, _hs2_last), dim=-1))"""
        decoder_text_rnn = torch.cat((_hs_1[:, -1, :], _hs_2[:, -1, :]), -1)  #torch.cat(_tfeats, 0)  # B*Choices, 512

        video_att = self.TpAtt(decoder_video_rnn, decoder_text_rnn)  # (B*Choices, 512)

        # ---- additional tasks ----
        if len(self.additional_tasks) > 0 and "qtclassif" in self.additional_tasks:
            qt_logits = self.qt_classifier(video_att)
            qt_classes = self.softmax_qtc(qt_logits)
        # ---- additional tasks ----

        output1 = self.linear(video_att)
        output2 = self.activation_functions(output1) * decoder_text_rnn
        output3 = self.linear2(output2)

        outputs = output3.reshape(batch_size, 5)
        _, predictions = torch.max(outputs, 1)

        _finaloutput = {"outputs": outputs,
                        "predictions": predictions}

        # ---- additional tasks ----
        if len(self.additional_tasks) > 0 and "qtclassif" in self.additional_tasks:
            _finaloutput["qt_classes"] = qt_classes
        # ---- additional tasks ----

        return _finaloutput

    def accuracy(self, logits, targets):
        # print(logits.size(), targets.size(), logits.type(), targets.type())
        # print(logits,targets)
        # targets = targets.int()
        correct = torch.sum(logits.eq(targets)).float()
        # print(correct,targets.size(0))
        # print(correct, targets.size(0), correct * 100.0 / targets.size(0))
        return correct * 100.0 / targets.size(0)

