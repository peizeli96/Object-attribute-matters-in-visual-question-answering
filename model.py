import torch
import torch.nn as nn
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from src.counting import Counter
from utils import tfidf_loading
BertLayerNorm = torch.nn.LayerNorm
from transformers import LxmertModel
from cross_attention import cross_attention1, cross_attention2
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm
from contrastive_loss import ContrastiveLoss
criterion_graph = ContrastiveLoss(measure='dot', margin=0.01, max_violation=False)
import os




class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class VisualGraphAggregator(nn.Module):
    def __init__(self, is_proj_dim=256, obj_dim=512, sem_dim=512):
        super(VisualGraphAggregator, self).__init__()
        self.im_sem_embed_dim = is_proj_dim
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, is_proj_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, is_proj_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(is_proj_dim * 2, obj_dim)


    def forward(self, vis, sem):

        vis_proj = self.im_embed_is(vis)  
        sem_proj = self.sem_embed_is(sem)  
        i_att = torch.matmul(vis_proj, sem_proj.permute(0, 2, 1))  
        vis_sem = F.softmax(i_att, dim=2)
        att = torch.matmul(vis_sem, sem_proj)
        i_new = self.im_sem_combined(torch.cat((vis_proj, att), 2))
        return vis + i_new

class SemanticGraphAggregator(nn.Module):
    def __init__(self, im_sem_embed_dim=512, obj_dim=768, sem_dim=768):
        super(SemanticGraphAggregator, self).__init__()
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, im_sem_embed_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, im_sem_embed_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(im_sem_embed_dim * 2, sem_dim)


    def forward(self, vis, emb):
        vis_proj = self.im_embed_is(vis)  
        emb_proj = self.sem_embed_is(emb)  
        similarity = torch.matmul(emb_proj, vis_proj.permute(0, 2, 1))
        att = F.softmax(similarity, dim=2)  
        i_att = torch.matmul(att, vis_proj)  
        combine = self.im_sem_combined(torch.cat((emb_proj, i_att), 2)) 
        return combine+emb


class Lxmert_Model(nn.Module):
    def __init__(self,dataset,opt):
        super(Lxmert_Model, self).__init__()
        self.lxrt_encoder = LxmertModel.from_pretrained('lxmert')
        self.logit_fc = nn.Sequential(nn.Linear(768, 768 * 2), nn.GELU(), BertLayerNorm(768 * 2, eps=1e-12), nn.Linear(768 * 2, dataset.num_ans_candidates))

    def forward(self, q, v, b):

        bert_output = self.lxrt_encoder(q, v, b)
        q_emb = bert_output[0]
        v_output = bert_output[1]
        cls = bert_output[2]
        return q_emb, v_output, cls, self.logit_fc(cls)

    def dim(self):
        return 768




class Visual_Attention(nn.Module):
    def __init__(self, dim_image_feats, dim_att_lstm, nb_hidden):
        super(Visual_Attention,self).__init__()
        self.fc_image_feats = nn.Linear(dim_image_feats, nb_hidden, bias=False)
        self.fc_att_lstm = nn.Linear(dim_att_lstm, nb_hidden, bias=False)
        self.act_tan = nn.Tanh()
        self.fc_att = nn.Linear(nb_hidden, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_feats, h1):

        att_lstm_emb = self.fc_att_lstm(h1)
        image_feats_emb = self.fc_image_feats(image_feats)
        similarity = torch.matmul(image_feats_emb, att_lstm_emb.permute(0, 2, 1))
        sim,_ = torch.topk(similarity, dim=-1, k=1)
        att = F.softmax(sim, dim=1)
        weighted_feats = att * image_feats
        return weighted_feats.sum(dim=1)







class Model(nn.Module):
    def __init__(self, dataset, opt):
        super(Model, self).__init__()

        self.lxmert_encoder = Lxmert_Model(dataset, opt)                

        self.vis_attri_graph = VisualGraphAggregator()
        self.atten_1 = Visual_Attention(dim_image_feats=512, dim_att_lstm=512, nb_hidden=1024)
        self.atten_2 = Visual_Attention(dim_image_feats=512, dim_att_lstm=512, nb_hidden=1024)

        # # Initial attribute embedding
        self.attribute_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, opt.op)
        self.ofa_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, opt.op)
        self.blip2_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, opt.op)

        # self.word_embw = QuestionEmbedding(300 if 'c' not in opt.op else 600, opt.num_hid, 1, False, .0)

        self.blip_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, opt.op)
        self.cap_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, opt.op)
        self.cap_embw = QuestionEmbedding(300 if 'c' not in opt.op else 600, opt.num_hid, 1, False, .0)

        if hasattr(opt, 'tfidf'):
            self.attribute_emb = tfidf_loading(opt.tfidf, self.attribute_emb, opt, 'data') 
            self.cap_emb = tfidf_loading(opt.tfidf, self.cap_emb, opt, 'data')
            self.ofa_emb = tfidf_loading(opt.tfidf, self.ofa_emb, opt, 'data')
            self.blip_emb = tfidf_loading(opt.tfidf, self.blip_emb, opt, 'data')           
            self.blip2_emb = tfidf_loading(opt.tfidf, self.blip2_emb, opt, 'data') 

     
        self.classifier = SimpleClassifier(512 , 512 * 2, dataset.num_ans_candidates, opt)


        self.proj_cls = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 512))
        self.proj_lan = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 512))
        self.proj_vis = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 512))

        self.proj_blip = nn.Sequential(nn.Linear(600, 512), nn.ReLU())
        self.proj_blip2 = nn.Sequential(nn.Linear(600, 512), nn.ReLU())
        self.proj_ofa =  nn.Sequential(nn.Linear(600, 512), nn.ReLU())





    def forward(self, v, b, q, kb_token, attribute_token, blip_token, blip2_token, ofa_token):

        l_output, v_output, cls, lxmert_logit = self.lxmert_encoder(q, v, b)
        lan_feat = self.proj_lan(l_output) 
        vis_feat = self.proj_vis(v_output)


        cap_featt = self.cap_emb(kb_token)
        cap_feat = self.cap_embw.forward_all(cap_featt)
        cap_feature = self.proj_cap(cap_feat)
        cap_lan_feature = self.self_cap_atten(torch.cat((cap_feature, lan_feat),1))


        attribute_feat = self.attribute_emb(attribute_token)
        attribute_feature = self.proj_attribute(attribute_feat)
        attri_vis_feature = self.self_atten_tri(torch.cat((attribute_feature, vis_feat),1))
     
        ################ 答案嵌入
        blip_emb = self.blip_emb(blip_token)
        blip_feat = self.proj_blip(blip_emb) 

        blip2_emb = self.blip2_emb(blip2_token)
        blip2_feat = self.proj_blip(blip2_emb)        

        ofa_emb = self.ofa_emb(ofa_token)
        ofa_feat = self.proj_ofa(ofa_emb) 

        ans_feat = torch.cat((blip_feat, blip2_feat, ofa_feat),1) 

        ans_ques_con = self.self_ans_lan_atten(torch.cat((ans_feat, lan_feat),1))
        ans_feature = ans_ques_con[:,:3,:]       
        lan_ans_feat = self.ques_kb(ans_feature, lan_feat)
        

        attri_feat_vis = self.vis_attri_graph(vis_feat, attri_vis_feature)
        obj_emb_relation = self.emb_attri_graph(attri_feat_vis, attri_vis_feature) 



        kb_feature = self.lan_kb(cap_lan_feature, lan_feat)[:,:10,:]

        cus_1 = self.atten_1(obj_emb_relation, lan_ans_feat)
        cus_2 = self.atten_2(kb_feature, lan_ans_feat)

        ans_logit = self.classifier(cus_1 + cus_2)    
        norm_cls = F.normalize(cus_1, dim=-1)
        norm_head_relation = F.normalize(cus_2, dim=-1)
        ans_loss = ContrastiveLossLoss(norm_cls, norm_head_relation) 

        return lxmert_logit, ans_logit, ans_loss    
