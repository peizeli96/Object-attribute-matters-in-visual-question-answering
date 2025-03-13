
import torch.nn as nn
import torch 
import math
BertLayerNorm = torch.nn.LayerNorm
from CrossattLayer import CrossattLayer, BertCrossattLayer




class Cross_attetion1(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_cross_1 = CrossattLayer()
        self.att_cross_2 = CrossattLayer()
        self.att_cross_3 = CrossattLayer()
        self.att_cross_4 = CrossattLayer()
        self.att_cross_5 = CrossattLayer()


    def forward(self, input_tensor):
        input_tensor_1 = self.att_cross_1(input_tensor, input_tensor)
        input_tensor_2 = self.att_cross_2(input_tensor_1, input_tensor_1)
        input_tensor_3 = self.att_cross_3(input_tensor_2, input_tensor_2)
        input_tensor_4 = self.att_cross_4(input_tensor_3, input_tensor_3)
        out = self.att_cross_5(input_tensor_4, input_tensor_4)
        
        return out    



class Cross_attetion2(nn.Module):
    def __init__(self):
        super(Cross_attetion2, self).__init__()
        self.v_pro = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.l_pro = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 256))

        self.vis_attention = BertCrossattLayer()
        self.text_attention = BertCrossattLayer()
        self.self_attention_1 = BertselfattLayer()
        self.dense = nn.Linear(512,512)
        self.activation = nn.Tanh()

        
    def forward(self, V_feature, attr_feature):

        V_feature = self.v_pro(V_feature)
        attr_feature = self.l_pro(attr_feature)

        v_attention_out = self.vis_attention(V_feature, attr_feature, attr_feature)

        attr_feature_out = self.text_attention(attr_feature, V_feature, V_feature)

        v_out = torch.cat((V_feature, v_attention_out),2)

        attr_out = torch.cat((attr_feature, attr_feature_out),2)

        feature_out = torch.cat((v_out, attr_out),1)

        feature_out = self.self_attention_1(feature_out, feature_out)

        # pool_out = self.activation(self.dense(lang_feat5))
        return feature_out
    




