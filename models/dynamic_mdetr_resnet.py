import time
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from .vl_encoder import build_vl_encoder
from utils.box_utils import xywh2xyxy
from utils.misc import NestedTensor, merge_nested_tensors
import math


def load_category_mapping(file_path):
      """Load category mapping from a text file."""
      with open(file_path, 'r') as f:
          categories = f.read().splitlines()
      category_to_idx = {category: idx for idx, category in enumerate(categories)}
      return category_to_idx, categories
  
def compute_contrastive_loss(batch_size,
                             num_templates,
                             category,
                             tem_cats,
                             vl_feat,
                             template_combined_src):
    
    target_feats = F.normalize(vl_feat, dim=-1)  # Use target features
    template_feats = F.normalize(template_combined_src, dim=-1)  # Use template combined features

    # print(target_feats.size()) #torch.Size([440, 8, 256])
    target_feats = target_feats.mean(dim=0, keepdim=True).repeat_interleave(num_templates, dim = 0) # (15,8,256)
    # print(target_feats.permute(1, 0, 2).size()) #torch.Size([8, 15, 256])

    # Compute similarity matrix
    # target_feats.permute(1, 0, 2) : torch.Size([8, 15, 256])  / template_feats.permute(1, 2, 0) : torch.Size([8, 256, 15]) 
    sim_matrix = torch.matmul(target_feats.permute(1, 0, 2), template_feats.permute(1, 2, 0))  # (bs, num_templates, num_templates)
    '''
    sim_matrix[i, j, k]
    : i번째 배치에서 j번째 타겟 특징(임베딩)과 k번째 템플릿 특징(임베딩) 간의 유사도 값
    '''
    
    # Positive and negative mask calculation
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)  # Initialize positive mask
    neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)  # Initialize negative ma
    for i in range(batch_size):
        for k in range(num_templates):  # 각 타겟의 k번째 템플릿에 대해
            for j in range(num_templates):  # 각 타겟의 j번째 템플릿과 비교
                # target의 카테고리와 템플릿의 카테고리 비교
                if category[i] == tem_cats[i][j]:
                    pos_mask[i, k, j] = 1  # 동일한 카테고리일 경우 positive mask 설정
                    neg_mask[i, k, j] = 0  # negative mask에서는 제외
                else:
                    pos_mask[i, k, j] = 0  # 카테고리가 다를 경우 positive mask에서는 제외
                    neg_mask[i, k, j] = 1  # negative mask에 포함

    # Contrastive loss calculation
    pos_loss = (pos_mask * F.logsigmoid(sim_matrix)).sum()
    neg_loss = (neg_mask * F.logsigmoid(-sim_matrix)).sum()
    contrastive_loss = -(pos_loss + neg_loss) / (pos_mask.sum() + neg_mask.sum() + 1e-8)  # Avoid division by zero


class CrossAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttentionModule, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        
    def forward(self, text_features, visual_features, text_mask=None, visual_mask=None):
        # Text to Visual attention
        text_to_visual, _ = self.cross_attn(text_features, visual_features, visual_features, key_padding_mask=visual_mask)
        
        # Visual to Text attention
        visual_to_text, _ = self.cross_attn(visual_features, text_features, text_features, key_padding_mask=text_mask)
        
        return text_to_visual, visual_to_text

class DynamicMDETR(nn.Module):
    def __init__(self, args):
        super(DynamicMDETR, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len
        self.uniform_grid = args.uniform_grid
        self.uniform_learnable = args.uniform_learnable
        self.different_transformer = args.different_transformer
        
        category_file_path = args.category_file_path
        self.category_to_idx, self.categories = load_category_mapping(category_file_path)

        # Add pseudo-class embedding (learnable token)
        self.pseudo_class_embedding = nn.Embedding(80, hidden_dim)
        self.contrastive_loss = args.contrastive_loss

        # Add context embedding layer
        # self.context_embedding = nn.Linear(hidden_dim, hidden_dim)  # For learning context from text and visual

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token 
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim) 
        self.vl_encoder = build_vl_encoder(args)
       

        self.vl_pos_embed_template = nn.Embedding(num_total*5, hidden_dim) 

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        
    
        # Cross-Attention Module
        self.use_cross_attention = args.use_cross_attention
        if self.use_cross_attention:
            self.cross_attention = CrossAttentionModule(d_model=hidden_dim, n_heads=8)


        for param in self.vl_pos_embed.parameters():
            param.requires_grad = True
        for param in self.vl_encoder.parameters():
            param.requires_grad = False
        for param in self.visumodel.parameters():
            param.requires_grad = False
        for param in self.textmodel.parameters():
            param.requires_grad = False
        for param in self.visu_proj.parameters():
            param.requires_grad = False
        for param in self.text_proj.parameters():
            param.requires_grad = False

        # Sampling relevant
        self.visual_feature_map_h = 20
        self.visual_feature_map_w = 20
        self.in_points = args.in_points
        self.stages = args.stages

        self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
        self.update_sampling_queries = nn.ModuleList(
            [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

        self.init_reference_point = nn.Embedding(1, 2)
        self.init_sampling_feature = nn.Embedding(1, hidden_dim)

        self.init_weights()

        if self.different_transformer:
            self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
        else:
            self.vl_transformer = build_vl_transformer(args)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def init_weights(self):
        nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
        nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
        self.init_reference_point.weight.requires_grad = False

        for i in range(self.stages):
            nn.init.zeros_(self.offset_generators[i].weight)
            nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
        if not self.uniform_learnable:
            self.offset_generators[0].weight.requires_grad = False
            self.offset_generators[0].bias.requires_grad = False

    def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
        bs, channel = sampling_query.shape
        if self.uniform_grid:
            if stage != 0:
                xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
                sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
            else:
                sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
        else:
            xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
            sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
        feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w)  # (bs, channel, h, w)
        pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w)  # (bs, channel, h, w)

        # [0,1] to [-1,1]
        sampled_points = (2 * sampled_points) - 1

        sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
                                       align_corners=False).squeeze(-1)  # (bs, channel, in_points)
        pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1)  # (bs, channel, in_points)

        return sampled_features, pe

    def forward(self, img_data, text_data, tem_imgs, tem_txts, category, tem_cats):
            B, num_templates = img_data.tensors.shape[0], tem_imgs[0].tensors.shape[0]

            # 1. Target

            # 1.1 Encoding
            
            # 1.1.1 Visual Encoder
            # visual backbone
            out, visu_pos = self.visumodel(img_data)
            visu_mask, visu_src = out # (B, H*W), (N_v, B, channel)
            visu_src = self.visu_proj(visu_src)  # (N_v, B, channel)

            # 1.1.2 Language Encoder
            # language bert
            text_fea = self.textmodel(text_data)
            text_src, text_mask = text_fea.decompose() # (B, N_l, hidden_dim), (B, N_l)
            assert text_mask is not None
            # text_src: (bs, max_len, channel)
            text_mask = text_mask.flatten(1)  # (B, max_len)
            text_src = self.text_proj(text_src).permute(1, 0, 2)  # (max_len, B, channel)
            
            vl_src = torch.cat([visu_src, text_src], dim=0)

            # 1.1.4 Concat visual features and language features
            vl_mask = torch.cat([visu_mask, text_mask], dim=1)
            vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)

            # 1.2. Multimodal Transformer
            # 1.2.1 Multimodal Transformer Encoder
            if self.vl_encoder is not None:
                vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
            else:
                vl_feat = vl_src

            # 1.2.2 Split back to visual features and language features, use language features as queries
            visu_feat = vl_feat[:self.num_visu_token] # (H*W, B, channel)
            language_feat = vl_feat[self.num_visu_token:] # (max_len, B, channel)
            v_pos = vl_pos[:self.num_visu_token]
            l_pos = vl_pos[self.num_visu_token:]


            # 2. Template Data 처리 (Target과 동일한 방식)
            ## 템플릿 피처 결합 및 평균 풀링
            
            # (B * Num_templates, C, H, W)
            tem_imgs_tensors = merge_nested_tensors(tem_imgs)
            # (B * Num_templates, L)
            tem_txts_tensors = merge_nested_tensors(tem_txts)
            
            # 2.1 Visual Encoder for Template
            tem_out, tem_visu_pos = self.visumodel(tem_imgs_tensors)
            # (B * Num_templates, N_v), (N_v, B * Num_templates, hidden_dim)
            tem_visu_mask, tem_visu_src = tem_out 
            # (N_v, B * Num_templates, hidden_dim)
            tem_visu_src = self.visu_proj(tem_visu_src)
            # (N_v, B * Num_templates, hidden_dim)

            # 2.2 Language Encoder for Template
            tem_text_fea = self.textmodel(tem_txts_tensors)
            # (B * Num_templates, N_l, hidden_dim), (B * Num_templates, N_l)
            tem_text_src, tem_text_mask = tem_text_fea.decompose() 
            # (B * Num_templates, N_l)
            tem_text_mask = tem_text_mask.flatten(1) 
            # (N_l, B * Num_templates, hidden_dim)
            tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)  
            # (N_l, 1, hidden_dimm) 
            # 마스크 생성
            batch_times_num_templates = tem_visu_src.shape[1]
            pseudo_class_mask = torch.zeros((batch_times_num_templates, 1), device=tem_visu_mask.device)  # (1, 1)

            tem_pseudo_class_feats = []



            for i in range(B):
                for j in range(num_templates):
                    # 2.3 Pseudo-class embedding for Template
                    tem_category_idx = torch.tensor(self.category_to_idx[tem_cats[i][j]], device=img_data.tensors.device)
                    tem_pseudo_class_feat = self.pseudo_class_embedding(tem_category_idx).unsqueeze(0)  # (1, 256)
                    tem_pseudo_class_feats.append(tem_pseudo_class_feat)

            # Tensor를 cat으로 이어 붙임
            tem_pseudo_class_feat = torch.cat(tem_pseudo_class_feats, dim=0).unsqueeze(0)  # (1, num_templates, 256)

            # 2.4 Apply Cross-Attention for Template
            if int(self.use_cross_attention) == 1:
                template_text_to_visual, template_visual_to_text = self.cross_attention(tem_text_src, tem_visu_src, tem_text_mask, tem_visu_mask)
                template_vl_src = torch.cat([template_visual_to_text, template_text_to_visual], dim=0)
            else:
                template_vl_src = torch.cat([tem_visu_src, tem_text_src], dim=0)

            # 템플릿 피처 결합 (Cross-Attention 적용된 Visual Feature, Language Feature, Pseudo-class Embedding)
            # (N_v + N_l + 1, B * num_templates, hidden_dim)
            combined_template_feat = torch.cat([template_vl_src, tem_pseudo_class_feat], dim=0)

            # (B * num_templates, hidden_dim)
            combined_template_mask = torch.cat([tem_visu_mask, tem_text_mask, pseudo_class_mask], dim=1)
            
            # (B * num_templates, hidden_dim)
            combined_template_mask = torch.cat([tem_visu_mask, tem_text_mask, pseudo_class_mask],dim=1)
            
            # Average pooling을 사용하여 템플릿 피처를 하나의 피처로 통합
            # (1, B * num_templates, hidden_dim)
            template_combined_feat = combined_template_feat.mean(dim=0, keepdim=True) 
            # (B * num_templates, 1)
            template_combined_mask = (combined_template_mask == 1).any(dim=1, keepdim=True).float()  

            # 모든 배치에 대해 병합
            # (1, B * num_templates, hidden_dim) -> (B, num_templates, hidden_dim)
            template_combined_src = template_combined_feat.view(B, num_templates, -1)
            # (num_templates, B)
            template_combined_mask = template_combined_mask.view(num_templates, B)

            # (num_templates, B, hidden_dim)
            template_combined_src = template_combined_src.permute(1, 0, 2) 
            # (B, num_templates)
            template_combined_mask = template_combined_mask.permute(1, 0)
            
            ### 3. Contrastive Loss Calculation
            contrastive_loss = 0
            if self.contrastive_loss == 1 :
                contrastive_loss = compute_contrastive_loss(batch_size=B,
                                         num_templates=num_templates,
                                         category=category,
                                         tem_cats=tem_cats,
                                         vl_feat=vl_feat,
                                         template_combined_src=template_combined_src,
                                         )

            
            ### 4. Dynamic Multimodal Transformer Decoder
            sampling_query = self.init_sampling_feature.weight.repeat(B, 1)
            reference_point = self.init_reference_point.weight.repeat(B, 1)

            # language query와 multimodal prompt결합하여 새로운 query 생성 
            # (N_l + num_templates, B, hidden_dim)
            new_query = torch.cat([language_feat, template_combined_src], dim=0)

            # (B, new_query.shape[0], hidden_dim)
            positional_embedding = self.vl_pos_embed_template.weight[:new_query.shape[0]].unsqueeze(1).repeat(1, B, 1)
            
            for i in range(self.stages):
                # 2d adaptive pooling
                sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

                ## Text guided decoding with one-layer transformer encoder-decoder
                ## language_feat와 template_combined_src를 결합하여 사용

                if self.different_transformer:
                    '''
                    l_pos : Encoder에서 학습되는 Positional embedding중에 Language 부분만 떼온 거
                    positional embedding역할
                    template_combined_pos -> concat
                    
                    [l_pos, template_combined_pos] -> Positional embedding 역할이 되는 건지??
                    
                    Feature 다뽑고 Query + Template -> Encoder
                    Multimodal prompt concat???
                    
                    그대로 -> template_combined_pos이 알아서 잘 학습된다 ???
                    '''
                
                    vg_hs = self.vl_transformer[i](sampled_features, None, new_query, pe, torch.cat([text_mask, template_combined_mask], dim=1), positional_embedding)[0]
                    # forward(self, src, src_mask, target, pos_embed, tgt_mask, tgt_pos=None)
                    # self.encoder(src, src_key_padding_mask=src_mask, pos=pos_embed)
                    # self.decoder(tgt, memory,  tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask,pos=pos_embed, query_pos=tgt_pos)
                    # vg_memory = self.vl_transformer[i].encoder(sampled_features, None, pe)
                    # vg_hs = self.vl_transformer[i].decoder(torch.cat([language_feat, template_combined_src], dim=0) ,vg_memory, text_mask, None, pe, l_pos)
                else:
                    vg_hs = self.vl_transformer[i](sampled_features, None, new_query, pe, torch.cat([text_mask, template_combined_mask], dim=1), positional_embedding)[0]
                    # vg_memory = self.vl_transformer[i].encoder(sampled_features, None, pe)
                    # vg_hs = self.vl_transformer[i].decoder(torch.cat([language_feat, template_combined_src], dim=0) ,vg_memory, text_mask, None, pe, l_pos)

                language_feat = vg_hs[0]
                text_select = (1 - torch.cat([text_mask, template_combined_mask], dim=1) * 1.0).unsqueeze(-1)
                # print(text_select.size()) torch.Size([8, 481, 1])
                # text_select = (1 - text_mask * 1.0).unsqueeze(-1) torch.Size([8, 40, 1])
                text_select_num = text_select.sum(dim=1)
                vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num
                
                pred_box = self.bbox_embed(vg_hs).sigmoid()
                
                reference_point = pred_box[:, :2]
                sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

            return pred_box, contrastive_loss

        
      # B = img_data.tensors.shape[0]

      # # Category를 숫자로 변환
      # category_idx = torch.tensor([self.category_to_idx[cat] for cat in category], device=img_data.tensors.device)

      # # 1. Template 피처 먼저 학습 (모든 카테고리 포함)
      # template_visu_feats = []
      # template_text_feats = []
      # template_visu_masks = []
      # template_text_masks = []
      # template_category_idx = []

      # for i in range(bs):
      #     batch_template_visu_feats = []
      #     batch_template_text_feats = []
      #     batch_template_visu_masks = []
      #     batch_template_text_masks = []

      #     for j in range(tem_imgs[i].tensors.shape[0]):
      #         # Visual Encoder for Template
      #         tem_out, tem_visu_pos = self.visumodel(tem_imgs[i].tensors[j].unsqueeze(0))
      #         tem_visu_mask, tem_visu_src = tem_out  # Mask와 Visual Feature 분리
      #         tem_visu_src = self.visu_proj(tem_visu_src)
      #         batch_template_visu_feats.append(tem_visu_src)
      #         batch_template_visu_masks.append(tem_visu_mask)

      #         # Template category index 추가
      #         tem_category_idx = torch.tensor(self.category_to_idx[tem_cats[i][j]], device=img_data.tensors.device)
      #         template_category_idx.append(tem_category_idx)

      #         # Language Encoder for Template
      #         tem_txt_tensor = tem_txts[i].tensors[j].unsqueeze(0)
      #         tem_txt_mask = tem_txts[i].mask[j].unsqueeze(0)

      #         tem_text_fea = self.textmodel(NestedTensor(tem_txt_tensor, tem_txt_mask))
      #         tem_text_src, tem_text_mask = tem_text_fea.decompose()
      #         tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)
      #         batch_template_text_feats.append(tem_text_src)
      #         batch_template_text_masks.append(tem_text_mask.unsqueeze(0))

      #     template_visu_feats.append(torch.cat(batch_template_visu_feats, dim=0))
      #     template_visu_masks.append(torch.cat(batch_template_visu_masks, dim=1))
      #     template_text_feats.append(torch.cat(batch_template_text_feats, dim=0))
      #     template_text_masks.append(torch.cat(batch_template_text_masks, dim=1))

      # # 템플릿 피처를 [batch_size, num_tokens, feature_dim] 형식으로 병합
      # template_visu_src = torch.cat(template_visu_feats, dim=1)
      # template_visu_mask = torch.cat(template_visu_masks, dim=0)
      # template_text_src = torch.cat(template_text_feats, dim=1)
      # template_text_mask = torch.cat(template_text_masks, dim=0)

      # template_text_mask = template_text_mask.squeeze(1)
      # template_text_mask = template_text_mask.view(template_text_mask.size(0), -1)

      # # 템플릿 데이터 통과시키기
      # combined_template_src = torch.cat([template_visu_src, template_text_src], dim=0)
      # combined_template_mask = torch.cat([template_visu_mask, template_text_mask], dim=1)

      # # Template positional embedding
      # vl_pos_template = self.vl_pos_embed_template.weight.unsqueeze(1).repeat(1, bs, 1)

      # # Template 멀티모달 인코더 통과 (템플릿 정보 학습)
      # if self.vl_encoder is not None:
      #     template_feat = self.vl_encoder(combined_template_src, combined_template_mask, vl_pos_template)
      # else:
      #     template_feat = combined_template_src

      # # 2. 카테고리별 러너블 임베딩 생성 (평균 계산)
      # # 2. 템플릿의 카테고리별 learnable embedding 생성
      # template_category_feat = {}
      # for i, cat in enumerate(tem_cats):
      #     # tem_cats[i]가 리스트 형태라서 첫 번째 요소로 카테고리 이름을 추출
      #     cat_name = cat[0]
      #     cat_idx = self.category_to_idx[cat_name]
          
      #     if cat_name not in template_category_feat:
      #         # 만약 현재 카테고리 키가 없으면 빈 리스트 초기화
      #         template_category_feat[cat_name] = []

      #     # `template_feat`의 i번째 배치의 피처 추가 (각 카테고리별 피처들 리스트 추가)
      #     template_category_feat[cat_name].append(template_feat[:, i, :])

      # # 카테고리별 피처들을 평균 내어 카테고리별 learnable embedding 생성
      # for cat_name, feats in template_category_feat.items():
      #     # 텐서 리스트를 배치 크기 기준으로 병합 후 평균
      #     template_category_feat[cat_name] = torch.stack(feats, dim=1).mean(dim=1)


      # # 3. 타겟 카테고리에 해당하는 템플릿 임베딩 선택 및 결합
      # selected_template_feats = []
      # for cat in category_idx:
      #     if cat.item() in template_category_feat:
      #         selected_template_feats.append(template_category_feat[cat.item()])
      #     else:
      #         selected_template_feats.append(torch.zeros_like(template_category_feat[list(template_category_feat.keys())[0]]))
      # selected_template_feats = torch.stack(selected_template_feats).to(img_data.tensors.device)  # (batch_size, num_tokens, hidden_dim)

      # # 4. 타겟 데이터와 결합하여 멀티모달 인코더 통과
      # # 4.1 타겟 데이터 인코딩
      # out, visu_pos = self.visumodel(img_data)
      # visu_mask, visu_src = out
      # visu_src = self.visu_proj(visu_src)
      # vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

      # text_fea = self.textmodel(text_data)
      # text_src, text_mask = text_fea.decompose()
      # text_src = self.text_proj(text_src).permute(1, 0, 2)

      # v_pos = vl_pos[:self.num_visu_token]
      # l_pos = vl_pos[self.num_visu_token:self.num_visu_token + self.num_text_token]

      # # 4.2 Pseudo-class embedding for the target category
      # pseudo_class_feat = selected_template_feats.mean(dim=1).unsqueeze(0)  # (1, batch_size, hidden_dim)
      # pseudo_mask = torch.zeros(bs, 1).to(text_mask.device)

      # # Pseudo-class embedding과 타겟 데이터 결합
      # # print(visu_src.size())  # (num_tokens, batch_size, hidden_dim)
      # # print(text_src.size())  # (num_tokens, batch_size, hidden_dim)
      # # print(pseudo_class_feat.size())  # (1, batch_size, hidden_dim)
      # combined_target_src = torch.cat([visu_src, text_src, pseudo_class_feat], dim=0)
      # combined_target_mask = torch.cat([visu_mask, text_mask, pseudo_mask], dim=1)

      # # 4.3 타겟 및 Pseudo-class embedding 멀티모달 인코더 통과
      # if self.vl_encoder is not None:
      #     target_feat = self.vl_encoder(combined_target_src, combined_target_mask, vl_pos)
      # else:
      #     target_feat = combined_target_src

      # # 5. Context Embedding
      # visu_feat = target_feat[:self.num_visu_token]
      # language_feat = target_feat[self.num_visu_token:self.num_visu_token + self.num_text_token]
      # pseudo_class_feat = target_feat[-1]

      # combined_feat = torch.cat((visu_feat, language_feat), dim=0)
      # context_embed = self.context_embedding(combined_feat.mean(dim=0))

      # # 6. Dynamic Multimodal Transformer Decoder
      # sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
      # reference_point = self.init_reference_point.weight.repeat(bs, 1)

      # for i in range(self.stages):
      #     # Combine pseudo-class embedding, context, and sampling query
      #     combined_query = torch.cat((sampling_query, pseudo_class_feat, context_embed), dim=1)
      #     combined_query = self.sampleing_proj(combined_query)
      #     sampled_features, pe = self.feautures_sampling(combined_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

      #     if self.different_transformer:
      #         vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
      #     else:
      #         vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

      #     language_feat = vg_hs[0]
      #     text_select = (1 - text_mask * 1.0).unsqueeze(-1)
      #     text_select_num = text_select.sum(dim=1)
      #     vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num

      #     pred_box = self.bbox_embed(vg_hs).sigmoid()

      #     reference_point = pred_box[:, :2]
      #     sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

      # return pred_box




# def load_category_mapping(file_path):
#       """Load category mapping from a text file."""
#       with open(file_path, 'r') as f:
#           categories = f.read().splitlines()
#       category_to_idx = {category: idx for idx, category in enumerate(categories)}
#       return category_to_idx, categories


# class DynamicMDETR(nn.Module):
#     def __init__(self, args):
#         super(DynamicMDETR, self).__init__()
#         hidden_dim = args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.num_visu_token = int((args.imsize / divisor) ** 2)
#         self.num_text_token = args.max_query_len
#         self.uniform_grid = args.uniform_grid
#         self.uniform_learnable = args.uniform_learnable
#         self.different_transformer = args.different_transformer

#         # Category to index mapping from the file
#         category_file_path = '/content/drive/MyDrive/fsod/Dynamic-MDETR/datasets/coco_80.txt'
#         self.category_to_idx, self.categories = load_category_mapping(category_file_path)

#         # Add pseudo-class embedding (learnable token)
#         self.pseudo_class_embedding = nn.Embedding(80, hidden_dim)

#         # Add context embedding layer
#         self.context_embedding = nn.Linear(hidden_dim, hidden_dim)  # For learning context from text and visual

#         self.visumodel = build_detr(args)
#         self.textmodel = build_bert(args)

#         num_total = self.num_visu_token + self.num_text_token 
#         self.vl_pos_embed = nn.Embedding(num_total+ 1, hidden_dim) 
#         self.vl_encoder = build_vl_encoder(args)

#         self.vl_pos_embed_template =  nn.Embedding(num_total*5, hidden_dim) 

#         self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
#         self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

#         # Sampling relevant
#         self.visual_feature_map_h = 20
#         self.visual_feature_map_w = 20
#         self.in_points = args.in_points
#         self.stages = args.stages
#         self.sampleing_proj = nn.Linear(hidden_dim * 3, hidden_dim)

#         self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
#         self.update_sampling_queries = nn.ModuleList(
#             [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

#         self.init_reference_point = nn.Embedding(1, 2)
#         self.init_sampling_feature = nn.Embedding(1, hidden_dim)

#         self.init_weights()


#         if self.different_transformer:
#             self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
#         else:
#             self.vl_transformer = build_vl_transformer(args)

#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


#     def init_weights(self):
#         nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
#         nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
#         self.init_reference_point.weight.requires_grad=False

#         for i in range(self.stages):
#             nn.init.zeros_(self.offset_generators[i].weight)
#             nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
#         if not self.uniform_learnable:
#             self.offset_generators[0].weight.requires_grad = False
#             self.offset_generators[0].bias.requires_grad = False

#     def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
#         bs, channel = sampling_query.shape
#         if self.uniform_grid:
#             if stage != 0:
#                 xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#                 sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#             else:
#                 sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
#         else:
#             xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#             sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#         feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
#         pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

#         # [0,1] to [-1,1]
#         sampled_points = (2 * sampled_points) - 1

#         sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
#                                        align_corners=False).squeeze(-1)  # (bs, channel, in_points)
#         pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

#         return sampled_features, pe

#     def forward(self, img_data, text_data, tem_imgs, tem_txts, category):
#       bs = img_data.tensors.shape[0]

#       # Category를 숫자로 변환
#       category_idx = torch.tensor([self.category_to_idx[cat] for cat in category], device=img_data.tensors.device)

#       # 1. Template 피처 먼저 학습
#       template_visu_feats = []
#       template_text_feats = []
#       template_visu_masks = []
#       template_text_masks = []

#       for i in range(bs):
#           batch_template_visu_feats = []
#           batch_template_text_feats = []
#           batch_template_visu_masks = []
#           batch_template_text_masks = []

#           for j in range(tem_imgs[i].tensors.shape[0]):
#               # Visual Encoder for Template
#               tem_out, tem_visu_pos = self.visumodel(tem_imgs[i].tensors[j].unsqueeze(0))
#               tem_visu_mask, tem_visu_src = tem_out  # Mask와 Visual Feature 분리
#               tem_visu_src = self.visu_proj(tem_visu_src)
#               batch_template_visu_feats.append(tem_visu_src)
#               batch_template_visu_masks.append(tem_visu_mask)
              

#               # Language Encoder for Template
#               tem_txt_tensor = tem_txts[i].tensors[j].unsqueeze(0)
#               tem_txt_mask = tem_txts[i].mask[j].unsqueeze(0)

#               tem_text_fea = self.textmodel(NestedTensor(tem_txt_tensor, tem_txt_mask))
#               tem_text_src, tem_text_mask = tem_text_fea.decompose()
#               tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)
#               batch_template_text_feats.append(tem_text_src)
#               batch_template_text_masks.append(tem_text_mask.unsqueeze(0))

#           template_visu_feats.append(torch.cat(batch_template_visu_feats, dim=0))
#           template_visu_masks.append(torch.cat(batch_template_visu_masks, dim=1))
#           template_text_feats.append(torch.cat(batch_template_text_feats, dim=0))
#           template_text_masks.append(torch.cat(batch_template_text_masks, dim=1))

#       # 템플릿 피처를 [batch_size, num_tokens, feature_dim] 형식으로 병합
#       template_visu_src = torch.cat(template_visu_feats, dim=1)
#       template_visu_mask = torch.cat(template_visu_masks, dim=0)
#       template_text_src = torch.cat(template_text_feats, dim=1)
#       template_text_mask = torch.cat(template_text_masks, dim=0)

#       template_text_mask = template_text_mask.squeeze(1)
#       template_text_mask = template_text_mask.view(template_text_mask.size(0), -1)
#       # print(template_text_mask.size())
#       # print(template_visu_mask.size())

#       # 템플릿 데이터 통과시키기
#       combined_template_src = torch.cat([template_visu_src, template_text_src], dim=0)
#       combined_template_mask = torch.cat([template_visu_mask, template_text_mask], dim=1)
      
#       # Template positional embedding
#       vl_pos_template = self.vl_pos_embed_template.weight.unsqueeze(1).repeat(1, bs, 1)

#       # print(combined_template_src.size()) #  torch.Size([2200, 16, 256])
#       # print(combined_template_mask.size()) # torch.Size([16, 2200])
#       # print(vl_pos_template.size()) # torch.Size([2200, 16, 256])

#       # Template 멀티모달 인코더 통과 (템플릿 정보 학습)
#       if self.vl_encoder is not None:
#           template_feat = self.vl_encoder(combined_template_src, combined_template_mask, vl_pos_template)
#       else:
#           template_feat = combined_template_src

#       # 2. Pseudo-class embedding 강화 (템플릿 정보 기반)
#       # 템플릿 피처를 평균내어 pseudo-class embedding에 반영
#       template_pseudo_class_feat = template_feat.mean(dim=0)
#       # print (template_pseudo_class_feat.size()) torch.Size([8, 256])

#       # 3. 타겟 데이터와 결합하여 멀티모달 인코더 통과
#       # 3.1 타겟 데이터 인코딩
#       out, visu_pos = self.visumodel(img_data)
#       visu_mask, visu_src = out
#       visu_src = self.visu_proj(visu_src)
#       vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

#       text_fea = self.textmodel(text_data)
#       text_src, text_mask = text_fea.decompose()
#       text_src = self.text_proj(text_src).permute(1, 0, 2)

#       v_pos = vl_pos[:self.num_visu_token]
#       l_pos = vl_pos[self.num_visu_token:self.num_visu_token + self.num_text_token]

#       # 3.2 Pseudo-class embedding for the target category
#       pseudo_class_feat = self.pseudo_class_embedding(category_idx) + template_pseudo_class_feat.unsqueeze(0)
#       pseudo_mask = torch.zeros(bs, 1).to(text_mask.device)

#       # Pseudo-class embedding과 타겟 데이터 결합
#       # print(visu_src.size()) torch.Size([400, 8, 256])
#       # print(text_src.size()) torch.Size([40, 8, 256])
#       # print(pseudo_class_feat.size()) torch.Size([1, 8, 256])
      
#       combined_target_src = torch.cat([visu_src, text_src, pseudo_class_feat], dim=0)
#       combined_target_mask = torch.cat([visu_mask, text_mask, pseudo_mask], dim=1)

#       # 3.3 타겟 및 Pseudo-class embedding 멀티모달 인코더 통과
#       if self.vl_encoder is not None:
#           target_feat = self.vl_encoder(combined_target_src, combined_target_mask, vl_pos)
#       else:
#           target_feat = combined_target_src

#       # 4. Context Embedding
#       visu_feat = target_feat[:self.num_visu_token]
#       language_feat = target_feat[self.num_visu_token:self.num_visu_token + self.num_text_token]
#       pseudo_class_feat = target_feat[-1]
      
#       combined_feat = torch.cat((visu_feat, language_feat), dim=0)
#       context_embed = self.context_embedding(combined_feat.mean(dim=0))

#       # 5. Dynamic Multimodal Transformer Decoder
#       sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
#       reference_point = self.init_reference_point.weight.repeat(bs, 1)

#       for i in range(self.stages):
#           # Combine pseudo-class embedding, context, and sampling query
#           combined_query = torch.cat((sampling_query, pseudo_class_feat, context_embed), dim=1)
#           combined_query = self.sampleing_proj(combined_query)
#           sampled_features, pe = self.feautures_sampling(combined_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

#           if self.different_transformer:
#               vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
#           else:
#               vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

#           language_feat = vg_hs[0]
#           text_select = (1 - text_mask * 1.0).unsqueeze(-1)
#           text_select_num = text_select.sum(dim=1)
#           vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num

#           pred_box = self.bbox_embed(vg_hs).sigmoid()

#           reference_point = pred_box[:, :2]
#           sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

#       return pred_box


    # def forward(self, img_data, text_data, tem_imgs, tem_txts, category):
    #     bs = img_data.tensors.shape[0]

    #      # Category를 숫자로 변환
    #     category_idx = torch.tensor([self.category_to_idx[cat] for cat in category], device=img_data.tensors.device)

    #     # 1. Feature Encoder - Target

    #     # 1.1 Visual Encoder
    #     out, visu_pos = self.visumodel(img_data)
    #     visu_mask, visu_src = out
    #     visu_src = self.visu_proj(visu_src)
    #     vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

    #     # 1.2 Language Encoder
    #     text_fea = self.textmodel(text_data)
    #     text_src, text_mask = text_fea.decompose()
    #     text_src = self.text_proj(text_src).permute(1, 0, 2)

    #     # Pseudo-class embedding for the target category (모든 템플릿에 대해 동일한 pseudo-class embedding 사용)
    #     pseudo_class_feat = self.pseudo_class_embedding(category_idx)
    #     pseudo_mask = torch.zeros(bs, 1).to(text_mask.device)

    #     # Pseudo-class embedding and target data combination
    #     combined_target_src = torch.cat([visu_src, text_src, pseudo_class_feat.unsqueeze(0)], dim=0)
    #     combined_target_mask = torch.cat([visu_mask, text_mask,pseudo_mask], dim=1)

    #     # Target and Pseudo-class embedding through Multimodal Encoder
    #     if self.vl_encoder is not None:
    #         target_feat = self.vl_encoder(combined_target_src, combined_target_mask, vl_pos)
    #     else:
    #         target_feat = combined_target_src
       
       
    #     # 타겟 피처 분리
    #     visu_feat = target_feat[:self.num_visu_token]
    #     language_feat = target_feat[self.num_visu_token:self.num_visu_token + self.num_text_token]
    #     pseudo_class_feat = target_feat[-1]

    #     v_pos = vl_pos[:self.num_visu_token]
    #     l_pos = vl_pos[self.num_visu_token:self.num_visu_token + self.num_text_token]

    #     # Combine visual and language features into a context embedding
    #     combined_feat = torch.cat((visu_feat, language_feat), dim=0)
    #     context_embed = self.context_embedding(combined_feat.mean(dim=0))  # Context derived from combined features

    #     # 2. Template Data 처리 (Target과 동일한 방식)
    #     template_visu_feats = []
    #     template_text_feats = []
    #     template_visu_masks = []
    #     template_text_masks = []
    #     template_visu_pos = []  # 템플릿에 대한 positional embedding

    #     for i in range(bs):
    #         # 개별 배치의 템플릿 피처 저장용 리스트
    #         batch_template_visu_feats = []
    #         batch_template_text_feats = []
    #         batch_template_visu_masks = []
    #         batch_template_text_masks = []
    #         batch_template_visu_pos = []

    #         for j in range(tem_imgs[i].tensors.shape[0]):  # 템플릿 개수만큼 반복
    #             # 2.1 Visual Encoder for Template
    #             tem_out, tem_visu_pos = self.visumodel(tem_imgs[i].tensors[j].unsqueeze(0))
    #             tem_visu_mask, tem_visu_src = tem_out  # Mask와 Visual Feature 분리
    #             tem_visu_src = self.visu_proj(tem_visu_src)
    #             batch_template_visu_feats.append(tem_visu_src)
    #             batch_template_visu_masks.append(tem_visu_mask)
                
    #             #  템플릿의 positional embedding 배치 크기 맞추기
    #             batch_template_visu_pos.append(tem_visu_pos)  

    #             # 2.2 Language Encoder for Template
    #             # 템플릿 텍스트 텐서를 올바른 방식으로 인덱싱
    #             tem_txt_tensor = tem_txts[i].tensors[j].unsqueeze(0)  # (1, 40)
    #             tem_txt_mask = tem_txts[i].mask[j].unsqueeze(0)       # (1, 40)
                
    #             # 텍스트 템플릿 데이터를 언어 모델에 입력
    #             tem_text_fea = self.textmodel(NestedTensor(tem_txt_tensor, tem_txt_mask))
    #             tem_text_src, tem_text_mask = tem_text_fea.decompose()  # Mask와 Text Feature 분리
    #             tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)  # (num_tokens, 1, hidden_dim)
    #             batch_template_text_feats.append(tem_text_src)
    #             batch_template_text_masks.append(tem_text_mask.unsqueeze(0))  # 배치 차원을 추가하여 [1, num_tokens]

    #         # 각 배치별 템플릿 피처를 병합
    #         template_visu_feats.append(torch.cat(batch_template_visu_feats, dim=0))  # 배치 차원을 유지하며 병합
    #         template_visu_masks.append(torch.cat(batch_template_visu_masks, dim=0))  # 배치 차원을 유지하며 병합
    #         template_text_feats.append(torch.cat(batch_template_text_feats, dim=0))  # 배치 차원을 유지하며 병합
    #         template_text_masks.append(torch.cat(batch_template_text_masks, dim=0))  # 배치 차원을 유지하며 병합

    #         # 템플릿에 대한 pos 결합
    #         template_visu_pos.append(torch.cat(batch_template_visu_pos, dim=0))

    #     # 템플릿 피처를 [batch_size, num_tokens, feature_dim] 형식으로 병합
    #     template_visu_src = torch.cat(template_visu_feats, dim=1)
    #     template_visu_mask = torch.cat(template_visu_masks, dim=0)  # [batch_size, num_tokens]
    #     template_text_src = torch.cat(template_text_feats, dim=1)
    #     template_text_mask = torch.cat(template_text_masks, dim=0)  # [batch_size, num_tokens]

    #     # 4. Dynamic Multimodal Transformer Decoder
    #     sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
    #     reference_point = self.init_reference_point.weight.repeat(bs, 1)

    #     for i in range(self.stages):
    #         # Combine pseudo-class embedding, context, and sampling query
    #         combined_query = torch.cat((sampling_query, pseudo_class_feat, context_embed), dim=1)
    #         combined_query = self.sampleing_proj(combined_query)
    #         sampled_features, pe = self.feautures_sampling(combined_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

    #         if self.different_transformer:
    #             vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
    #         else:
    #             vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

    #         language_feat = vg_hs[0]
    #         text_select = (1 - text_mask * 1.0).unsqueeze(-1)
    #         text_select_num = text_select.sum(dim=1)
    #         vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num

    #         pred_box = self.bbox_embed(vg_hs).sigmoid()

    #         reference_point = pred_box[:, :2]
    #         sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

    #     return pred_box


# class DynamicMDETR(nn.Module):
#     def __init__(self, args):
#         super(DynamicMDETR, self).__init__()
#         hidden_dim = args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.num_visu_token = int((args.imsize / divisor) ** 2)
#         self.num_text_token = args.max_query_len
#         self.uniform_grid = args.uniform_grid
#         self.uniform_learnable = args.uniform_learnable
#         self.different_transformer = args.different_transformer

#         # Add pseudo-class embedding (learnable token)
#         self.pseudo_class_embedding = nn.Embedding(1, hidden_dim)

#         # Add context embedding layer
#         self.context_embedding = nn.Linear(hidden_dim, hidden_dim)  # For learning context from text and visual

#         self.visumodel = build_detr(args)
#         self.textmodel = build_bert(args)

#         num_total = self.num_visu_token + self.num_text_token 
#         self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
#         self.vl_encoder = build_vl_encoder(args)

#         self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
#         self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

#         # Sampling relevant
#         self.visual_feature_map_h = 20
#         self.visual_feature_map_w = 20
#         self.in_points = args.in_points
#         self.stages = args.stages
#         self.sampleing_proj = nn.Linear(hidden_dim * 3, hidden_dim)

#         self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
#         self.update_sampling_queries = nn.ModuleList(
#             [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

#         self.init_reference_point = nn.Embedding(1, 2)
#         self.init_sampling_feature = nn.Embedding(1, hidden_dim)

#         self.init_weights()

#         if self.different_transformer:
#             self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
#         else:
#             self.vl_transformer = build_vl_transformer(args)

#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

#     def init_weights(self):
#         nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
#         nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
#         self.init_reference_point.weight.requires_grad=False

#         for i in range(self.stages):
#             nn.init.zeros_(self.offset_generators[i].weight)
#             nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
#         if not self.uniform_learnable:
#             self.offset_generators[0].weight.requires_grad = False
#             self.offset_generators[0].bias.requires_grad = False

#     def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
#         bs, channel = sampling_query.shape
#         if self.uniform_grid:
#             if stage != 0:
#                 xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#                 sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#             else:
#                 sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
#         else:
#             xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#             sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#         feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
#         pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

#         # [0,1] to [-1,1]
#         sampled_points = (2 * sampled_points) - 1

#         sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
#                                        align_corners=False).squeeze(-1)  # (bs, channel, in_points)
#         pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

#         return sampled_features, pe

#     def forward(self, img_data, text_data, tem_imgs, tem_txts, category):
#         bs = img_data.tensors.shape[0]
#         # print(category)

#         print(img_data.tensors.size())  # torch.Size([16, 3, 640, 640])
#         print(text_data.tensors.size()) # torch.Size([16, 40])
#         # template는 리스트 ; 길이는 배치사이즈
#         print(tem_imgs[0].tensors.size()) # torch.Size([5, 3, 640, 640])
#         print(tem_txts[0].tensors.size()) # torch.Size([5, 40])

#         # 1. Feature Encoder - Target

#         # 1.1 Visual Encoder
#         out, visu_pos = self.visumodel(img_data)
#         visu_mask, visu_src = out
#         visu_src = self.visu_proj(visu_src)

#         # 1.2 Language Encoder
#         text_fea = self.textmodel(text_data)
#         text_src, text_mask = text_fea.decompose()
#         text_src = self.text_proj(text_src).permute(1, 0, 2)

#         # 2. Template Data 처리 (Target과 동일한 방식)
#         template_visu_feats = []
#         template_text_feats = []
#         template_visu_masks = []
#         template_text_masks = []
#         template_visu_pos = []  # 템플릿에 대한 positional embedding

#         for i in range(bs):
#             # 개별 배치의 템플릿 피처 저장용 리스트
#             batch_template_visu_feats = []
#             batch_template_text_feats = []
#             batch_template_visu_masks = []
#             batch_template_text_masks = []
#             batch_template_visu_pos = []

#             for j in range(tem_imgs[i].tensors.shape[0]):  # 템플릿 개수만큼 반복
#                 # 2.1 Visual Encoder for Template
#                 tem_out, tem_visu_pos = self.visumodel(tem_imgs[i].tensors[j].unsqueeze(0))
#                 tem_visu_mask, tem_visu_src = tem_out  # Mask와 Visual Feature 분리
#                 tem_visu_src = self.visu_proj(tem_visu_src)
#                 batch_template_visu_feats.append(tem_visu_src)
#                 batch_template_visu_masks.append(tem_visu_mask)
                
#                 #  템플릿의 positional embedding 배치 크기 맞추기
#                 batch_template_visu_pos.append(tem_visu_pos)  

#                 # 2.2 Language Encoder for Template
#                 # 템플릿 텍스트 텐서를 올바른 방식으로 인덱싱
#                 tem_txt_tensor = tem_txts[i].tensors[j].unsqueeze(0)  # (1, 40)
#                 tem_txt_mask = tem_txts[i].mask[j].unsqueeze(0)       # (1, 40)
                
#                 # 텍스트 템플릿 데이터를 언어 모델에 입력
#                 tem_text_fea = self.textmodel(NestedTensor(tem_txt_tensor, tem_txt_mask))
#                 tem_text_src, tem_text_mask = tem_text_fea.decompose()  # Mask와 Text Feature 분리
#                 tem_text_src = self.text_proj(tem_text_src).permute(1, 0, 2)  # (num_tokens, 1, hidden_dim)
#                 batch_template_text_feats.append(tem_text_src)
#                 batch_template_text_masks.append(tem_text_mask.unsqueeze(0))  # 배치 차원을 추가하여 [1, num_tokens]

#             # 각 배치별 템플릿 피처를 병합
#             template_visu_feats.append(torch.cat(batch_template_visu_feats, dim=0))  # 배치 차원을 유지하며 병합
#             template_visu_masks.append(torch.cat(batch_template_visu_masks, dim=0))  # 배치 차원을 유지하며 병합
#             template_text_feats.append(torch.cat(batch_template_text_feats, dim=0))  # 배치 차원을 유지하며 병합
#             template_text_masks.append(torch.cat(batch_template_text_masks, dim=0))  # 배치 차원을 유지하며 병합

#             # 템플릿에 대한 pos 결합
#             template_visu_pos.append(torch.cat(batch_template_visu_pos, dim=0))

#         # 템플릿 피처를 [batch_size, num_tokens, feature_dim] 형식으로 병합
#         template_visu_src = torch.cat(template_visu_feats, dim=1)
#         template_visu_mask = torch.cat(template_visu_masks, dim=0)  # [batch_size, num_tokens]
#         template_text_src = torch.cat(template_text_feats, dim=1)
#         template_text_mask = torch.cat(template_text_masks, dim=0)  # [batch_size, num_tokens]

#         # 템플릿 positional embedding 병합
#         # template_visu_pos = torch.cat(template_visu_pos, dim=0)  # [batch_size, num_tokens, pos_dim]

#         # 3. 멀티모달 인코더에 타겟 데이터와 템플릿 데이터 통합
#         combined_vl_src = torch.cat([visu_src, text_src, template_visu_src, template_text_src], dim=0)

        
#         # print(visu_mask.size()) #
#         # print(text_mask.size()) #
#         # print(template_visu_mask.size()) # 
#         # print(template_text_mask.size()) #  

#         # 템플릿과 타겟의 마스크를 같은 차원으로 맞추기
#         combined_vl_mask = torch.cat([visu_mask, text_mask, template_visu_mask, template_text_mask], dim=1)  # 배치 차원으로 결합

#         vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
#         tem_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
#         combined_vl_pos = torch.cat([vl_pos, tem_pos], dim=0)

#         print(combined_vl_src.size()) #  torch.Size([880, 16, 256])
#         print(combined_vl_mask.size()) # torch.Size([16, 880])
#         print(combined_vl_pos.size()) #  torch.Size([16, 512, 20, 20])


#         # 4. Multimodal Transformer
#         if self.vl_encoder is not None:
#             vl_feat = self.vl_encoder(combined_vl_src, combined_vl_mask, combined_vl_pos)
#         else:
#             vl_feat = combined_vl_src

#         # 타겟 및 템플릿 피처 분리 및 반환
#         visu_feat = vl_feat[:self.num_visu_token]
#         language_feat = vl_feat[self.num_visu_token:self.num_visu_token + self.num_text_token]

#         template_start_idx = self.num_visu_token + self.num_text_token
#         template_visu_feat = vl_feat[template_start_idx:template_start_idx + len(template_visu_feats)]
#         template_text_feat = vl_feat[template_start_idx + len(template_visu_feats):]
            

#         # 3. Context Embedding (contextual information added)
#         # Combine visual and language features into a context embedding
#         combined_feat = torch.cat((visu_feat, language_feat), dim=0)
#         context_embed = self.context_embedding(combined_feat.mean(dim=0))  # Context derived from combined features

#         # 4. Dynamic Multimodal Transformer Decoder
#         sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
#         pseudo_class_feat = pseudo_class_feat.squeeze(0)
#         pseudo_class_feat = pseudo_class_feat.view(bs, -1)
#         reference_point = self.init_reference_point.weight.repeat(bs, 1)

#         for i in range(self.stages):
#             # Combine pseudo-class embedding, context, and sampling query
#             combined_query = torch.cat((sampling_query, pseudo_class_feat, context_embed), dim=1)
#             combined_query = self.sampleing_proj(combined_query)
#             sampled_features, pe = self.feautures_sampling(combined_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

#             if self.different_transformer:
#                 vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
#             else:
#                 vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

#             language_feat = vg_hs[0]
#             text_select = (1 - text_mask * 1.0).unsqueeze(-1)
#             text_select_num = text_select.sum(dim=1)
#             vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num

#             pred_box = self.bbox_embed(vg_hs).sigmoid()

#             reference_point = pred_box[:, :2]
#             sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

#         return pred_box



# class DynamicMDETR(nn.Module):
#     def __init__(self, args):
#         super(DynamicMDETR, self).__init__()
#         hidden_dim = args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.num_visu_token = int((args.imsize / divisor) ** 2)
#         self.num_text_token = args.max_query_len
#         self.uniform_grid = args.uniform_grid
#         self.uniform_learnable = args.uniform_learnable
#         self.different_transformer = args.different_transformer

#         # Add pseudo-class embedding (learnable token)
#         self.pseudo_class_embedding = nn.Embedding(1, hidden_dim)

#         self.visumodel = build_detr(args)
#         self.textmodel = build_bert(args)

#         num_total = self.num_visu_token + self.num_text_token + 1  # +1 for pseudo-class embedding
#         self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
#         self.vl_encoder = build_vl_encoder(args)

#         self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
#         self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

#         # Sampling relevant
#         self.visual_feature_map_h = 20
#         self.visual_feature_map_w = 20
#         self.in_points = args.in_points
#         self.stages = args.stages
#         self.sampleing_proj = nn.Linear(hidden_dim * 2, hidden_dim)

#         self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
#         self.update_sampling_queries = nn.ModuleList(
#             [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

#         self.init_reference_point = nn.Embedding(1, 2)
#         self.init_sampling_feature = nn.Embedding(1, hidden_dim)

#         self.init_weights()
#         if self.different_transformer:
#             self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
#         else:
#             self.vl_transformer = build_vl_transformer(args)
#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
#         if self.uniform_grid:
#             h = int(math.sqrt(self.in_points))
#             w = h
#             step = 1 / h
#             start = 1 / h / 2

#             new_h = torch.tensor([start + i * step for i in range(h)]).view(-1, 1).repeat(1, w)
#             new_w = torch.tensor([start + j * step for j in range(w)]).repeat(h, 1)
#             grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
#             grid = grid.view(-1, 2)  # (in_points, 2)
#             self.initial_sampled_points = torch.nn.Parameter(grid.unsqueeze(0))  # (1, in_points, 2)

#     def init_weights(self):
#         nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
#         nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
#         self.init_reference_point.weight.requires_grad=False

#         for i in range(self.stages):
#             nn.init.zeros_(self.offset_generators[i].weight)
#             nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
#         if not self.uniform_learnable:
#             self.offset_generators[0].weight.requires_grad = False
#             self.offset_generators[0].bias.requires_grad = False

#     def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
#         bs, channel = sampling_query.shape
#         if self.uniform_grid:
#             if stage != 0:
#                 xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#                 sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#             else:
#                 sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
#         else:
#             xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#             sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#         feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
#         pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

#         # [0,1] to [-1,1]
#         sampled_points = (2 * sampled_points) - 1

#         sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
#                                        align_corners=False).squeeze(-1)  # (bs, channel, in_points)
#         pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

#         return sampled_features, pe

#     def forward(self, img_data, text_data):
#         bs = img_data.tensors.shape[0]

#         # 1. Feature Encoder

#         # 1.1 Visual Encoder
#         # visual backbone
#         out, visu_pos = self.visumodel(img_data)
#         visu_mask, visu_src = out # (B, H*W), (H*W, B, channel)
#         visu_src = self.visu_proj(visu_src)  # (H*W, B, channel)

#         # 1.2 Language Encoder
#         # language bert
#         text_fea = self.textmodel(text_data)
#         text_src, text_mask = text_fea.decompose()
#         assert text_mask is not None
#         # text_src: (bs, max_len, channel)
#         text_mask = text_mask.flatten(1)  # (B, max_len)
#         text_src = self.text_proj(text_src).permute(1, 0, 2)  # (max_len, B, channel)

#         # 1.3 Concat visual features, language features, and pseudo-class embedding
#         pseudo_class_emb = self.pseudo_class_embedding.weight.unsqueeze(1).repeat(1, bs, 1)  # Add pseudo-class embedding
#         vl_src = torch.cat([visu_src, text_src, pseudo_class_emb], dim=0)
#         # print(torch.zeros(1, bs).to(text_mask.device).shape)
#         pseudo_mask = torch.zeros(bs, 1).to(text_mask.device)  # [bs, 1] 크기로 생성
#         vl_mask = torch.cat([visu_mask, text_mask, pseudo_mask.to(text_mask.device)], dim=1)  # Mask for pseudo-class embedding
#         vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

#         # 2. Multimodal Transformer
#         # 2.1 Multimodal Transformer Encoder
#         if self.vl_encoder is not None:
#             vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
#         else:
#             vl_feat = vl_src

#         # 2.2 Split back to visual features, language features, and pseudo-class embedding
#         visu_feat = vl_feat[:self.num_visu_token] # (H*W, B, channel)
#         language_feat = vl_feat[self.num_visu_token:self.num_visu_token + self.num_text_token] # (max_len, B, channel)
#         pseudo_class_feat = vl_feat[self.num_visu_token + self.num_text_token:]  # (1, B, channel)

#         v_pos = vl_pos[:self.num_visu_token]
#         l_pos = vl_pos[self.num_visu_token:self.num_visu_token + self.num_text_token]

#         # 2.3 Dynamic Multimodal Transformer Decoder
#         # Initialize sampling query and reference point for the first features sampling
#         # sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
#         # reference_point = self.init_reference_point.weight.repeat(bs, 1)
#         # pred_box = None

#         # for i in range(0, self.stages):
#         #     # 2D adaptive sampling
#         #     sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

#         #     # Text guided decoding with one-layer transformer encoder-decoder
#         #     if self.different_transformer:
#         #         vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
#         #     else:
#         #         vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

#         #     # Prediction Head
#         #     language_feat = vg_hs[0]

#         #     text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
#         #     text_select_num = text_select.sum(dim=1)  # (bs, 1)

#         #     # new language queries
#         #     vg_hs = (text_select * vg_hs[0].permute(1,0,2)).sum(dim=1) / text_select_num  # (bs, channel)

#         #     pred_box = self.bbox_embed(vg_hs).sigmoid()

#         #     # Update reference point and sampling query
#         #     reference_point = pred_box[:, :2]
#         #     sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))
        
#         # 2.3 Dynamic Multimodal Transformer Decoder
#         # Initialize sampling query and reference point for the first features sampling
#         sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
#         pseudo_class_feat = pseudo_class_feat.squeeze(0)  # (B, channel) 형태로 변경하여 결합 준비
#         pseudo_class_feat = pseudo_class_feat.view(bs, -1)  # 크기 조정
#         reference_point = self.init_reference_point.weight.repeat(bs, 1)

#         for i in range(0, self.stages):
#             # 2D adaptive sampling
#             combined_query = torch.cat((sampling_query, pseudo_class_feat), dim=1)  # Pseudo-class feature를 결합
#             combined_query = self.sampleing_proj(combined_query) 
#             # print(combined_query.shape)
#             sampled_features, pe = self.feautures_sampling(combined_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

#             # Text guided decoding with one-layer transformer encoder-decoder
#             if self.different_transformer:
#                 vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
#             else:
#                 vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

#             # Prediction Head
#             language_feat = vg_hs[0]

#             text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
#             text_select_num = text_select.sum(dim=1)  # (bs, 1)

#             # New language queries
#             vg_hs = (text_select * vg_hs[0].permute(1, 0, 2)).sum(dim=1) / text_select_num  # (bs, channel)

#             pred_box = self.bbox_embed(vg_hs).sigmoid()

#             # Update reference point and sampling query with pseudo-class feature
#             reference_point = pred_box[:, :2]
#             sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))


#         return pred_box

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncodingSine(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 20):
        super(PositionalEncodingSine, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding

    def forward(self, token_embedding):
        return self.pos_embedding[:token_embedding.size(0), :] 



# class DynamicMDETR(nn.Module):
#     def __init__(self, args):
#         super(DynamicMDETR, self).__init__()
#         hidden_dim = args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.num_visu_token = int((args.imsize / divisor) ** 2)
#         self.num_text_token = args.max_query_len
#         self.uniform_grid = args.uniform_grid
#         self.uniform_learnable = args.uniform_learnable
#         self.different_transformer = args.different_transformer

#         self.visumodel = build_detr(args)
#         self.textmodel = build_bert(args)

#         num_total = self.num_visu_token + self.num_text_token
#         self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
#         self.vl_encoder = build_vl_encoder(args)

#         self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
#         self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

#         # Sampling relevant
#         self.visual_feature_map_h = 20
#         self.visual_feature_map_w = 20
#         self.in_points = args.in_points
#         self.stages = args.stages

#         self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points * 2) for i in range(args.stages)])
#         self.update_sampling_queries = nn.ModuleList(
#             [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for i in range(args.stages)])

#         self.init_reference_point = nn.Embedding(1, 2)
#         self.init_sampling_feature = nn.Embedding(1, hidden_dim)

#         self.init_weights()
#         if self.different_transformer:
#             self.vl_transformer = nn.ModuleList([build_vl_transformer(args) for i in range(args.stages)])
#         else:
#             self.vl_transformer = build_vl_transformer(args)
#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
#         if self.uniform_grid:
#             h = int(math.sqrt(self.in_points))
#             w = h
#             step = 1 / h
#             start = 1 / h / 2

#             new_h = torch.tensor([start + i * step for i in range(h)]).view(-1, 1).repeat(1, w)
#             new_w = torch.tensor([start + j * step for j in range(w)]).repeat(h, 1)
#             grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
#             grid = grid.view(-1, 2)  # (in_points, 2)
#             self.initial_sampled_points = torch.nn.Parameter(grid.unsqueeze(0))  # (1, in_points, 2)

#     def init_weights(self):
#         nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
#         nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
#         self.init_reference_point.weight.requires_grad=False

#         for i in range(self.stages):
#             nn.init.zeros_(self.offset_generators[i].weight)
#             nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
#         if not self.uniform_learnable:
#             self.offset_generators[0].weight.requires_grad = False
#             self.offset_generators[0].bias.requires_grad = False

#     def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
#         bs, channel = sampling_query.shape
#         if self.uniform_grid:
#             if stage != 0:
#                 xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#                 sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#             else:
#                 sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)
#         else:
#             xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
#             sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
#         feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
#         pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

#         # [0,1] to [-1,1]
#         sampled_points = (2 * sampled_points) - 1

#         sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
#                                        align_corners=False).squeeze(-1)  # (bs, channel, in_points)
#         pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

#         return sampled_features, pe

#     def forward(self, img_data, text_data):
#         bs = img_data.tensors.shape[0]

#         # 1. Feature Encoder

#         # 1.1 Visual Encoder
#         # visual backbone
#         out, visu_pos = self.visumodel(img_data)
#         visu_mask, visu_src = out # (B, H*W), (H*W, B, channel)
#         visu_src = self.visu_proj(visu_src)  # (H*W, B, channel)

#         # 1.2 Language Encoder
#         # language bert
#         text_fea = self.textmodel(text_data)
#         text_src, text_mask = text_fea.decompose()
#         assert text_mask is not None
#         # text_src: (bs, max_len, channel)
#         text_mask = text_mask.flatten(1)  # (B, max_len)
#         text_src = self.text_proj(text_src).permute(1, 0, 2)  # (max_len, B, channel)

#         # 1.3 Concat visual features and language features
#         vl_src = torch.cat([visu_src, text_src], dim=0)
#         vl_mask = torch.cat([visu_mask, text_mask], dim=1)
#         vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)


#         # 2. Multimodal Transformer
#         # 2.1 Multimodal Transformer Encoder
#         if self.vl_encoder is not None:
#             vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
#         else:
#             vl_feat = vl_src

#         # 2.2 Split back to visual features and language features, use language features as queries
#         visu_feat = vl_feat[:self.num_visu_token] # (H*W, B, channel)
#         language_feat = vl_feat[self.num_visu_token:] # (max_len, B, channel)
#         v_pos = vl_pos[:self.num_visu_token]
#         l_pos = vl_pos[self.num_visu_token:]

#         # 2.3 Dynamic Multimodal Transformer Decoder
#         # Initialize sampling query and reference point for the first features sampling
#         sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
#         reference_point = self.init_reference_point.weight.repeat(bs, 1)
#         pred_box = None

#         for i in range(0, self.stages):
#             # 2D adaptive sampling
#             sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

#             # Text guided decoding with one-layer transformer encoder-decoder
#             if self.different_transformer:
#                 vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
#             else:
#                 vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

#             # Prediction Head
#             language_feat = vg_hs[0]

#             text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
#             text_select_num = text_select.sum(dim=1)  # (bs, 1)

#             # new language queries
#             vg_hs = (text_select * vg_hs[0].permute(1,0,2)).sum(dim=1) / text_select_num  # (bs, channel)

#             pred_box = self.bbox_embed(vg_hs).sigmoid()

#             # Update reference point and sampling query
#             reference_point = pred_box[:, :2]
#             sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

#         return pred_box


# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#             # x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
#         return x


# class PositionalEncodingSine(nn.Module):
#     def __init__(self, emb_size: int, maxlen: int = 20):
#         super(PositionalEncodingSine, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         self.pos_embedding = pos_embedding

#     def forward(self, token_embedding):
#         return self.pos_embedding[:token_embedding.size(0), :]