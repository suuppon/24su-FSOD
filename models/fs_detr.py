import torch

from .detr import DETR
from .pseudo_embedding import PseudoEmbedding
from .transformer import Transformer

class FSDETR(DETR):
    def __init__(self, 
                 backbone, 
                 transformer: Transformer, 
                 num_classes: int, 
                 num_queries: int, 
                 aux_loss: bool=False, 
                 with_pseudo: bool=False):
        
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss)
        self.with_pseudo = with_pseudo
        if with_pseudo:
            self.pseudo_embedding = PseudoEmbedding(num_classes, transformer.d_model)
            
    def get_template_feature(self, template):
        # Extract template features
        template_feature = self.backbone(template)
        return template_feature
    
    def forward(self, samples, templates=None, template_labels=None):
            
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        
        if self.with_pseudo:
            template_features = []
            
            for template in templates:
                template_feature = self.get_template_feature(template)
                template_features.append(template_feature)
                
            # TODO : template feature concat 후에 Average pooling or attention pooling 구현
            # template_features = torch.cat(template_features, dim=1)
            # template_features = torch.mean(template_features, dim=1)
            
            pseudo_embeddings = self.pseudo_embedding(template_features, template_labels)
            template_features = torch.cat([template_features, pseudo_embeddings], dim=1)
            
            tgt = torch.cat([template_features, tgt], dim=1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        
        # hs에서 template features에 대한 결과만 반환
        return hs[:, :template_features.shape[1]]