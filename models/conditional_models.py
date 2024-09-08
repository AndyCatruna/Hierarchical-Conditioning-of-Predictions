import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionHeadAttention(nn.Module):
    def __init__(self, args, emb_dim, num_classes):
        super().__init__()
        self.args = args
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, emb_dim))
        self.use_residual = args.use_residual

        emb_dim = emb_dim // 4
        self.projection = nn.Linear(emb_dim * 4, emb_dim)
        self.attention_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=args.gating_num_heads, dim_feedforward=emb_dim * 4, dropout=args.dropout, activation='gelu', batch_first=True)
        if args.use_consistency_loss:
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
        else:
            self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=args.gating_depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.classifier = nn.Linear(emb_dim, num_classes)

        self.squeeze_excite = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, emb_dim),
            nn.Sigmoid()
        )

    def forward(self, x, previous_emb=None):
        # Project the embeddings into lower dimension
        features = self.projection(x)

        # Incorporate previous embedding
        if previous_emb is not None:
            scaling = self.squeeze_excite(previous_emb)
            scaling = scaling.unsqueeze(1)
            features = features * scaling
        
        # Add the cls token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        features = torch.cat((cls_token, features), dim=1)

        # Attention
        after_attention = self.attention(features)

        # Get the cls token
        cls_token = after_attention[:, 0, :]
        
        # Classify
        class_logits = self.classifier(cls_token)

        return class_logits, cls_token

class ConditionalBaseModel(nn.Module):
    def __init__(self, args, emb_dim, make_classes, model_classes, year_classes, full_classes):
        super().__init__()
        
        self.backbone = nn.Identity()
        self.predict_car_model = args.predict_car_model
        self.predict_full = args.predict_full
        self.predict_angle = args.predict_angle
        self.predict_km = args.predict_km
        self.predict_price = args.predict_price
        self.sigmoid = nn.Sigmoid()

        if args.predict_angle:
            self.angle_head = PredictionHeadAttention(args, emb_dim, 1)
            
        if args.predict_car_model:
            self.make_fc = PredictionHeadAttention(args, emb_dim, make_classes)
            self.model_fc = PredictionHeadAttention(args, emb_dim, model_classes)
            self.year_fc = PredictionHeadAttention(args, emb_dim, year_classes)
        
        if args.predict_full:
            self.full_fc = PredictionHeadAttention(args, emb_dim, full_classes)

        if args.predict_km:
            self.km_head = PredictionHeadAttention(args, emb_dim, 1)

        if args.predict_price:
            self.price_head = PredictionHeadAttention(args, emb_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.size()) == 4:
            features = features.flatten(2).transpose(1, 2)
        
        # Use previous embeddings to help predict current embeddings
        current_emb = None
        
        angle = None
        if self.predict_angle:
            angle, current_emb = self.angle_head(features, current_emb)
            angle = self.sigmoid(angle)

        make = None
        model = None
        year = None
        if self.predict_car_model:
            make, current_emb = self.make_fc(features, current_emb)
            model, current_emb = self.model_fc(features, current_emb)
            year, current_emb = self.year_fc(features, current_emb)

        full = None
        if self.predict_full:
            full, current_emb = self.full_fc(features, current_emb)

        km = None
        if self.predict_km:
            km, current_emb = self.km_head(features, current_emb)
            km = self.sigmoid(km)
        
        price = None
        if self.predict_price:
            price, current_emb = self.price_head(features, current_emb)
            price = self.sigmoid(price)
        
        return angle, make, model, year, full, km, price

class SwinSmallConditionalModel(ConditionalBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, full_classes):
        super().__init__(args, 768, make_classes, model_classes, year_classes, full_classes)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.backbone.reset_classifier(0, '')

class VitSmallConditionalModel(ConditionalBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, full_classes):
        super().__init__(args, 384, make_classes, model_classes, year_classes, full_classes)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.reset_classifier(0, '')

class ConditionalStanfordBaseModel(nn.Module):
    def __init__(self, args, emb_dim, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__()
        
        self.backbone = nn.Identity()
        self.predict_car_model = args.predict_car_model
        self.predict_body = args.predict_body
        self.predict_full = args.predict_full

        if args.predict_body:
            self.body_head = PredictionHeadAttention(args, emb_dim, body_classes)
        
        if args.predict_car_model:
            self.make_fc = PredictionHeadAttention(args, emb_dim, make_classes)
            self.model_fc = PredictionHeadAttention(args, emb_dim, model_classes)
            self.year_fc = PredictionHeadAttention(args, emb_dim, year_classes)

        if args.predict_full:
            self.full_fc = PredictionHeadAttention(args, emb_dim, full_classes)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.size()) == 4:
            features = features.flatten(2).transpose(1, 2)
        # Use previous embeddings to help predict current embeddings
        current_emb = None
        
        body = None
        if self.predict_body:
            body, current_emb = self.body_head(features, current_emb)

        make = None
        model = None
        year = None
        if self.predict_car_model:
            make, current_emb = self.make_fc(features, current_emb)
            model, current_emb = self.model_fc(features, current_emb)
            year, current_emb = self.year_fc(features, current_emb)

        full = None
        if self.predict_full:
            full, current_emb = self.full_fc(features, current_emb)
        
        return body, make, model, year, full

class SwinSmallConditionalStanfordModel(ConditionalStanfordBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__(args, 768, make_classes, model_classes, year_classes, body_classes, full_classes)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.backbone.reset_classifier(0, '')

class VitSmallConditionalStanfordModel(ConditionalStanfordBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__(args, 384, make_classes, model_classes, year_classes, body_classes, full_classes)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.reset_classifier(0, '')

