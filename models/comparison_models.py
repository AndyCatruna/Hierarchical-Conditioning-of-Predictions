import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedBaseModel(nn.Module):
    def __init__(self, args, emb_dim, full_classes):
        super().__init__()
        
        self.backbone = nn.Identity()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(emb_dim, full_classes)

    def forward(self, x):
        emb = self.backbone(x)
        activation = self.gelu(emb)

        full = self.fc(activation)

        return full

class SwinSmallUnifiedModel(UnifiedBaseModel):
    def __init__(self, args, full_classes):
        super().__init__(args, 768, full_classes)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()

class VitSmallUnifiedModel(UnifiedBaseModel):
    def __init__(self, args, full_classes):
        super().__init__(args, 384, full_classes)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()

class SeparateBaseModel(nn.Module):
    def __init__(self, args, emb_dim, make_classes, model_classes, year_classes, full_classes):
        super().__init__()
        
        self.backbone = nn.Identity()
        self.predict_car_model = args.predict_car_model
        self.predict_full = args.predict_full
        self.predict_angle = args.predict_angle
        self.predict_km = args.predict_km
        self.predict_price = args.predict_price
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

        hidden_dim = 256

        if args.predict_angle:
            self.angle_emb = nn.Linear(emb_dim, hidden_dim)
            self.angle_head = nn.Linear(hidden_dim, 1)
            
        if args.predict_car_model:
            self.make_emb = nn.Linear(emb_dim, hidden_dim)
            self.make_fc = nn.Linear(hidden_dim, make_classes)

            self.model_emb = nn.Linear(emb_dim, hidden_dim)
            self.model_fc = nn.Linear(hidden_dim, model_classes)

            self.year_emb = nn.Linear(emb_dim, hidden_dim)
            self.year_fc = nn.Linear(hidden_dim, year_classes)
        
        if args.predict_full:
            self.full_emb = nn.Linear(emb_dim, hidden_dim)
            self.full_fc = nn.Linear(hidden_dim, full_classes)

        if args.predict_km:
            self.km_emb = nn.Linear(emb_dim, hidden_dim)
            self.km_head = nn.Linear(hidden_dim, 1)

        if args.predict_price:
            self.price_emb = nn.Linear(emb_dim, hidden_dim)
            self.price_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.backbone(x)
        activation = self.gelu(emb)

        angle = None
        if self.predict_angle:
            angle_emb = self.angle_emb(activation)
            angle = self.angle_head(self.gelu(angle_emb))
            angle = self.sigmoid(angle)

        make = None
        model = None
        year = None
        full = None
        if self.predict_car_model:
            make_emb = self.make_emb(activation)
            make = self.make_fc(self.gelu(make_emb))

            model_emb = self.model_emb(activation)
            model = self.model_fc(self.gelu(model_emb))

            year_emb = self.year_emb(activation)
            year = self.year_fc(self.gelu(year_emb))
        
        if self.predict_full:
            full_emb = self.full_emb(activation)
            full = self.full_fc(self.gelu(full_emb))
        
        km = None
        if self.predict_km:
            km_emb = self.km_emb(activation)
            km = self.km_head(self.gelu(km_emb))
            km = self.sigmoid(km)

        price = None
        if self.predict_price:
            price_emb = self.price_emb(activation)
            price = self.price_head(self.gelu(price_emb))
            price = self.sigmoid(price)

        if self.predict_full:
            return angle, make, model, year, full, km, price
        else:
            return angle, make, model, year, km, price

class SwinSmallSeparateModel(SeparateBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, full_classes):
        super().__init__(args, 768, make_classes, model_classes, year_classes, full_classes)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()

class VitSmallSeparateModel(SeparateBaseModel):
    def __init__(self, args, make_classes, model_classes, year_classes, full_classes):
        super().__init__(args, 384, make_classes, model_classes, year_classes, full_classes)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()

class SeparateBaseStanfordModel(nn.Module):
    def __init__(self, args, emb_dim, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__()
        
        self.backbone = nn.Identity()
        self.predict_car_model = args.predict_car_model
        self.predict_body = args.predict_body
        self.predict_full = args.predict_full
        self.gelu = nn.GELU()

        hidden_dim = 256
        
        if args.predict_body:
            self.body_emb = nn.Linear(emb_dim, hidden_dim)
            self.body_fc = nn.Linear(hidden_dim, body_classes)

        if args.predict_car_model:
            self.make_emb = nn.Linear(emb_dim, hidden_dim)
            self.make_fc = nn.Linear(hidden_dim, make_classes)

            self.model_emb = nn.Linear(emb_dim, hidden_dim)
            self.model_fc = nn.Linear(hidden_dim, model_classes)

            self.year_emb = nn.Linear(emb_dim, hidden_dim)
            self.year_fc = nn.Linear(hidden_dim, year_classes)

        if args.predict_full:
            self.full_emb = nn.Linear(emb_dim, hidden_dim)
            self.full_fc = nn.Linear(hidden_dim, full_classes)

    def forward(self, x):
        emb = self.backbone(x)
        activation = self.gelu(emb)

        body = None
        if self.predict_body:
            body_emb = self.body_emb(activation)
            body = self.body_fc(self.gelu(body_emb))

        make = None
        model = None
        year = None
        full = None

        if self.predict_car_model:
            make_emb = self.make_emb(activation)
            make = self.make_fc(self.gelu(make_emb))

            model_emb = self.model_emb(activation)
            model = self.model_fc(self.gelu(model_emb))

            year_emb = self.year_emb(activation)
            year = self.year_fc(self.gelu(year_emb))

        if self.predict_full:
            full_emb = self.full_emb(activation)
            full = self.full_fc(self.gelu(full_emb))

        return body, make, model, year, full
    
class SwinSmallStanfordSeparateModel(SeparateBaseStanfordModel):
    def __init__(self, args, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__(args, 768, make_classes, model_classes, year_classes, body_classes, full_classes)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.backbone.head = nn.Identity()

class VitSmallStanfordSeparateModel(SeparateBaseStanfordModel):
    def __init__(self, args, make_classes, model_classes, year_classes, body_classes, full_classes):
        super().__init__(args, 384, make_classes, model_classes, year_classes, body_classes, full_classes)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()
