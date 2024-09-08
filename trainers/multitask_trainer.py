import torch
import numpy as np
from .consistency_loss import ConsistencyLoss

class MultiTaskTrainer():
    def __init__(self, args, train_loader, device, optimizer, scaler, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.class_criterion, self.angle_criterion, self.regression_criterion = criterion
        self.alpha = 10

        if self.args.use_consistency_loss:
            self.model_consistency_loss = ConsistencyLoss(self.train_loader.dataset.make_model_hierarchy_table)
            self.year_consistency_loss = ConsistencyLoss(self.train_loader.dataset.model_year_hierarchy_table)
            self.consistency_coeff = 0.25
            if 'stanford' in self.args.dataset:
                self.make_consistency_loss = ConsistencyLoss(self.train_loader.dataset.body_make_hierarchy_table)

    def train(self, model):
        model.train()
        total_train_loss = 0
        for i, data in enumerate(self.train_loader):
            images = data['img'].to(self.device)
            make_label = data['make_label'].to(self.device)
            model_label = data['model_label'].to(self.device)
            year_label = data['year_label'].to(self.device)
            full_label = data['full_label'].to(self.device)

            if 'stanford' in self.args.dataset:
                body_label = data['body_label'].to(self.device)

            if 'autovit' in self.args.dataset: 
                angle_label = data['angle'].to(self.device).unsqueeze(1)
                price_label = data['price'].to(self.device).unsqueeze(1)
                km_label = data['km'].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if 'stanford' in self.args.dataset:
                    if self.args.predict_full:
                        body_pred, make_pred, model_pred, year_pred, full_pred = model(images)
                    else:
                        body_pred, make_pred, model_pred, year_pred = model(images)
                else:
                    angle_pred, make_pred, model_pred, year_pred, full_pred, km_pred, price_pred = model(images)
                    
                loss = 0
                if self.args.predict_body:
                    loss += self.class_criterion(body_pred, body_label)

                if self.args.predict_angle:
                    loss += self.angle_criterion(angle_pred, angle_label) * self.alpha
                
                if self.args.predict_car_model:
                    make_loss = self.class_criterion(make_pred, make_label)
                    model_loss = self.class_criterion(model_pred, model_label)
                    year_loss = self.class_criterion(year_pred, year_label)
                    loss += make_loss + model_loss + year_loss
                
                if self.args.predict_full:
                    full_loss = self.class_criterion(full_pred, full_label)
                    loss += full_loss
                    
                if self.args.predict_km:
                    loss += self.regression_criterion(km_pred, km_label) * self.alpha
                
                if self.args.predict_price:
                    loss += self.regression_criterion(price_pred, price_label) * self.alpha

                if self.args.use_consistency_loss:
                    make_cons_loss = 0
                    if 'stanford' in self.args.dataset:
                        make_cons_loss = self.make_consistency_loss(body_pred, make_pred)
                    model_cons_loss = self.model_consistency_loss(make_pred, model_pred)
                    year_cons_loss = self.year_consistency_loss(model_pred, year_pred)
                    loss += self.consistency_coeff * (make_cons_loss + model_cons_loss + year_cons_loss)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_train_loss += loss.item()
            if self.args.debug:
                break

        self.scheduler.step()

        train_loss = np.round(total_train_loss / len(self.train_loader), 2)
        print("Train Loss: " + str(train_loss))
        
        return train_loss
