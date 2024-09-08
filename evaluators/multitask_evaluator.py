import torch
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import f1_score
from common import save_checkpoint

class MultiTaskEvaluator():
    def __init__(self, args, test_loader, device, criterion, run_name, km_metric, angle_metric, price_metric, lookup_table=None):
        self.args = args
        self.test_loader = test_loader
        self.device = device
        self.class_criterion, self.angle_criterion, self.regression_criterion = criterion
        self.best_accuracy = 0
        self.run_name = run_name
        self.angle_metric = angle_metric
        self.km_metric = km_metric
        self.price_metric = price_metric
        self.top_performances = [1000, 0, 0, 0, 0, 1000000, 1000000]

        self.top_f1 = [0, 0, 0, 0, 0]

        if args.predict_body:
            self.top_performances[0] = 0

        self.lookup_table = lookup_table
    
    def test(self, model):
        model.eval()
        total_val_loss = 0
        total = 0.0
        angle_total = 0.0
        body_accuracy = 0.0
        make_accuracy = 0.0
        model_accuracy = 0.0
        year_accuracy = 0.0
        km_total = 0.0
        price_total = 0.0
        avg_accuracy = 0.0

        all_price_preds = []
        all_price_labels = []
        
        all_km_preds = []
        all_km_labels = []

        # For f1 score
        all_body_preds = []
        all_body_labels = []

        all_make_preds = []
        all_make_labels = []

        all_model_preds = []
        all_model_labels = []

        all_year_preds = []
        all_year_labels = []

        predicted_price_denormalized = []
        gt_price_denormalized = []

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images = data['img'].to(self.device)
                make_label = data['make_label'].to(self.device)
                model_label = data['model_label'].to(self.device)
                year_label = data['year_label'].to(self.device)

                if 'stanford' in self.args.dataset:
                    body_label = data['body_label'].to(self.device)
                if 'autovit' in self.args.dataset:
                    angle_label = data['angle'].to(self.device).unsqueeze(1)
                    price_label = data['price'].to(self.device).unsqueeze(1)
                    km_label = data['km'].to(self.device).unsqueeze(1)

                if 'stanford' in self.args.dataset:
                    if self.args.predict_full:
                        body_pred, make_pred, model_pred, year_pred, _ = model(images)
                    else:
                        body_pred, make_pred, model_pred, year_pred = model(images)

                    all_body_preds.extend(body_pred.argmax(dim=1).cpu().numpy().tolist())
                    all_body_labels.extend(body_label.cpu().numpy().tolist())
                else:
                    angle_pred, make_pred, model_pred, year_pred, _, km_pred, price_pred = model(images)
                
                if self.args.predict_car_model:
                    all_make_preds.extend(make_pred.argmax(dim=1).cpu().numpy().tolist())
                    all_make_labels.extend(make_label.cpu().numpy().tolist())

                    all_model_preds.extend(model_pred.argmax(dim=1).cpu().numpy().tolist())
                    all_model_labels.extend(model_label.cpu().numpy().tolist())

                    all_year_preds.extend(year_pred.argmax(dim=1).cpu().numpy().tolist())
                    all_year_labels.extend(year_label.cpu().numpy().tolist())

                body_loss = 0
                if self.args.predict_body:
                    body_loss = self.class_criterion(body_pred, body_label)

                angle_loss = 0
                if self.args.predict_angle:
                    angle_loss = self.angle_criterion(angle_pred, angle_label)

                make_loss, model_loss, year_loss = 0, 0, 0
                
                if self.args.predict_car_model:
                    make_loss = self.class_criterion(make_pred, make_label)
                    model_loss = self.class_criterion(model_pred, model_label)
                    year_loss = self.class_criterion(year_pred, year_label)

                km_loss = 0
                price_loss = 0
                if self.args.predict_km:
                    km_loss = self.regression_criterion(km_pred, km_label)
                    all_km_preds.extend(km_pred.squeeze().cpu().numpy().tolist())
                    all_km_labels.extend(km_label.squeeze().cpu().numpy().tolist())

                if self.args.predict_price:
                    price_loss = self.regression_criterion(price_pred, price_label)
                    all_price_preds.extend(price_pred.squeeze().cpu().numpy().tolist())
                    all_price_labels.extend(price_label.squeeze().cpu().numpy().tolist())
                    gt_prices, predicted_prices = self.price_metric.denormalize(price_label, price_pred)
                    predicted_price_denormalized.extend(predicted_prices.squeeze().cpu().numpy().tolist())
                    gt_price_denormalized.extend(gt_prices.squeeze().cpu().numpy().tolist())

                loss = body_loss + angle_loss + make_loss + model_loss + year_loss + km_loss + price_loss
                
                total_val_loss += loss.item()

                if self.args.predict_angle:
                    angle_total += self.angle_metric(angle_pred, angle_label)
                
                if self.args.predict_body:
                    _, body_predicted = torch.max(body_pred, 1)
                    body_accuracy += (body_predicted == body_label).sum().item()

                if self.args.predict_car_model:
                    _, make_predicted = torch.max(make_pred, 1)
                    _, model_predicted = torch.max(model_pred, 1)
                    _, year_predicted = torch.max(year_pred, 1)

                if self.args.predict_km:
                    km_total += self.km_metric(km_pred, km_label)
                if self.args.predict_price:
                    price_total += self.price_metric(price_pred, price_label)

                total += make_label.size(0)

                if self.args.predict_car_model:
                    make_accuracy += (make_predicted == make_label).sum().item()
                    model_accuracy += (model_predicted == model_label).sum().item()
                    year_accuracy += (year_predicted == year_label).sum().item()


        test_loss = np.round(total_val_loss / len(self.test_loader), 2)
        
        angle_mae = 0
        if self.args.predict_angle:
            angle_mae = np.round(angle_total.item() / total, 2)

        if self.args.predict_body:
            body_accuracy = np.round(100 * body_accuracy / total, 2)

            body_f1 = np.round(f1_score(all_body_labels, all_body_preds, average='weighted'), 4)

        if self.args.predict_car_model:
            make_accuracy = np.round(100 * make_accuracy / total, 2)
            model_accuracy = np.round(100 * model_accuracy / total, 2)
            year_accuracy = np.round(100 * year_accuracy / total, 2)
            avg_accuracy = np.round((make_accuracy + model_accuracy + year_accuracy) / 3, 2)
            
            make_f1 = np.round(f1_score(all_make_labels, all_make_preds, average='weighted'), 4)
            model_f1 = np.round(f1_score(all_model_labels, all_model_preds, average='weighted'), 4)
            year_f1 = np.round(f1_score(all_year_labels, all_year_preds, average='weighted'), 4)
            avg_f1 = np.round((make_f1 + model_f1 + year_f1) / 3, 4)

        km_mae = 0
        price_mae = 0
        km_r2 = 0
        price_r2 = 0
        if self.args.predict_km:
            km_mae = np.round(km_total.item() / total, 2)
            all_km_labels = np.array(all_km_labels)
            all_km_preds = np.array(all_km_preds)
            km_r2 = np.round(r2_score(all_km_labels, all_km_preds), 2)
        if self.args.predict_price:
            price_mae = np.round(price_total.item() / total, 2)
            all_price_labels = np.array(all_price_labels)
            all_price_preds = np.array(all_price_preds)
            price_r2 = np.round(r2_score(all_price_labels, all_price_preds), 2)
            price_median_error = np.round(np.median(np.abs(np.array(gt_price_denormalized) - np.array(predicted_price_denormalized))), 2)

        print("Test Loss: " + str(test_loss))
        
        if self.args.predict_body:
            print("Body Accuracy: " + str(body_accuracy))
            if body_accuracy > self.top_performances[0]:
                self.top_performances[0] = body_accuracy

            print("Body F1: " + str(body_f1))
            if body_f1 > self.top_f1[0]:
                self.top_f1[0] = body_f1

        if self.args.predict_angle:
            print("Angle MAE: " + str(angle_mae))
            if angle_mae < self.top_performances[0]:
                self.top_performances[0] = angle_mae
        
        if self.args.predict_car_model:
            print("Make Accuracy: " + str(make_accuracy), end=" ")
            print("Model Accuracy: " + str(model_accuracy), end=" ")
            print("Year Accuracy: " + str(year_accuracy), end=" ")
            print("AVG Accuracy: " + str(avg_accuracy))

            if make_accuracy > self.top_performances[1]:
                self.top_performances[1] = make_accuracy
            if model_accuracy > self.top_performances[2]:
                self.top_performances[2] = model_accuracy
            if year_accuracy > self.top_performances[3]:
                self.top_performances[3] = year_accuracy
            if avg_accuracy > self.top_performances[4]:
                self.top_performances[4] = avg_accuracy

            print("Make F1: " + str(make_f1), end=" ")
            print("Model F1: " + str(model_f1), end=" ")
            print("Year F1: " + str(year_f1), end=" ")
            print("AVG F1: " + str(avg_f1))

            if make_f1 > self.top_f1[1]:
                self.top_f1[1] = make_f1
            if model_f1 > self.top_f1[2]:
                self.top_f1[2] = model_f1
            if year_f1 > self.top_f1[3]:
                self.top_f1[3] = year_f1
            if avg_f1 > self.top_f1[4]:
                self.top_f1[4] = avg_f1
                if self.args.save_checkpoint:
                    save_checkpoint(model, 'saved_models/' + self.run_name + '.pth')

        if self.args.predict_km:
            print("KM MAE: " + str(km_mae))
            print("KM R2: " + str(km_r2))
            if km_mae < self.top_performances[6]:
                self.top_performances[6] = km_mae
    
        if self.args.predict_price:
            print("Price MAE: " + str(price_mae))
            print("Price Median Error: " + str(price_median_error))
            print("Price R2: " + str(price_r2))
            if price_mae < self.top_performances[7]:
                self.top_performances[7] = price_mae
                torch.save(model.state_dict(), 'saved_models/' + self.run_name + '.pth')
                print("Saved best model")

        print("=" * 100 + "\n")
        print("Top Performances")
        columns = []
        performances = []
        wandb_dict = {}
        if self.args.predict_body:
            columns.append('Body Accuracy')
            performances.append(self.top_performances[0])
            wandb_dict['Body Accuracy'] = body_accuracy
            wandb_dict['[MAX] Body Accuracy'] = self.top_performances[0]

            wandb_dict['Body F1'] = body_f1
            wandb_dict['[MAX] Body F1'] = self.top_f1[0]

        if self.args.predict_angle:
            # columns[0] = 'Angle MAE'
            columns.append('Angle MAE')
            performances.append(self.top_performances[0])
            wandb_dict['Angle MAE'] = angle_mae
            wandb_dict['[MAX] Angle MAE'] = self.top_performances[0]

        if self.args.predict_car_model:
            # columns[1] = 'Make Accuracy'
            # columns[2] = 'Model Accuracy'
            # columns[3] = 'Year Accuracy'
            columns.append('Make Accuracy')
            columns.append('Model Accuracy')
            columns.append('Year Accuracy')
            columns.append('AVG Accuracy')

            performances.append(self.top_performances[1])
            performances.append(self.top_performances[2])
            performances.append(self.top_performances[3])
            performances.append(self.top_performances[4])

            wandb_dict['Make Accuracy'] = make_accuracy
            wandb_dict['Model Accuracy'] = model_accuracy
            wandb_dict['Year Accuracy'] = year_accuracy
            wandb_dict['AVG Accuracy'] = avg_accuracy

            wandb_dict['[MAX] Make Accuracy'] = self.top_performances[1]
            wandb_dict['[MAX] Model Accuracy'] = self.top_performances[2]
            wandb_dict['[MAX] Year Accuracy'] = self.top_performances[3]
            wandb_dict['[MAX] AVG Accuracy'] = self.top_performances[4]

            wandb_dict['Make F1'] = make_f1
            wandb_dict['Model F1'] = model_f1
            wandb_dict['Year F1'] = year_f1
            wandb_dict['AVG F1'] = avg_f1

            wandb_dict['[MAX] Make F1'] = self.top_f1[1]
            wandb_dict['[MAX] Model F1'] = self.top_f1[2]
            wandb_dict['[MAX] Year F1'] = self.top_f1[3]
            wandb_dict['[MAX] AVG F1'] = self.top_f1[4]

        if self.args.predict_km:
            # columns[4] = 'KM MAE'
            columns.append('KM MAE')
            performances.append(self.top_performances[5])
            wandb_dict['KM MAE'] = km_mae
            wandb_dict['KM R2'] = km_r2

            wandb_dict['[MAX] KM MAE'] = self.top_performances[5]
            
        if self.args.predict_price:
            columns.append('Price MAE')
            performances.append(self.top_performances[6])
            wandb_dict['Price MAE'] = price_mae
            wandb_dict['Price Median Error'] = price_median_error
            wandb_dict['Price R2'] = price_r2

            wandb_dict['[MAX] Price MAE'] = self.top_performances[6]

        df = pd.DataFrame([performances], columns=columns)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

        wandb_dict['Test Loss'] = test_loss
        return wandb_dict