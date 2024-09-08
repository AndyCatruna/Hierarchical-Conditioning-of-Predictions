import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import save_checkpoint

class UnifiedEvaluator():
    def __init__(self, args, test_loader, device, criterion, lookup_table):
        self.args = args
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.lookup_table = lookup_table
        self.top_performances = [0, 0, 0, 0, 0, 0]
        self.top_f1 = [0, 0, 0, 0, 0, 0]

    def test(self, model):
        model.eval()
        total_val_loss = 0
        total = 0.0
        body_accuracy = 0.0
        make_accuracy = 0.0
        model_accuracy = 0.0
        year_accuracy = 0.0
        avg_accuracy = 0.0

        # For f1 score
        all_body_preds = []
        all_body_labels = []

        all_make_preds = []
        all_make_labels = []

        all_model_preds = []
        all_model_labels = []

        all_year_preds = []
        all_year_labels = []


        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images = data['img'].to(self.device)
                make_label = data['make_label'].to(self.device)
                model_label = data['model_label'].to(self.device)
                year_label = data['year_label'].to(self.device)
                full_label = data['full_label'].to(self.device)
                if 'stanford' in self.args.dataset:
                    body_label = data['body_label'].to(self.device)

                pred = model(images)

                loss = self.criterion(pred, full_label)
                
                total_val_loss += loss.item()

                _, full_predictions = torch.max(pred, 1)

                # Get body, make, model, year from lookup table
                if self.args.predict_body:
                    body_predicted = torch.tensor([self.lookup_table[full_predictions[i].item()]['body'] for i in range(full_predictions.size(0))]).to(self.device)
                    all_body_preds.extend(body_predicted.cpu().numpy())
                    all_body_labels.extend(body_label.cpu().numpy())

                make_predicted = torch.tensor([self.lookup_table[full_predictions[i].item()]['make'] for i in range(full_predictions.size(0))]).to(self.device)
                model_predicted = torch.tensor([self.lookup_table[full_predictions[i].item()]['model'] for i in range(full_predictions.size(0))]).to(self.device)
                year_predicted = torch.tensor([self.lookup_table[full_predictions[i].item()]['year'] for i in range(full_predictions.size(0))]).to(self.device)

                all_make_preds.extend(make_predicted.cpu().numpy())
                all_make_labels.extend(make_label.cpu().numpy())

                all_model_preds.extend(model_predicted.cpu().numpy())
                all_model_labels.extend(model_label.cpu().numpy())

                all_year_preds.extend(year_predicted.cpu().numpy())
                all_year_labels.extend(year_label.cpu().numpy())

                total += make_label.size(0)

                if self.args.predict_body:
                    body_accuracy += (body_predicted == body_label).sum().item()

                if self.args.predict_car_model:
                    make_accuracy += (make_predicted == make_label).sum().item()
                    model_accuracy += (model_predicted == model_label).sum().item()
                    year_accuracy += (year_predicted == year_label).sum().item()

        test_loss = np.round(total_val_loss / len(self.test_loader), 2)


        if self.args.predict_body:
            body_accuracy = np.round(100 * body_accuracy / total, 2)
            body_f1 = f1_score(all_body_labels, all_body_preds, average='weighted')

        if self.args.predict_car_model:
            make_accuracy = np.round(100 * make_accuracy / total, 2)
            model_accuracy = np.round(100 * model_accuracy / total, 2)
            year_accuracy = np.round(100 * year_accuracy / total, 2)
            avg_accuracy = np.round((make_accuracy + model_accuracy + year_accuracy) / 3, 2)

            make_f1 = np.round(f1_score(all_make_labels, all_make_preds, average='weighted'), 4)
            model_f1 = np.round(f1_score(all_model_labels, all_model_preds, average='weighted'), 4)
            year_f1 = np.round(f1_score(all_year_labels, all_year_preds, average='weighted'), 4)
            avg_f1 = np.round((make_f1 + model_f1 + year_f1) / 3, 4)

        print("Test Loss: " + str(test_loss))
        
        if self.args.predict_body:
            print("Body Accuracy: " + str(body_accuracy))
            if body_accuracy > self.top_performances[0]:
                self.top_performances[0] = body_accuracy

            if body_f1 > self.top_f1[0]:
                self.top_f1[0] = body_f1

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

            if make_f1 > self.top_f1[1]:
                self.top_f1[1] = make_f1
            if model_f1 > self.top_f1[2]:
                self.top_f1[2] = model_f1
            if year_f1 > self.top_f1[3]:
                self.top_f1[3] = year_f1
            if avg_f1 > self.top_f1[4]:
                self.top_f1[4] = avg_f1
                if self.args.save_checkpoint:
                    save_checkpoint(model, 'saved_models/' + self.args.run_name + '.pth')

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
            
        df = pd.DataFrame([performances], columns=columns)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

        wandb_dict['Test Loss'] = test_loss
        return wandb_dict