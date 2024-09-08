import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import random
from common import rand_bbox

class StanfordCarsMultiTask(Dataset):
    def __init__(self, data_path, csv_path, transform, mixup=False, cutmix=False, unified=False, masking=False):
        self.transform = transform
        self.data_path = data_path
        self.data = pd.read_csv(csv_path)
        self.num_classes = len(self.data.label.unique())
        self.num_make_classes = len(self.data.make.unique())
        self.num_model_classes = len(self.data.model.unique())
        self.num_body_classes = len(self.data.body.unique())
        self.num_year_classes = len(self.data.year.unique())
        self.mixup = mixup
        self.cutmix = cutmix

        self.num_make = self.num_make_classes
        self.num_model = self.num_model_classes
        self.num_year = self.num_year_classes
        self.body_make_hierarchy_table = None
        self.make_model_hierarchy_table = None
        self.model_year_hierarchy_table = None
        if masking:
            self.body_make_hierarchy_table, self.make_model_hierarchy_table, self.model_year_hierarchy_table = self.construct_hierarchy_table(self.data)

        # For unified model
        self.lookup_table = None
        if unified:
            self.lookup_table = self.construct_lookup_table()
        else:
            self.lookup_table = self.construct_reverse_lookup_table()

    def construct_lookup_table(self):
        lookup_table = {}
        for idx, row in self.data.iterrows():
            label = row.label - 1
            make = row.make
            model = row.model
            body = row.body
            year = row.year
            if label not in lookup_table:
                lookup_table[label] = {'make': make, 'model': model, 'body': body, 'year': year}
        return lookup_table


    def construct_reverse_lookup_table(self):
        lookup_table = {}
        for idx, row in self.data.iterrows():
            label = row.label - 1
            make = row.make
            model = row.model
            year = row.year

            combined_label = (make, model, year)
            if combined_label not in lookup_table:
                lookup_table[combined_label] = label
        
        return lookup_table

        
    def construct_hierarchy_table(self, data):
        body_make_hierarchy_table = {}
        make_model_hierarchy_table = {}
        model_year_hierarchy_table = {}

        for i in range(len(data)):
            body = data.iloc[i].body
            make = data.iloc[i].make
            model = data.iloc[i].model
            year = data.iloc[i].year

            if body not in body_make_hierarchy_table:
                body_make_hierarchy_table[body] = [make]
            else:
                if make not in body_make_hierarchy_table[body]:
                    body_make_hierarchy_table[body].append(make)
                
            if make not in make_model_hierarchy_table:
                make_model_hierarchy_table[make] = [model]
            else:
                if model not in make_model_hierarchy_table[make]:
                    make_model_hierarchy_table[make].append(model)
            
            if model not in model_year_hierarchy_table:
                model_year_hierarchy_table[model] = [year]
            else:
                if year not in model_year_hierarchy_table[model]:
                    model_year_hierarchy_table[model].append(year)
        
        for i in range(self.num_make):
            if i not in body_make_hierarchy_table:
                body_make_hierarchy_table[i] = []

        for i in range(self.num_model):
            if i not in model_year_hierarchy_table:
                model_year_hierarchy_table[i] = []
        
        for i in range(self.num_year):
            if i not in make_model_hierarchy_table:
                make_model_hierarchy_table[i] = []

        return body_make_hierarchy_table, make_model_hierarchy_table, model_year_hierarchy_table

    def __len__(self):
        return len(self.data)

    def __mixup__(self, img, make_label, model_label, body_label, year_label, full_label):
        mix_row = self.data.sample().iloc[0]

        mix_img_path, mix_make_label, mix_model_label, mix_body_label, mix_year_label, mix_full_label = '../stanford_cars_' + mix_row.path, int(mix_row.make), int(mix_row.model), int(mix_row.body), int(mix_row.year), int(mix_row.label - 1)
        
        mix_img_path = self.data_path + mix_row.path
        
        mix_make_one_hot_label = torch.zeros(self.num_make_classes)
        mix_make_one_hot_label[mix_make_label] = 1

        mix_model_one_hot_label = torch.zeros(self.num_model_classes)
        mix_model_one_hot_label[mix_model_label] = 1

        mix_body_one_hot_label = torch.zeros(self.num_body_classes)
        mix_body_one_hot_label[mix_body_label] = 1

        mix_year_one_hot_label = torch.zeros(self.num_year_classes)
        mix_year_one_hot_label[mix_year_label] = 1

        mix_full_one_hot_label = torch.zeros(self.num_classes)
        mix_full_one_hot_label[mix_full_label] = 1


        mix_img = Image.open(mix_img_path)

        if self.transform:
            mix_img = self.transform(mix_img)
        
        lam = random.random()
        image = lam * img + (1 - lam) * mix_img
        make_label = lam * make_label + (1 - lam) * mix_make_one_hot_label
        model_label = lam * model_label + (1 - lam) * mix_model_one_hot_label
        body_label = lam * body_label + (1 - lam) * mix_body_one_hot_label
        year_label = lam * year_label + (1 - lam) * mix_year_one_hot_label
        full_label = lam * full_label + (1 - lam) * mix_full_one_hot_label
        
        return image, make_label, model_label, body_label, year_label, full_label
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path, make_label, model_label, body_label, year_label, full_label = self.data_path + row.path, int(row.make), int(row.model), int(row.body), int(row.year), int(row.label - 1)
        make_one_hot_label = torch.zeros(self.num_make_classes)
        make_one_hot_label[make_label] = 1

        model_one_hot_label = torch.zeros(self.num_model_classes)
        model_one_hot_label[model_label] = 1

        body_one_hot_label = torch.zeros(self.num_body_classes)
        body_one_hot_label[body_label] = 1

        year_one_hot_label = torch.zeros(self.num_year_classes)
        year_one_hot_label[year_label] = 1

        full_one_hot_label = torch.zeros(self.num_classes)
        full_one_hot_label[full_label] = 1

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)        

        if self.mixup and random.random() < 0.25:
            img, make_one_hot_label, model_one_hot_label, body_one_hot_label, year_one_hot_label, full_one_hot_label = self.__mixup__(img, make_one_hot_label, model_one_hot_label, body_one_hot_label, year_one_hot_label, full_one_hot_label)

        if self.mixup:
            output = {'img': img, 'make_label': make_one_hot_label, 'model_label': model_one_hot_label, 'body_label': body_one_hot_label, 'year_label': year_one_hot_label, 'full_label': full_one_hot_label}
        else:
            output = {'img': img, 
                    'make_label': make_label,
                    'model_label': model_label,
                    'body_label': body_label,
                    'year_label': year_label,
                    'full_label': full_label,
                    }


        return output