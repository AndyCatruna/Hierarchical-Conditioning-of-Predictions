import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import random
from common import rand_bbox

class AutovitMultiTaskCars(Dataset):
    def __init__(self, data_path, csv_path, transform, mixup=False, masking=False, unified=False, dataset_percent=None):
        self.transform = transform
        self.data_path = data_path
        self.data = pd.read_csv(csv_path)
        self.num_classes = self.data.full_class.max() + 1
        self.num_make = self.data.make_class.max() + 1
        self.num_model = self.data.model_class.max() + 1
        self.num_year = self.data.year_class.max() + 1 
        
        if dataset_percent:
            print('Initial dataset size: ', len(self.data))
            self.data = self.reduce_dataset(dataset_percent)
            print('Reduced dataset size: ', len(self.data))

        self.make_model_hierarchy_table = None
        self.model_year_hierarchy_table = None
        if masking:
            self.make_model_hierarchy_table, self.model_year_hierarchy_table = self.construct_hierarchy_table(self.data)

        print('num_classes: ', self.num_classes, end=' ')
        print('num_make: ', self.num_make, end=' ')
        print('num_model: ', self.num_model, end=' ')
        print('num_year: ', self.num_year)
        self.mixup = mixup

        # Normalize data
        self.max_km = self.data.km.max()
        self.min_km = self.data.km.min()
        self.data['km'] = (self.data['km'] - self.min_km) / (self.max_km - self.min_km)

        self.max_price = self.data.price.max()
        self.min_price = self.data.price.min()
        self.data['price'] = (self.data['price'] - self.min_price) / (self.max_price - self.min_price)

        self.data['angle'] = self.data['angle'] / 360

        # For unified model
        self.lookup_table = None
        if unified:
            self.lookup_table = self.construct_lookup_table()
        else:
            self.lookup_table = self.construct_reverse_lookup_table()

    def oversample_data(self, data):
        mean_samples = data.full_class.value_counts().mean()
        mean_samples = int(mean_samples)
        unique_full_classes = data.full_class.unique()
        new_data = []
        for full_class in unique_full_classes:
            class_data = data[data.full_class == full_class]
            num_samples = len(class_data)
            if num_samples < mean_samples:
                to_sample = mean_samples - num_samples
                class_data = class_data.sample(n=to_sample, random_state=1, replace=True)
            new_data.append(class_data)
        
        return pd.concat(new_data)
    
    def reduce_dataset(self, dataset_percent):
        # Get the dataset percent for each full class
        unique_full_classes = self.data.full_class.unique()
        new_data = []
        for full_class in unique_full_classes:
            # Sample the dataset_percent for each class
            class_data = self.data[self.data.full_class == full_class]
            class_data = class_data.sample(frac=dataset_percent, random_state=1)
            new_data.append(class_data)

        return pd.concat(new_data)

    def reduce_balance_dataset(self, dataset_percent):
        num_samples_per_class = int(10 * dataset_percent)
        unique_full_classes = self.data.full_class.unique()
        new_data = []
        for full_class in unique_full_classes:
            # Sample the dataset_percent for each class
            class_data = self.data[self.data.full_class == full_class]
            to_sample = min(num_samples_per_class, len(class_data))
            class_data = class_data.sample(n=to_sample, random_state=1, replace=True)
            new_data.append(class_data)
        
        return pd.concat(new_data)

    def construct_lookup_table(self):
        lookup_table = {}
        for idx, row in self.data.iterrows():
            label = row.full_class
            make = row.make_class
            model = row.model_class
            year = row.year_class
            if label not in lookup_table:
                lookup_table[label] = {'make': make, 'model': model, 'body': None, 'year': year}

        for i in range(self.num_classes):
            if i not in lookup_table:
                # Add empty entry
                lookup_table[i] = {'make': self.num_make, 'model': self.num_model, 'body': None, 'year': self.num_year}

        return lookup_table

    def construct_reverse_lookup_table(self):
        lookup_table = {}
        for idx, row in self.data.iterrows():
            label = row.full_class
            make = row.make_class
            model = row.model_class
            year = row.year_class

            combined_label = (make, model, year)
            if combined_label not in lookup_table:
                lookup_table[combined_label] = label
        
        return lookup_table

    def __len__(self):
        return len(self.data)
    
    def construct_hierarchy_table(self, data):
        make_model_hierarchy_table = {}
        model_year_hierarchy_table = {}

        for i in range(len(data)):
            make = data.iloc[i].make_class
            model = data.iloc[i].model_class
            year = data.iloc[i].year_class

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
        
        for i in range(self.num_model):
            if i not in model_year_hierarchy_table:
                model_year_hierarchy_table[i] = []
        
        for i in range(self.num_year):
            if i not in make_model_hierarchy_table:
                make_model_hierarchy_table[i] = []

        return make_model_hierarchy_table, model_year_hierarchy_table

    def __mixup__(self, img, full_label, make_label, model_label, year_label, angle_label, km_label, price_label):
        mix_row = self.data.sample().iloc[0]
        mix_img_path = self.data_path + mix_row.img_path
        
        mix_img_label = mix_row.full_class
        mix_img_make_label = mix_row.make_class
        mix_img_model_label = mix_row.model_class
        mix_img_year_label = mix_row.year_class

        mix_label = torch.zeros(self.num_classes)
        mix_label[mix_img_label] = 1

        mix_label_make = torch.zeros(self.num_make)
        mix_label_make[mix_img_make_label] = 1

        mix_label_model = torch.zeros(self.num_model)
        mix_label_model[mix_img_model_label] = 1

        mix_label_year = torch.zeros(self.num_year)
        mix_label_year[mix_img_year_label] = 1

        mix_label_angle = mix_row.angle
        mix_label_km = mix_row.km
        mix_label_price = mix_row.price

        mix_img = Image.open(mix_img_path)

        if self.transform:
            mix_img = self.transform(mix_img)
        
        lam = random.random()
        image = lam * img + (1 - lam) * mix_img
        full_label = lam * full_label + (1 - lam) * mix_label
        make_label = lam * make_label + (1 - lam) * mix_label_make
        model_label = lam * model_label + (1 - lam) * mix_label_model
        year_label = lam * year_label + (1 - lam) * mix_label_year
        price_label = lam * price_label + (1 - lam) * mix_label_price
        km_label = lam * km_label + (1 - lam) * mix_label_km
        angle_label = (lam * angle_label + (1 - lam) * mix_label_angle) % 1

        price_label = np.float32(price_label)
        km_label = np.float32(km_label)

        return image, full_label, make_label, model_label, year_label, angle_label, km_label, price_label
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.data_path + row.img_path

        img_label = row.full_class
        make_label = row.make_class
        model_label = row.model_class
        year_label = row.year_class

        one_hot_full_label = torch.zeros(self.num_classes)
        one_hot_full_label[img_label] = 1

        one_hot_label_make = torch.zeros(self.num_make)
        one_hot_label_make[make_label] = 1

        one_hot_label_model = torch.zeros(self.num_model)
        one_hot_label_model[model_label] = 1

        one_hot_label_year = torch.zeros(self.num_year)
        one_hot_label_year[year_label] = 1

        img = Image.open(img_path)

        price = row.price
        km = row.km
        angle = row.angle

        # Convert to float
        price = np.float32(price)
        km = np.float32(km)

        if self.transform:
            img = self.transform(img)        

        if self.mixup and random.random() < 0.25:
            img, one_hot_full_label, one_hot_label_make, one_hot_label_model, one_hot_label_year, angle, km, price = self.__mixup__(img, one_hot_full_label, one_hot_label_make, one_hot_label_model, one_hot_label_year, angle, km, price)

        if self.mixup:
            output = {'img': img, 'make_label': one_hot_label_make, 'model_label': one_hot_label_model, 'year_label': one_hot_label_year, 'price': price, 'km': km, 'angle': angle, 'full_label': one_hot_full_label}
        else:
            output = {'img': img, 'make_label': make_label, 'model_label': model_label, 'year_label': year_label, 'price': price, 'km': km, 'angle': angle, 'full_label': img_label}

        return output