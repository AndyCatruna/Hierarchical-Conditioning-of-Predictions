import torch
import torchvision.transforms as transforms
from datasets import *
from models import *
from trainers import *
from evaluators import *

def get_loaders(args):
	train_path = args.dataset_path + '/train.csv'
	test_path = args.dataset_path + '/test.csv'
	if args.dataset == 'autovit-multi':
		# train_path = '../autovit_dataset/train.csv'
		# test_path = '../autovit_dataset/test.csv'

		train_dataset=AutovitMultiTaskCars(args.dataset_path, train_path, transform=get_augmentation(args), mixup=args.mixup, masking=args.use_consistency_loss, unified = 'unified' in args.model, dataset_percent=args.dataset_percent)
		test_dataset=AutovitMultiTaskCars(args.dataset_path, test_path, transform=transforms.ToTensor())
	
	elif args.dataset == 'stanford-multi':
		# train_path = '../stanford_dataset/train.csv'
		# test_path = '../stanford_dataset/test.csv'

		train_dataset=StanfordCarsMultiTask(args.dataset_path, train_path, transform=get_augmentation(args), mixup=args.mixup, cutmix=args.cutmix, unified='unified' in args.model, masking=args.use_consistency_loss)
		test_dataset=StanfordCarsMultiTask(args.dataset_path, test_path, transform=transforms.ToTensor())

	elif args.dataset == 'car_model_dataset-multi':
		# train_path = '../car_model_dataset_dataset/train.csv'
		# test_path = '../car_model_dataset_dataset/test.csv'

		train_dataset=CarModelMultiTaskCars(args.dataset_path, train_path, transform=get_augmentation(args), mixup=args.mixup, masking=args.use_consistency_loss, unified = 'unified' in args.model, dataset_percent=args.dataset_percent)
		test_dataset=CarModelMultiTaskCars(args.dataset_path, test_path, transform=transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, pin_memory=True
	)
	
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True
	)
		
	return train_loader, test_loader

def get_scheduler(optimizer, args, steps):
	if args.scheduler == 'step':
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.95)
	elif args.scheduler == 'onecycle':
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps, epochs=args.epochs)
	
	return scheduler

def get_model(args, info=None):
	if info is not None:
		if 'autovit' in args.dataset or 'car_model_dataset' in args.dataset:
			make_classes, model_classes, year_classes, full_classes = info
		if args.dataset == 'stanford-multi':
			make_classes, model_classes, year_classes, body_classes, full_classes = info
	
	# Unified models
	if args.model == 'swin-small-unified':
		model = SwinSmallUnifiedModel(args=args, full_classes=full_classes)
	elif args.model == 'vit-small-unified':
		model = VitSmallUnifiedModel(args=args, full_classes=full_classes)

	# Separate models
	elif args.model == 'swin-small-separate':
		if args.dataset == 'stanford-multi':
			model = SwinSmallStanfordSeparateModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, body_classes=body_classes, full_classes=full_classes)
		else:
			model = SwinSmallSeparateModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, full_classes=full_classes)	
	elif args.model == 'vit-small-separate':
		if args.dataset == 'stanford-multi':
			model = VitSmallStanfordSeparateModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, body_classes=body_classes, full_classes=full_classes)
		else:
			model = VitSmallSeparateModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, full_classes=full_classes)

	
	# Hierarchical Conditioning models
	elif args.model == 'swin-small-conditional':
		if args.dataset == 'stanford-multi':
			model = SwinSmallConditionalStanfordModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, body_classes=body_classes, full_classes=full_classes)
		else:
			model = SwinSmallConditionalModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, full_classes=full_classes)
	elif args.model == 'vit-small-conditional':
		if args.dataset == 'stanford-multi':
			model = VitSmallConditionalStanfordModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, body_classes=body_classes, full_classes=full_classes)
		else:
			model = VitSmallConditionalModel(args=args, make_classes=make_classes, model_classes=model_classes, year_classes=year_classes, full_classes=full_classes)
	
	return model

def get_augmentation(args):
	if args.augmentation == 'no-aug':
		return transforms.Compose([transforms.ToTensor()])
	elif args.augmentation == 'weak-aug':
		return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=224, scale=(0.8, 1)), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.ToTensor()])
	elif args.augmentation == 'rand-aug':
		return transforms.Compose([transforms.RandAugment(), transforms.ToTensor()])
	elif args.augmentation == 'auto-aug':
		return transforms.Compose([transforms.AutoAugment(), transforms.ToTensor()])

class CircularMeanAbsoluteError(torch.nn.Module):
	def __init__(self):
		super(CircularMeanAbsoluteError, self).__init__()

	def forward(self, y_true, y_pred):
		# Convert angles to radians
		y_true_rad = torch.deg2rad(y_true * 360)
		y_pred_rad = torch.deg2rad(y_pred * 360)

		# Calculate the circular difference
		diff = torch.atan2(torch.sin(y_true_rad - y_pred_rad), torch.cos(y_true_rad - y_pred_rad))

		# Convert back to degrees and take the absolute value
		diff_deg = torch.rad2deg(diff)
		abs_diff_deg = torch.abs(diff_deg)

		abs_diff_deg = abs_diff_deg / 360

		return torch.mean(abs_diff_deg)

class CircularDistance(torch.nn.Module):
	def __init__(self):
		super(CircularDistance, self).__init__()

	def forward(self, y_true, y_pred):
		# Convert angles to radians
		y_true_rad = torch.deg2rad(y_true * 360)
		y_pred_rad = torch.deg2rad(y_pred * 360)

		# Calculate the circular difference
		diff = torch.atan2(torch.sin(y_true_rad - y_pred_rad), torch.cos(y_true_rad - y_pred_rad))

		# Convert back to degrees and take the absolute value
		diff_deg = torch.rad2deg(diff)
		abs_diff_deg = torch.abs(diff_deg)

		return torch.sum(abs_diff_deg)
	
def reverse_min_max_norm(x, min_val, max_val):
	return (x * (max_val - min_val)) + min_val

class RegressionMetric(torch.nn.Module):
	def __init__(self, min_val, max_val):
		super(RegressionMetric, self).__init__()
		self.min_val = min_val
		self.max_val = max_val

	def forward(self, y_true, y_pred):
		y_true_denormalized = reverse_min_max_norm(y_true, self.min_val, self.max_val)
		y_pred_denormalized = reverse_min_max_norm(y_pred, self.min_val, self.max_val)

		return torch.sum(torch.abs(y_true_denormalized - y_pred_denormalized))
	
	def denormalize(self, y_true, y_pred):
		y_true_denormalized = reverse_min_max_norm(y_true, self.min_val, self.max_val)
		y_pred_denormalized = reverse_min_max_norm(y_pred, self.min_val, self.max_val)

		return y_true_denormalized, y_pred_denormalized

def get_info(args, train_loader):
	info=None

	if 'autovit' in args.dataset or 'car_model_dataset' in args.dataset:
		make_classes = train_loader.dataset.num_make
		model_classes = train_loader.dataset.num_model
		year_classes = train_loader.dataset.num_year
		full_classes = train_loader.dataset.num_classes
		info = (make_classes, model_classes, year_classes, full_classes)
	if args.dataset == 'stanford-multi':
		num_make_classes = train_loader.dataset.num_make_classes
		num_model_classes = train_loader.dataset.num_model_classes
		num_year_classes = train_loader.dataset.num_year_classes
		num_body_classes = train_loader.dataset.num_body_classes
		num_full_classes = train_loader.dataset.num_classes
		info = (num_make_classes, num_model_classes, num_year_classes, num_body_classes, num_full_classes)

	return info

def get_trainer_evaluator(args, train_loader, test_loader, device, criterion, optimizer, scaler, scheduler, run_name):
	if args.trainer == 'multitask':
		criterion = (nn.CrossEntropyLoss(), CircularMeanAbsoluteError(), nn.MSELoss())
		if args.loss_weight == 'static':
			trainer = MultiTaskTrainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)
	else:
		trainer = ClassicTrainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)

	if args.trainer == 'multitask':
		if 'stanford' in args.dataset or 'car_model_dataset' in args.dataset:
			km_metric = None
			price_metric = None
			angle_metric = None
		else:
			price_metric = RegressionMetric(train_loader.dataset.min_price, train_loader.dataset.max_price)
			km_metric = RegressionMetric(train_loader.dataset.min_km, train_loader.dataset.max_km)
			angle_metric = CircularDistance()
		evaluator = MultiTaskEvaluator(args, test_loader, device, criterion, run_name, km_metric, price_metric, angle_metric, lookup_table=train_loader.dataset.lookup_table)
	else:
		evaluator = UnifiedEvaluator(args, test_loader, device, criterion, train_loader.dataset.lookup_table)
	
	return trainer, evaluator
