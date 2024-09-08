import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
import wandb
import random

from common import *
from utils import *

# Define the arguments
args = define_args()
config = vars(args)

# Set the seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the run name
if not args.run_name:
	run_name = args.model + '_' + str(args.exp_index)
else:
	run_name = args.run_name

# Wandb
if args.wandb_log:
	project_name = 'multitask_cars'

	wandb.init(project=project_name, group=args.wandb_group)
	wandb.run.name = run_name
	wandb.config.update({'config': config})

# Hyperparameters
workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
train_loader, test_loader = get_loaders(args)

# Use mixed precision
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True

# Additional info for multi-task datasets
info = get_info(args, train_loader)

# Model
model = get_model(args, info)
model.to(device)

# Training Helpers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_scheduler(optimizer, args, len(train_loader))
trainer, evaluator = get_trainer_evaluator(args, train_loader, test_loader, device, criterion, optimizer, scaler, scheduler, run_name)

print("\n" + "-" * 50)
print(run_name)
print("-" * 50 + "\n")

# Training
for epoch in range(epochs):
	print("EPOCH " + str(epoch))
	
	train_loss = trainer.train(model)
	
	if args.save_checkpoint:
		torch.save(model.state_dict(), 'saved_models/' + args.run_name + '.pth')
		print("Saved last model")
	
	log_dict = evaluator.test(model)

	current_lr = trainer.scheduler.get_last_lr()[0]

	log_dict["Train Loss"] = train_loss
	log_dict["LR"] = current_lr
	
	if args.wandb_log:
		wandb.log(log_dict)

wandb.finish()