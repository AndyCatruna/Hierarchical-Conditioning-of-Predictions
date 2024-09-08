import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):
    def __init__(self, pred1_to_pred2_table):
        super(ConsistencyLoss, self).__init__()

        self.table = pred1_to_pred2_table
        for key in self.table.keys():
            self.table[key] = torch.tensor(self.table[key]).cuda()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred1_logits, pred2_logits):
        # Get the make predictions
        pred1 = pred1_logits.argmax(dim=1)

        # Get indices of the correct model predictions for each make in the batch
        pred2_correct_indices = [self.table[pred.item()] for pred in pred1]
        
        pred2_consistency_labels = torch.zeros(pred2_logits.size(0), pred2_logits.size(1), dtype=torch.float).cuda()

        for i in range(len(pred2_correct_indices)):
            current_correct_indices = pred2_correct_indices[i]
            
            # Transform current_correct_indices to a long tensor
            current_correct_indices = current_correct_indices.long()
            number_of_correct_indices = len(current_correct_indices)
            if number_of_correct_indices > 0:
                pred2_consistency_labels[i, current_correct_indices] = 1 / number_of_correct_indices

        loss = self.cross_entropy(pred2_logits, pred2_consistency_labels)

        return loss