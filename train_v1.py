import copy
from dataset import train_dataloader, val_dataloader
from model import efficientnet_b0
import torch
from utils import EarlyStopping
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1, step_size=10)
es = EarlyStopping(patience=5, min_delta=1e-5)

def train_model(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, early_stopping=None):
    since = time.time()
    training_accuracies, training_losses, val_accuracies, val_losses = [], [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            data_size = len(dataloader)
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Training'):
                    y_logits = model(inputs)
                    y_preds = y_logits.argmax(dim=1)
                    loss = criterion(y_logits, labels)
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_corrects += (y_preds == labels).sum().item() / len(y_preds)
            if phase == 'Training':
                scheduler.step()
            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects / data_size
            if phase == 'Training':
                training_accuracies.append(epoch_acc)
                training_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc)
                val_losses.append(epoch_loss)
            if phase == 'Validation' and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print('{} Loss after epoch: {:.4f}, Acc after epoch: {:.4f}\n'.format(
            phase, epoch_loss, epoch_acc))
        if early_stopping is not None:
            early_stopping(val_losses[-1])
            if early_stopping.early_stopping:
                print('Early Stopping Initiated')
                break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, training_accuracies, training_losses, val_accuracies, val_losses

if __name__ == '__main__':
    model_ft, training_accuracies, training_losses, val_accuracies, val_losses = train_model(model=model,
                                                                                         train_loader= train_dataloader,
                                                                                         val_loader=val_dataloader,
                                                                                         criterion=criterion,
                                                                                         optimizer=optimizer,
                                                                                         num_epochs=100,
                                                                                         scheduler=exp_lr_scheduler,
                                                                                         early_stopping=es)


