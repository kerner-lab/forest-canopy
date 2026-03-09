"""
Training loop
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

def regression_loop(model,
                        model_name,
                        epochs,
                        train_dl,
                        val_dl,
                        save_dir,
                        wandb_exp,
                        device,
                        optimizer,
                        criterion,
                        waves=None):
    """
    Training loop for fine-tuning
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_step = 0
    epoch_number = 0

    for epoch in range(epochs):
        print('Training epoch {}'.format(epoch + 1))
        train_correct = 0
        train_total = 0
        model.train()
        total_loss = 0
        val_total_loss = 0
        epoch_number +=1

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            if model_name.lower() == 'dofa':
                logits = model(imgs, waves)
            else:
                logits = model(imgs)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1

            total_loss += loss.item()

        train_acc = train_correct / train_total
        print(f"train accuracy={train_acc:.4f}")
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch + 1}: train loss={avg_loss:.4f}")

        wandb_exp.log({
            'train loss': avg_loss,
            'train acc': train_acc,
            'epoch': epoch,
            'step': global_step
        })

        if val_dl is not None and len(val_dl) > 0:
            # validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)

                    if model_name.lower() == 'dofa':
                        logits = model(imgs, waves)
                    else:
                        logits = model(imgs)
                    preds = logits.argmax(dim=1)

                    vloss = criterion(logits, labels)
                    val_total_loss += vloss.item()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            print(f"val accuracy={val_acc:.4f}")
            avg_vloss = val_total_loss / len(val_dl)

            wandb_exp.log({
                'val loss': avg_vloss,
                'val acc': val_acc,
                'epoch': epoch,
                'step': global_step
            })


def finetune(model,
             model_name,
             train_dataset,
             val_dataset,
             finetune,
             save_dir,
             wandb_exp,
             epochs: int,
             batch_size: int,
             learning_rate: float,
             weight_decay: float = 0,
             ):

    # --- Split dataset into training and validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name.lower() == 'croma':
        for _ in range(3):
            model.to(device)
            for module in model.modules():
                module.to(device)

    # --- DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if val_dataset is not None:
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_dl = None

    # Maintain criterion between models
    class_criterion = torch.nn.MSELoss()

    # --- Freeze backbone if just want to train head of model
    if finetune == 'head':
        # freeze backbone
        for p in model.parameters():
            p.requires_grad = False
        # Backbone-specific head unfreeze
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        elif hasattr(model, "fc"):
            # ResNet models
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):
            # Clay, Galileo
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "decode_head"):
            # Dinov3
            for p in model.decode_head.parameters():
                p.requires_grad = True
        elif hasattr(model, "decoder"):
            # Terrafm
            for p in model.decoder.parameters():
                p.requires_grad = True
    elif finetune == "full":
        # Unfreeze everything for full finetuning
        for p in model.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # put everything on device
    model.to(device)

    # Run regression loop
    regression_loop(model, model_name, epochs, train_dl, val_dl, save_dir, wandb_exp, device,
                            optimizer, class_criterion, waves)
