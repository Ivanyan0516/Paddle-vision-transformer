import paddle
import paddle.nn as nn
from resnet import ResNet18
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=10):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image) #inference
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = nn.functional.softmax(out, axis=1)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc.cpu().numpy()[0], batch_size)
        if batch_id > 0 and batch_id % report_freq == 0:
            print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')
        # if batch_id > 0:
        #     print(f'------ Batch {batch_id}, loss={loss_meter.avg}, acc={acc_meter.avg}')
    print(f'----- Epoch[{epoch}/{total_epoch}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')

def validation(model, dataloader, criterion, report_freq=10):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)  # inference
        loss = criterion(out, label)

        pred = nn.functional.softmax(out, axis=1)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc.cpu().numpy()[0], batch_size)
        # if batch_id > 0:
        #     print(f'------ Batch {batch_id}, loss={loss_meter.avg}, acc={acc_meter.avg}')
        if batch_id > 0 and batch_id % report_freq == 0:
            print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')
    print(f'----- Validation Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')

def main():
    total_epoch = 20
    batch_size = 16

    model = ResNet18(num_classes=10)

    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(train_dataset, mode='train', batch_size=batch_size)
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(val_dataset, mode='test', batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch=total_epoch)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)

    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                          parameters=model.parameters(),
                                          momentum=0.9,
                                          weight_decay=5e-4)

    for epoch in range(1, total_epoch+1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, total_epoch)
        scheduler.step()
        validation(model, val_dataloader, criterion)



if __name__ == '__main__':
    main()
