import argparse
import time
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import loaddata
import torch
import sobel
from models import modules, net, resnet, densenet, senet
from tensorboard_logger import Logger


torch.cuda.set_device(0)
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def main():
    global args
    args = parser.parse_args()
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model_final = net.modelfinal()
    model = model.cuda()
    model_final = model_final.cuda()
    batch_size = 1
    train_loader = loaddata.getTrainingData(batch_size)
    optimizer = torch.optim.Adam(model_final.parameters(), args.lr, weight_decay=args.weight_decay)
    logger = Logger(logdir='experiment_cnn', flush_secs=1)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model,model_final, optimizer, epoch,logger )

        if epoch % 10 == 0:
           save_checkpoint({'state_dict': model.state_dict()},filename='modelcheckpoint.pth.tar')
           save_checkpoint({'state_dict_final': model_final.state_dict()},filename='finalmodelcheckpoint.pth.tar')
           print('save: (epoch: %d)' % (epoch+ 1))


def train(train_loader,  model,model_final, optimizer, epoch,logger ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    #梯度与预先相似度
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()

    for i, sample_batched in enumerate(train_loader):
        total_step = len(train_loader)
        image, depth,focal,_ = sample_batched['image'], sample_batched['depth'], sample_batched['focal'],sample_batched['depth1']
        depth = depth.cuda()
        image = image.cuda()
        focal = focal .cuda()
        focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0)  # 2*12*3*256*256
        focal = focal.squeeze(1)

        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)

        optimizer.zero_grad()

        out = model(image)
        output = model_final(out, focal)

        # 梯度
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        # 法线
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # loss
        loss_depth = torch.log(torch.abs(output - depth) + 1.0).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1.0).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) +1.0).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        logger.log_value('loss',loss,step=i+(epoch-1)*total_step)

        losses.update(loss.item(), image.size(0))
        loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
 

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):

    torch.save(state, filename)



if __name__ == '__main__':
    main()
