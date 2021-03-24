import argparse
import torch
import torch.nn.parallel
import PIL.Image as Image
from models import modules, net, resnet, densenet, senet
import numpy as np
import time
import loaddata
import pdb
import matplotlib.pyplot as plt
import os
import pdb
import loaddata
import matplotlib.pyplot as plt
import os

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.misc import imresize
import os
from materics_evaluate import get_FM
from scipy.misc import imresize
import matplotlib.image
import matplotlib.pyplot as plt


import matplotlib.image
import matplotlib.pyplot as plt


plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048],in_dim=2048)
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def imsave(file_name, img, img_size):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert(ndim == 2 or ndim == 3,
           'img must be a 2 or 3 dimensional tensor')

    img = img.numpy()
    img = imresize(img, [img_size[1][0], img_size[0][0]], interp='nearest')
    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray')

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model_final = net.ConvA()
    model = model.cuda()
    model_final = model_final.cuda()
    # model.load_state_dict(torch.load('H:\code\Revisiting_Single_Depth_Estimation-master\checkpoint.pth.tar'))
    checkpoint = torch.load('H:\code\model_hci_test\hcicheckpointnew.pth.tar')
    checkpoint_final = torch.load('H:\code\model_end4\\hcifinalcheckpoint_1.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model_final.load_state_dict(checkpoint_final['state_dict_final'])
    model.eval()

    nyu2_loader = loaddata.getTestingData()
  
    val(nyu2_loader, model, model_final)
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

def val(nyu2_loader, model, model_final):
    with torch.no_grad():
     batch_time = AverageMeter()
     end = time.time()
     for i, (sample_batched,img_name, img_size) in  enumerate(nyu2_loader):
        image, depth, focal,depth1 = sample_batched['image'], sample_batched['depth'], sample_batched['focal'],sample_batched['depth1']
        image = torch.autograd.Variable(image, volatile=True).cuda()

        focal = focal.cuda()
        focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0)  # 2*12*3*256*256
        focal = focal.squeeze(1)
        out = model(image)
        output = model_final(out, focal)
        # out1 =out.cpu().data.resize_(64,64)
        out2 = output.cpu().data.resize_(256, 256)
        depth= depth.cpu().data.resize_(256, 256)
        imsave(os.path.join('D:\\4DLF\\additional\out2',img_name[0] + '.png'), out2, img_size)
        plt.imsave(os.path.join('D:\\4DLF\\additional\out2', img_name[0] + '_color.png'), out2.cpu().data.resize_(256, 256),
                   cmap='jet')
        batch_time.update(time.time() - end)
        end = time.time()
        print(
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'


              .format(batch_time=batch_time,))
        # imsave(os.path.join('H:\code\model_end4\\testdepth', img_name[0] + '.png'), depth, img_size)
        plt.imsave(os.path.join('H:\code\model_end4\color', img_name[0] + '.png'), depth.cpu().data.resize_(256,256), cmap='jet')
        # imsave(os.path.join('data/out1', img_name[0] + '.png'), out2, img_size)
        # plt.imsave(os.path.join('H:\code\model_end4\lfsddata\\fan1', img_name[0]  + '.png'), out2.cpu().data.resize_(256,256), cmap='r')
        # plt.imsave(os.path.join('data/guide1 - 副本', str(i) + '.png'), guide1.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide2', str(i) + '.png'), guide2.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide2 - 副本', str(i) + '.png'), guide2.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide3', str(i) + '.png'), guide3.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide3 - 副本', str(i) + '.png'), guide3.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide4', str(i) + '.png'), guide4.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide4 - 副本', str(i) + '.png'), guide4.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide5', str(i) + '.png'), guide5.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide5 - 副本', str(i) + '.png'), guide5.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide6', str(i) + '.png'), guide6.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide6 - 副本', str(i) + '.png'), guide6.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide7', str(i) + '.png'), guide7.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide7 - 副本', str(i) + '.png'), guide7.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide8', str(i) + '.png'), guide8.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide8 - 副本', str(i) + '.png'), guide8.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide9', str(i) + '.png'), guide9.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide9 - 副本', str(i) + '.png'), guide9.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide10', str(i) + '.png'), guide10.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide10 - 副本', str(i) + '.png'), guide10.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide11', str(i) + '.png'), guide11.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide11 - 副本', str(i) + '.png'), guide11.cpu().data.resize_(256, 256), cmap='gray')
        # plt.imsave(os.path.join('data/guide12', str(i) + '.png'), guide12.cpu().data.resize_(256, 256), cmap='jet')
        # plt.imsave(os.path.join('data/guide12 - 副本', str(i) + '.png'), guide12.cpu().data.resize_(256, 256), cmap='gray')

        # guide1=guide1[:,1,:,:]
        # matplotlib.image.imsave(os.path.join('data/guide1',str(i) + '.png'), guide1.view(guide1.size(1), guide1.size(2)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide2', str(i) + '.png'),
        #                         depth.view(guide2.size(2), guide2.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide3', str(i) + '.png'),
        #                         depth.view(guide3.size(2), guide3.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide4', str(i) + '.png'),
        #                         depth.view(guide4.size(2), guide4.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide5', str(i) + '.png'),
        #                         depth.view(guide5.size(2), guide5.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide6', str(i) + '.png'),
        #                         depth.view(guide6.size(2), guide6.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide7', str(i) + '.png'),
        #                         depth.view(guide7.size(2), guide7.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide8', str(i) + '.png'),
        #                         depth.view(guide8.size(2), guide8.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide9', str(i) + '.png'),
        #                         depth.view(guide9.size(2), guide9.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide10', str(i) + '.png'),
        #                         depth.view(guide10.size(2), guide10.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide11', str(i) + '.png'),
        #                         depth.view(guide11.size(2), guide11.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/guide12', str(i) + '.png'),
        #                         depth.view(guide12.size(2), guide12.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/out3', str(i) + '.png'),
        #                         depth.view(guide1.size(2), guide1.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave(os.path.join('data/demo/color',img_name[0]  + '.png'), out.view(out.size(2), out.size(3)).data.cpu().numpy())
        # matplotlib.image.imsave('data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
    torch.cuda.empty_cache()

    print("\n evaluating ....")
#     eva(
#         salpath='H:\code\model_end4\data' + '/',
#         gtpath='H:\code\model_end4\\testdepth' + '/')
#     # rms, mae = get_FM(salpath=MapRoot+'/', gtpath=test_dataRoot+'/test_depth1/')
#     # print('RMS:', rms)
#     # print('MAE:', mae)
#     avgmetrics = get_FM(
#         salpath='H:\code\model_end\data\out2' + '/',
#         gtpath='H:\code\cahnnel\data\demo\\test\depth' + '/')
#     print('MSE:', avgmetrics[0, 0])
#     print('RMS:', avgmetrics[0, 1])
#     print('Log RMS:', avgmetrics[0, 2])
#     print('Absolute relative:', avgmetrics[0, 3])
#     print('Squared relative:', avgmetrics[0, 4])
#     print('Accuracy-thr=1.25:', avgmetrics[0, 5])
#     print('Accuracy-thr=1.25**2:', avgmetrics[0, 6])
#     print('Accuracy-thr=1.25**3:', avgmetrics[0, 7])
#     print('BadPixelS:', avgmetrics[0, 8])
#     print('Bumpiness:', avgmetrics[0, 9])
#
#
# def eva(salpath, gtpath, ignore_zero=True):
#     gtdir = gtpath
#     depdir = salpath
#     files = os.listdir(gtdir)
#     eps = np.finfo(float).eps
#
#     delta1_accuracy = 0
#     delta2_accuracy = 0
#     delta3_accuracy = 0
#     rmse_linear_loss = 0
#     rmse_log_loss = 0
#     abs_relative_difference_loss = 0
#     squared_relative_difference_loss = 0
#
#     for i, name in enumerate(files):
#         if not os.path.exists(gtdir + name):
#             print(gtdir + name, 'does not exist')
#         gt = Image.open(gtdir + name)
#
#         gt = np.array(gt, dtype=np.uint8)
#         gt = (gt - gt.min()) / (gt.max() - gt.min() + eps)
#
#         gt = torch.from_numpy(gt).float()
#
#         pred = Image.open(depdir + name).convert('L')
#         pred = pred.resize((np.shape(gt)[1], np.shape(gt)[0]))
#         pred = np.array(pred, dtype=np.float)
#         pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)
#
#
#         pred = torch.from_numpy(pred).float()
#
#         # if len(pred.shape) != 2:
#         #    pred= pred[:, :, 0]
#         # pred= np.copy(pred)
#         #
#         if ignore_zero:
#             pred[gt == 0] = 0.0
#         #
#         # else:
#         #     numPixels = gt.size
#         delta1_accuracy += threeshold_percentage(pred, gt, 1.25)
#         delta2_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25)
#         delta3_accuracy += threeshold_percentage(pred, gt, 1.25 * 1.25 * 1.25)
#         rmse_linear_loss += rmse_linear(pred, gt)
#         rmse_log_loss += rmse_log(pred, gt)
#         abs_relative_difference_loss += abs_relative_difference(pred, gt)
#         squared_relative_difference_loss += squared_relative_difference(pred, gt)
#
#     delta1_accuracy /= (i + 1)
#     delta2_accuracy /= (i + 1)
#     delta3_accuracy /= (i + 1)
#     rmse_linear_loss /= (i + 1)
#     rmse_log_loss /= (i + 1)
#     abs_relative_difference_loss /= (i + 1)
#     squared_relative_difference_loss /= (i + 1)
#
#     # logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
#     # print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))
#     print(
#         '    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(
#             delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss,
#             rmse_log_loss,
#             abs_relative_difference_loss, squared_relative_difference_loss))
#
#
# def threeshold_percentage(output, target, threeshold_val):
#     output = output.view(1, 1, 256, 256)
#     target = target.view(1, 1, 256, 256)
#
#     d1 = torch.exp(output) / torch.exp(target)
#     d2 = torch.exp(target) / torch.exp(output)
#
#     # d1 = output/target
#     # d2 = target/output
#     max_d1_d2 = torch.max(d1, d2)
#     zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
#     one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
#     bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
#     count_mat = torch.sum(bit_mat, (1, 2, 3))
#     threeshold_mat = count_mat / (output.shape[2] * output.shape[3])
#     return threeshold_mat.mean()
#
#
# def rmse_linear(output, target):
#     output = output.view(1, 1, 256, 256)
#     target = target.view(1, 1, 256, 256)
#     actual_output = torch.exp(output)
#     actual_target = torch.exp(target)
#     # actual_output = output
#     # actual_target = target
#     diff = actual_output - actual_target
#     diff2 = torch.pow(diff, 2)
#     mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
#     rmse = torch.sqrt(mse)
#     return rmse.mean()
#
#
# def rmse_log(output, target):
#     output = output.view(1, 1, 256, 256)
#     target = target.view(1, 1, 256, 256)
#     diff = output - target
#     # diff = torch.log(output) - torch.log(target)
#     diff2 = torch.pow(diff, 2)
#     mse = torch.sum(diff2, (1, 2, 3)) / (output.shape[2] * output.shape[3])
#     rmse = torch.sqrt(mse)
#     return mse.mean()
#
#
# def abs_relative_difference(output, target):
#     output = output.view(1, 1, 256, 256)
#     target = target.view(1, 1, 256, 256)
#     actual_output = torch.exp(output)
#     actual_target = torch.exp(target)
#     # actual_output = output
#     # actual_target = target
#     abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
#     abs_relative_diff = torch.sum(abs_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
#     return abs_relative_diff.mean()
#
#
# def squared_relative_difference(output, target):
#     output = output.view(1, 1, 256, 256)
#     target = target.view(1, 1, 256, 256)
#     actual_output = torch.exp(output)
#     actual_target = torch.exp(target)
#     # actual_output = output
#     # actual_target = target
#     square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
#     square_relative_diff = torch.sum(square_relative_diff, (1, 2, 3)) / (output.shape[2] * output.shape[3])
#     return square_relative_diff.mean()


if __name__ == '__main__':
    main()
