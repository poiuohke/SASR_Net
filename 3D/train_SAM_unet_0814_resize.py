import argparse
import os
import numpy as np
from tqdm import tqdm
import torch

from dataloaders import make_data_loader
from modeling.SASR_Unet_3D import *
from utils.loss import SegmentationLosses, Dice_loss, SR_loss
from utils.fa_loss import FALoss
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

w_sr = 1.0
w_fa = 1.0
w_sr_seg = 0.2
w_adapt_fa = 0.0

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = SASR_Unet(in_channels=1, out_channels=1, final_sigmoid=True, sr=args.IS_SR)

        model = nn.DataParallel(model)
        if args.cuda:
            model.cuda()
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.best_pred = 0.0

        if args.pretrained:
            pretrained_dict = torch.load(args.pretrained)
            model.load_state_dict(pretrained_dict['state_dict'])

        # Clear start epoch if fine-t/uning////
        if args.ft:
            args.start_epoch = 0


    def training(self, epoch, sr):
        train_loss = 0.0
        loss_seg_all = 0.0
        loss_sr_all = 0.0
        loss_fa_all = 0.0
        loss_offset_fa_all = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if sr:
                image_size = [i // 2 for i in image.size()[2:]]
                input_img = torch.nn.functional.interpolate(image, size=image_size, mode='trilinear',
                                                            align_corners=True)
            else:
                input_img = image
                target = target
            if self.args.cuda:
                input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            if not sr:
                output = self.model(input_img)
                loss_seg = Dice_loss(output, target)
                loss_sr = torch.FloatTensor([0.0, ]).cuda()
                loss_fa = torch.FloatTensor([0.0, ]).cuda()
                loss_offset_fa = torch.FloatTensor([0.0, ]).cuda()

            else:
                output, output_sr, fea_seg_up, fea_sr_up, offset_seg, offset_sr = self.model(input_img)
                # 加入FALoss
                loss_seg = Dice_loss(output, target)
                loss_sr = SR_loss(output_sr, image, target)
                loss_fa = FALoss()(fea_seg_up, fea_sr_up)
                loss_offset_fa = FALoss()(offset_seg, offset_sr)

            loss = loss_seg + w_sr * loss_sr + w_fa * loss_fa + w_adapt_fa * loss_offset_fa #+ w_adapt_fa * loss_adapt_fa  # + w_sr_seg*loss_sr_seg

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            loss_seg_all += loss_seg.item()
            loss_sr_all += loss_sr.item()
            loss_fa_all += loss_fa.item()
            loss_offset_fa_all += loss_offset_fa.item()
            tbar.set_description('Train loss: %.3f, Dice loss: %.3f, SR loss: %.3f, FA loss: %.3f, Offet FA loss: %.3f' \
                                 % (train_loss / (i + 1), loss_seg_all / (i + 1), loss_sr_all / (i + 1), loss_fa_all / (i + 1),
                                    loss_offset_fa_all / (i + 1)))#
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/loss_seg_all_epoch', loss_seg_all / (i + 1), epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, filename='checkpoint_'+str(train_loss / (i + 1))+'.pth.tar')

    def validation(self, epoch, sr):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        Dice = 0.0
        count = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if sr:
                image_size = [i // 2 for i in image.size()[2:]]
                input_img = torch.nn.functional.interpolate(image, size=image_size, mode='trilinear',
                                                            align_corners=True)
            else:
                input_img = image
                target = target
            if self.args.cuda:
                input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()

            with torch.no_grad():
                if not sr:
                    output = self.model(input_img)
                else:
                    output, _, _, _, _, _ = self.model(input_img)

            loss = Dice_loss(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = pred[:, 0, :, :, :]
            target = target[:, 0, :, :, :]
            dice = 1 - loss.item()
            Dice = Dice + dice
            count = count + 1

        Dice_avg = Dice / count
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Dice: {}".format(Dice_avg))
        print('Loss: %.3f' % test_loss)

        new_pred = Dice_avg
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def test(self, epoch, sr):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        Dice = 0.0
        count = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if sr:
                image_size = [i // 2 for i in image.size()[2:]]
                input_img = torch.nn.functional.interpolate(image, size=image_size, mode='trilinear',
                                                            align_corners=True)
            else:
                input_img = image
                target = target
            if self.args.cuda:
                input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()

            with torch.no_grad():
                if not sr:
                    output = self.model(input_img)
                else:
                    output, _, _, _, _, _ = self.model(input_img)

            loss = Dice_loss(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = pred[:, 0, :, :, :]
            target = target[:, 0, :, :, :]
            dice = 1 - loss.item()
            Dice = Dice + dice
            count = count + 1

        Dice_avg = Dice / count
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Dice: {}".format(Acc, Acc_class, mIoU, FWIoU, Dice))
        print("Dice: {}".format(Dice_avg))
        print('Loss: %.3f' % test_loss)

        new_pred = Dice_avg
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch SASR-Unet Training")
    parser.add_argument('--backbone', type=str, default='-SASR',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--IS-SR', default=True,
                        help='use super resolution or not')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='nodule',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--data-path', type=str, default='/home2/LUNG_DATA/npy_96_LIDC/image/')
    parser.add_argument('--label-path', type=str, default='/home2/LUNG_DATA/npy_96_LIDC/label/')
    parser.add_argument('--train-data-list', type=str,
                        default='/home2/LUNG_DATA/LIDC-IDRI/train_nodules.csv')
    parser.add_argument('--val-data-list', type=str,
                        default='/home2/LUNG_DATA/LIDC-IDRI/val_nodules.csv')
    parser.add_argument('--test-data-list', type=str,
                        default='/home2/LUNG_DATA/npy_48/test/test_all_dia.csv')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
    parser.add_argument('--base-size', type=int, default=96,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=True,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 1000,
            'pascal': 50,
            'nodule': 100
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.005,
            'pascal': 0.007,
            'nodule': 0.0001,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'Unet-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch, args.IS_SR)
        if not trainer.args.no_val and (epoch % args.eval_interval == (args.eval_interval - 1)):
            trainer.validation(epoch, args.IS_SR)

    trainer.writer.close()
    trainer.validation(epoch,args.IS_SR)

if __name__ == "__main__":
    main()
