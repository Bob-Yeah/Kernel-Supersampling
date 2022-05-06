import os
import math
from decimal import Decimal

import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm
import time

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        if (not args.test_only):
            self.optimizer = utility.make_optimizer(args, self.model)
            self.scheduler = utility.make_scheduler(args, self.optimizer)

            if self.args.load != '.':
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
                )
                for _ in range(len(ckp.log)): self.scheduler.step()

            self.error_last = 1e8
        else:
            if self.args.load != '.':
                # model_ = torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
                # for i in model_.keys(): print(i)
                # print("##################################################")
                # for i in self.model.state_dict().keys(): print(i)
                # print(self.model.state_dict())
                self.model.get_model().load_state_dict(
                    torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
                )
                print('Total params: %.2fK' % (sum(p.numel() for p in self.model.get_model().parameters())/1000.0))
                # self.model = torch.load(os.path.join(ckp.dir, 'optimizer.pt'))

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        # torch.autograd.set_detect_anomaly(True)
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, name) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            # print(lr.shape)
            # print(hr.shape)
            timer_data.hold()
            timer_model.tic()
            N,C,H,W = lr.size()
            _,_,outH,outW = hr.size()

            self.optimizer.zero_grad()
            sr = self.model(lr[:,0:self.args.n_colors,:,:])
            # m = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # ulr = m(lr)
            # sr = sr + ulr

            # print(lr.shape)
            # print(hr.shape)
            # print(sr.shape)
            if (epoch % 10 == 1):
                import scipy.misc as misc
                import numpy as np
                import skimage.color as sc

                normalized = sr[0].data.mul(255)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                hr_norm = hr[0].data.mul(255)
                hr_img = hr_norm.byte().permute(1, 2, 0).cpu().numpy()
                # out = np.concatenate([ndarr,ndarr,ndarr], 2)
                # # out = sc.ycbcr2rgb(out)
                out = ndarr
                # # high = sc.ycbcr2rgb(hr_img)
                # high = np.concatenate([hr_img,hr_img,hr_img], 2)
                high = hr_img
                misc.imsave('saveout_run_y.png', out)
                misc.imsave('saveout_hr_y.png', high)

            loss = self.loss(sr, hr[:,0:self.args.n_colors,:,:])
            
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model  #.module
        if (epoch % 10 == 0):

            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
            )
            ## save models

    def test(self):  
        import scipy.misc as misc
        def _save_results(filename, save_list, scale):
            filename = '{}/results/{}_x{}_'.format(self.ckp.dir, filename, scale)
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].data.mul(self.args.rgb_range)
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                misc.imsave('{}{}.png'.format(filename, p), ndarr)
        
        self.model.eval()
        timer_test = utility.timer()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_acc_ssim = 0

                for idx_img, (lr, hr, filename) in enumerate(self.loader_test):
                    filename = filename[0]
                    print("test file:", filename)
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    N,C,H,W = lr.size()
                    scale = int(self.args.scale)
                    
                    timer_test.tic()
                    start_time = time.time()

                    sr = self.model(lr[:,0:self.args.n_colors,:,:])

                    # your code
                    elapsed_time = time.time() - start_time
                    print("duration:",elapsed_time)
                    m = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    ulr = m(lr)
                    # sr = ulr + sr

                    print(lr.shape)
                    print(hr.shape)
                    print(sr.shape)

                    timer_test.hold()

                    save_list = [sr]
                    if not no_eval:
                        acc_i = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc += acc_i

                        print(acc_i)
                        
                        # eval_acc_ssim += utility.calc_ssim(
                        #     sr, hr, scale,
                        #     benchmark=self.loader_test.dataset.benchmark
                        # )

                        # print(eval_acc_ssim)
                        # print(utility.calc_ssim(
                        #     lr, hr, scale,
                        #     benchmark=self.loader_test.dataset.benchmark
                        # ))
                        save_list.extend([ulr, hr])
                    _save_results(filename,save_list,scale)
                    # print(save_list)
                    
                # print('[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                #     self.args.data_test,
                #     scale,
                #     eval_acc  / len(self.loader_test),
                #     eval_acc_ssim / len(self.loader_test)
                # ))

                print('[{} x{}]\tPSNR: {:.3f}'.format(
                    self.args.data_test,
                    scale,
                    eval_acc  / len(self.loader_test),
                ))
                    


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs