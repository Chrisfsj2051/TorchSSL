import json

from torch.utils.data import DataLoader

DataLoader
import numpy as np
import torch

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report
from torch.cuda import current_device
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib

import torch.nn as nn

from helpers import expected_calibration_error, maximum_calibration_error, average_calibration_error
from models.flexmatch.flexmatch import FlexMatch
from .vc_utils import consistency_loss, consistency_loss_prob
from train_utils import ce_loss, EMA

from copy import deepcopy


class VC(FlexMatch):

    def __init__(self,
                 save_interval,
                 version,
                 pseudo_alg,
                 uncertainty_alg,
                 pseudo_alg_warmup_iter,
                 cali_train_warmup_iter,
                 use_cali_out_warmup_iter,
                 cali_loss_weight,
                 cali_loss_type,
                 lb_strong_aug_warmup_iter,
                 *args, **kwargs):
        super(VC, self).__init__(*args, **kwargs)
        self.save_interval = save_interval
        self.version = version
        self.pseudo_alg_warmup_iter = pseudo_alg_warmup_iter
        self.cali_train_warmup_iter = cali_train_warmup_iter
        self.cali_loss_weight = cali_loss_weight
        self.pseudo_alg = pseudo_alg
        self.uncertainty_alg = uncertainty_alg
        self.cali_loss_type = cali_loss_type
        self.use_cali_out_warmup_iter = use_cali_out_warmup_iter
        self.lb_strong_aug_warmup_iter = lb_strong_aug_warmup_iter


    def choose_pseudo_alg(self):
        if self.pseudo_alg_warmup_iter > self.it:
            return consistency_loss
        elif self.pseudo_alg == 'flexmatch':
            return consistency_loss
        elif self.pseudo_alg == 'probabilitic':
            return consistency_loss_prob
        else:
            return NotImplementedError

    def before_train(self, args):
        self.model.train()
        self.model.version = self.version
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)
        self.best_eval_acc, self.best_it = 0.0, 0
        if args.resume == True:
            eval_dict, _ = self.evaluate(args=args)
            self.print_fn(
                f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, "
                f"{eval_dict}, at {self.best_it} iters")
        self.p_model = None
        self.p_target = None
        if not hasattr(self, 'selected_label'):
            self.selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
            self.classwise_acc = torch.zeros((args.num_classes,))
        self.selected_label = self.selected_label.cuda(args.gpu)
        self.classwise_acc = self.classwise_acc.cuda(args.gpu)

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, is_pseudo_test=False):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_pred_vc = []
        y_logits = []
        y_logits_vc = []
        for pair in eval_loader:
            _, x, _, y = pair
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _, _, _, logits_vc = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_pred_vc.extend(torch.max(logits_vc, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            y_logits_vc.extend(logits_vc.softmax(dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch

        top1 = accuracy_score(y_true, y_pred)
        top1_vc = accuracy_score(y_true, y_pred_vc)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cls_report = classification_report(y_true, y_pred)
        confs_vc = torch.FloatTensor([max(x) for x in y_logits_vc])
        confs = torch.FloatTensor([max(x) for x in y_logits])
        ece_vc = expected_calibration_error(
            confs_vc, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        mce_vc = maximum_calibration_error(
            confs_vc, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        ace_vc = average_calibration_error(
            confs_vc, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        ece = expected_calibration_error(
            confs, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        mce = maximum_calibration_error(
            confs, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        ace = average_calibration_error(
            confs, torch.LongTensor(y_pred), torch.LongTensor(y_true)).item()
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC,
                'ece': ece, 'mce': mce, 'ace': ace,
                'top1_vc': top1_vc,
                'ece_vc': ece_vc, 'mce_vc': mce_vc, 'ace_vc': ace_vc}, cls_report

    def calc_holdout_logits(self, args, logger=None):
        if self.loader_dict['holdout'] is None:
            return
        holdout_per_class = args.num_holdout // self.num_classes
        self.holdout_logits = torch.zeros(self.num_classes, self.num_classes).cuda(args.gpu)
        for (_, x, y) in self.loader_dict['holdout']:
            with torch.no_grad():
                x, y = x.cuda(args.gpu), y.cuda(args.gpu)
                logits, recon_r, uncertainty, (mu, logvar), cali_output = self.model(x)
                logits = logits.softmax(-1).reshape(self.num_classes, holdout_per_class, -1)
                self.holdout_logits += logits.sum(1)
        self.holdout_logits /= holdout_per_class

    def train(self, args, logger=None):
        ngpus_per_node = torch.cuda.device_count()
        self.model.train()
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (_, x_lb, x_lb_s, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s, _) in zip(
                self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            if self.it > args.num_train_iter * args.epoch:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            if self.it > self.lb_strong_aug_warmup_iter:
                x_lb = torch.cat([x_lb, x_lb_s], 0)
                y_lb = torch.cat([y_lb, y_lb], 0)

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            pseudo_counter = Counter(self.selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                (logits, recon_r, uncertainty, (mu, logvar), cali_output
                 ) = self.model(inputs)

                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                recon_r_ulb_w, recon_r_ulb_s = recon_r[num_lb:].chunk(2)
                uncertainty_ulb_w, uncertainty_ulb_s = uncertainty[num_lb:].chunk(2)
                mu_ulb_w, mu_ulb_s = mu[num_lb:].chunk(2)
                logvar_ulb_w, logvar_ulb_s = logvar[num_lb:].chunk(2)
                cali_score_ulb_w, cali_score_ulb_s = cali_output[num_lb:].chunk(2)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                if self.cali_train_warmup_iter > self.it:
                    cali_loss_weight = 0
                else:
                    cali_loss_weight = self.cali_loss_weight

                if self.use_cali_out_warmup_iter > self.it:
                    preds_x_ulb_w = logits_x_ulb_w
                else:
                    preds_x_ulb_w = cali_score_ulb_w

                if self.use_cali_out_warmup_iter == self.it:
                    self.selected_label = (torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1).cuda(args.gpu)
                    self.classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
                    pseudo_counter = Counter(self.selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                        for i in range(args.num_classes):
                            self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

                unsup_loss, mask, select, pseudo_lb, self.p_model = self.choose_pseudo_alg()(
                    logits_x_ulb_s,
                    preds_x_ulb_w,
                    self.classwise_acc,
                    self.p_target,
                    self.p_model,
                    'ce', T, p_cutoff,
                    use_hard_labels=args.hard_label,
                    use_DA=False
                )

                if x_ulb_idx[select == 1].nelement() != 0:
                    self.selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

                if self.cali_loss_type == 'mse':
                    variation_loss = F.mse_loss(recon_r_ulb_w.softmax(1), uncertainty_ulb_w) * mask
                else:
                    recon_r_ulb_w_log_softmax = torch.log_softmax(recon_r_ulb_w, -1)
                    variation_loss = torch.mean(-uncertainty_ulb_w * recon_r_ulb_w_log_softmax, 1) * mask
                kl_loss = -0.5 * torch.sum(1 + logvar_ulb_w - mu_ulb_w ** 2 - logvar_ulb_w.exp(), dim=1) * mask
                variation_loss = cali_loss_weight * variation_loss.mean()
                kl_loss = cali_loss_weight * kl_loss.mean()
                total_loss = sup_loss + self.lambda_u * unsup_loss + variation_loss + kl_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/recon_loss'] = variation_loss.detach()
            tb_dict['train/kl_loss'] = kl_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.mean().detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % 1000 == 0:
                self.calc_holdout_logits(args, logger)

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % self.save_interval == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)
                    self.save_model(f'iter_{self.it}.pth', save_path)

            if self.it % (self.num_eval_iter * 2) == 0:
                ulb_eval_dict, cls_report = self.evaluate(
                    eval_loader=self.loader_dict['eval_ulb'], args=args, is_pseudo_test=True)
                self.print_fn(
                    f"[ULB EVAL] {self.it} iteration, USE_EMA: {self.ema_m != 0}, "
                    f"{ulb_eval_dict}, at {self.best_it} iters")
                self.print_fn(f"[ULB EVAL]:\n{cls_report}")

            if self.it % self.num_eval_iter == 0:
                eval_dict, _ = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    self.best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, "
                    f"{tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, "
                    f"at {self.best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == self.best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter * args.epoch:
                self.num_eval_iter = 1000

        return

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'it': self.it,
                'ema_model': ema_model,
                'selected_label': self.selected_label.cpu(),
                'classwise_acc': self.classwise_acc.cpu()
            },
            save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path,
                                map_location=lambda storage, loc: storage.cuda(current_device()))
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.selected_label = checkpoint['selected_label']
        self.classwise_acc = checkpoint['classwise_acc']

        try:
            self.model.load_state_dict(checkpoint['model'])
            self.ema_model = deepcopy(self.model)
            self.ema_model.load_state_dict(checkpoint['ema_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            self.print_fn(e)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.ema_model = deepcopy(self.model)
            self.ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
            self.print_fn(f'Warning: Optimizer is not loaded due to {e}')


if __name__ == "__main__":
    pass
