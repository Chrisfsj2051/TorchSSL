import numpy as np
import torch

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix
from torch.cuda import current_device
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib

from helpers import expected_calibration_error, maximum_calibration_error, average_calibration_error
from models.fixmatch.fixmatch import FixMatch
from .vc_utils import consistency_loss, consistency_loss_prob
from train_utils import ce_loss, EMA

from copy import deepcopy


class VC(FixMatch):

    def __init__(self, version, prob_warmup_iters, cali_warmup_iters, *args, **kwargs):
        super(VC, self).__init__(*args, **kwargs)
        self.version = version
        self.prob_warmup_iters = prob_warmup_iters
        self.cali_warmup_iters = cali_warmup_iters

    def before_train(self, args):
        self.model.train()
        self.model.version = self.version
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        self.best_eval_acc, self.best_it = 0.0, 0
        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        y_logits_vc = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            _, _, _, _, pseudo_labels_onehot = self.model(x, test_mode=False)
            y_logits_vc.extend(pseudo_labels_onehot.cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch

        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

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
                'ece': ece, 'mce': mce, 'ace': ace, 'ece_vc': ece_vc, 'mce_vc': mce_vc,
                'ace_vc': ace_vc}

    def train(self, args, logger=None):
        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)

        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter * args.epoch:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                (logits, recon_r, uncertainty, (mu, logvar), cali_output
                 ) = self.model(inputs, test_mode=False)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                _, recon_r_ulb_s = recon_r[num_lb:].chunk(2)
                _, uncertainty_ulb_s = uncertainty[num_lb:].chunk(2)
                _, mu_ulb_s = mu[num_lb:].chunk(2)
                _, logvar_ulb_s = logvar[num_lb:].chunk(2)
                cali_score_ulb_w, _ = cali_output[num_lb:].chunk(2)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
                cali_loss_weight = 0.0
                if self.prob_warmup_iters > 0:
                    unsup_loss, mask, select, pseudo_lb = consistency_loss(
                        logits_x_ulb_s, logits_x_ulb_w,
                        'ce', T, p_cutoff, use_hard_labels=args.hard_label)
                    self.prob_warmup_iters -= 1
                    self.cali_warmup_iters -= 1
                    assert self.cali_warmup_iters > self.prob_warmup_iters
                else:
                    cali_loss_weight = 5.0
                    if self.cali_warmup_iters > 0:
                        use_softmax = True
                        logits_x_ulb_w_ = logits_x_ulb_w
                        self.cali_warmup_iters -= 1
                    else:
                        logits_x_ulb_w_ = cali_score_ulb_w
                        use_softmax = False
                    unsup_loss, mask, select, pseudo_lb = consistency_loss_prob(
                        logits_x_ulb_s, logits_x_ulb_w_,
                        'ce', T, p_cutoff, use_hard_labels=args.hard_label,
                        use_softmax=use_softmax
                    )

                variation_loss = cali_loss_weight * (F.mse_loss(recon_r_ulb_s, uncertainty_ulb_s) * select).mean()
                kl_loss = cali_loss_weight * (-0.5 * torch.mean(1 + logvar_ulb_s - mu_ulb_s.pow(2) - logvar_ulb_s.exp()))

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
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)
                    self.save_model(f'iter_{self.it}.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
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
        # eval_dict = self.evaluate(args=args)
        # eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        # return eval_dict

    def load_model(self, load_path):
        checkpoint = torch.load(load_path,
                                map_location=lambda storage, loc: storage.cuda(current_device()))
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        try:
            self.model.load_state_dict(checkpoint['model'])
            self.ema_model = deepcopy(self.model)
            self.ema_model.load_state_dict(checkpoint['ema_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(e)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.ema_model = deepcopy(self.model)
            self.ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
            print(f'Warning: Optimizer is not loaded due to {e}')


if __name__ == "__main__":
    pass
