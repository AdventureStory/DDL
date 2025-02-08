import sys
sys.path.append("..")
import argparse
import logging
import math
import os
import random
import time
import json
import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models
import torch.nn as nn

from data import DATASET_GETTERS
from models_ema import ModelEMA
from models import resnet18
from exp_utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint2, reduce_tensor, model_load_state_dict)
from mmd import get_MMD

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
time_stamp = str(int(time.time()))

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='18_for_teacher_lr3', help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./model', type=str, help='save path')
parser.add_argument('--total-steps', default=50000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=7, type=int, help='number of classes')
parser.add_argument('--resize', default=224, type=int, help='resize image')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.001, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=32, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-3, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=5, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--lambda-mmd', default=0.2, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--w_mmd2', default=0.1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=2, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--log", type=str, default='/mnt/beegfs/home/cv/dai.guan/lyq/Master/Face2Exp/log',
                    help="For distributed training: local_rank")
parser.add_argument("--pretrain_model", type=str, default='/mnt/beegfs/home/cv/dai.guan/lyq/Master/Face2Exp/pretrain_model/resnet18_msceleb.pth',
                    help="For distributed training: local_rank")
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def makeDirs(root):
    if not os.path.exists(root):
        os.makedirs(root)

def config_save(args):
    makeDirs(args.log)
    json_path = os.path.join(args.log, args.name+'.json')
    
    
    args_dict = {}

    for arg in vars(args):
        if arg == "writer":
            continue
        args_dict[arg] = getattr(args, arg)
        
    args_dict['device'] = str(args_dict['device'])
    with open(json_path,'w') as write_f:
        json.dump(args_dict,write_f,indent = 4, ensure_ascii=False)
 
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader_t,labeled_loader_s, unlabeled_loader, test_loader, val_loader_t, val_loader_s, all_labeled_loader,
               teacher_model, student_model, avg_teacher_model,avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = Experiment basic")
    logger.info(f"   Total steps = {args.total_steps}")

    record_file = f'experiment/experiment_{args.name}_{time_stamp}/'
    args.save_path = record_file
    if not os.path.exists(record_file):
        os.makedirs(record_file)

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        val_epoch = 0
        
        labeled_loader_t.sampler.set_epoch(labeled_epoch)
        labeled_loader_s.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
        val_loader_t.sampler.set_epoch(val_epoch)
        val_loader_s.sampler.set_epoch(val_epoch)

    labeled_iter_t = iter(labeled_loader_t)
    labeled_iter_s = iter(labeled_loader_s)
    unlabeled_iter = iter(unlabeled_loader)

    val_iter_t = iter(val_loader_t)
    val_iter_s = iter(val_loader_s)
    all_labeled_iter = iter(all_labeled_loader)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            t_losses_mmd = AverageMeter()

            s_losses_l = AverageMeter()
            s_losses_u = AverageMeter()
            s_losses_mpl = AverageMeter()
            s_losses_mmd = AverageMeter()

            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()
        if step % 2 == 0:
            labeled_iter = labeled_iter_t
            labeled_loader = labeled_loader_t
            val_iter = val_iter_t
            val_loader = val_loader_t

        else:
            labeled_iter = labeled_iter_s
            labeled_loader = labeled_loader_s
            val_iter = val_iter_s
            val_loader = val_loader_s

        try:
            images_l, targets = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = next(labeled_iter)

        try:
            #             (images_uw, images_us), _ = unlabeled_iter.next()
            (images_uw, images_us), unlabeled_targets = next(unlabeled_iter)
            
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            #             (images_uw, images_us), _ = unlabeled_iter.next()
            (images_uw, images_us), unlabeled_targets = next(unlabeled_iter)

        try:
            images_val, val_targets = next(val_iter)

        except:
            if args.world_size > 1:
                val_epoch += 1
                val_loader.sampler.set_epoch(val_epoch)
            val_iter = iter(val_loader)
            images_val, val_targets = next(val_iter)

       
        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        images_val = images_val.to(args.device)
        
        targets = targets.to(args.device)
        unlabeled_targets = unlabeled_targets.to(args.device)
        val_targets = val_targets.to(args.device)

        if step % 2 == 0:
            with amp.autocast(enabled=args.amp): 
                batch_size = images_l.shape[0]
                batch_size_val = images_val.shape[0]
                
                images = torch.cat((images_l, images_uw, images_us))
                t_logits, t_features = teacher_model(images) #(batch size, num_class)
                t_logits_l = t_logits[:batch_size] 

                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)

                features_l = t_features[:batch_size] 
                features_uw, features_us = t_features[batch_size:].chunk(2) 

                features_l =torch.squeeze(features_l,3)
                features_l =torch.squeeze(features_l,2)

                features_us =torch.squeeze(features_us,3)
                features_us =torch.squeeze(features_us,2)

                dim_min = min( features_us.size(0), features_l.size(0))
                features_l = features_l[:dim_min]
                features_us = features_us[:dim_min]

                t_loss_mmd = get_MMD(features_l,features_us)


                del t_logits
                targets = torch.tensor(targets, dtype=torch.int64)
                
                t_loss_l = criterion(t_logits_l, targets)

                soft_pseudo_label_t = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
                max_probs_t, hard_pseudo_label_t = torch.max(soft_pseudo_label_t, dim=-1)

                mask_t = max_probs_t.ge(args.threshold).float()
                 # log-softmax 将更高地惩罚似然空间中更大的错误
               
                t_loss_u_no_mask = torch.mean(
                    -(soft_pseudo_label_t * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1)
                )
                t_loss_u = torch.mean(
                    -(soft_pseudo_label_t * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask_t
                )
                # Loss_u 无标签损失的系数逐步增大
                weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
               
                t_loss_uda = t_loss_l + weight_u * t_loss_u + args.lambda_mmd * t_loss_mmd

                ct_images = torch.cat((images_val, images_us))
                s_logits, _ = student_model(ct_images)
                s_logits_val = s_logits[:batch_size_val]
                s_logits_us = s_logits[batch_size_val:]
                del s_logits

              
                s_loss_l_old = F.cross_entropy(s_logits_val.detach(), val_targets)
                
                s_loss_ct = criterion(s_logits_us, hard_pseudo_label_t)

            s_scaler.scale(s_loss_ct).backward()
            if args.grad_clip > 0:
                s_scaler.unscale_(s_optimizer)
                nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
            s_scaler.step(s_optimizer)
            s_scaler.update()
            s_scheduler.step()

            if args.ema > 0:
                avg_student_model.update_parameters(student_model)
                # avg_teacher_model.update_parameters(teacher_model)
            with amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    
                    s_logits_val, _ = student_model(images_val)
                
                s_loss_l_new = F.cross_entropy(s_logits_val.detach(), val_targets)
               
                dot_product = s_loss_l_old - s_loss_l_new
                #             moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
                #             dot_product = dot_product - moving_dot_product
                _, hard_pseudo_label_t = torch.max(t_logits_us.detach(), dim=-1)
                
                
                t_loss_ad = max(0, dot_product) * F.cross_entropy(t_logits_us, hard_pseudo_label_t) - min(0, dot_product) * args.w_mmd2 * get_MMD(features_l,features_us)

                t_loss = t_loss_uda + t_loss_ad

            t_scaler.scale(t_loss).backward()
            if args.grad_clip > 0:
                t_scaler.unscale_(t_optimizer)
                nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
            t_scaler.step(t_optimizer)
            t_scaler.update()
            t_scheduler.step() 
            if args.ema > 0:
                avg_teacher_model.update_parameters(teacher_model)

            teacher_model.zero_grad()
            student_model.zero_grad()
            if args.world_size > 1:
                s_loss = reduce_tensor(s_loss.detach(), args.world_size)
                t_loss = reduce_tensor(t_loss.detach(), args.world_size)
                t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
                t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
                t_loss_ad = reduce_tensor(t_loss_ad.detach(), args.world_size)
                mask = reduce_tensor(mask_t, args.world_size)

            s_losses.update(s_loss_ct.item())
            t_losses.update(t_loss.item())
            t_losses_l.update(t_loss_l.item())
            t_losses_u.update(t_loss_u.item())
            t_losses_mpl.update(t_loss_ad.item())
            t_losses_mmd.update(t_loss_mmd.item())
            mean_mask.update(mask_t.mean().item())

            batch_time.update(time.time() - end)

            pbar.set_description(
                f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
                f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
                f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
            pbar.update()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("lr", get_lr(s_optimizer), step)

        else:
            with amp.autocast(enabled=args.amp): 
                batch_size = images_l.shape[0]
                batch_size_val = images_val.shape[0]
                
                images = torch.cat((images_l, images_uw, images_us))
                s_logits, s_features = student_model(images) #(batch size, num_class)
                s_logits_l = s_logits[:batch_size] 
                s_logits_uw, s_logits_us = s_logits[batch_size:].chunk(2)

                features_l = s_features[:batch_size] 
                features_uw, features_us = s_features[batch_size:].chunk(2)

                features_l =torch.squeeze(features_l,3)
                features_l =torch.squeeze(features_l,2)

                features_us =torch.squeeze(features_us,3)
                features_us =torch.squeeze(features_us,2)

                dim_min = min( features_us.size(0), features_l.size(0))
                features_l = features_l[:dim_min]
                features_us = features_us[:dim_min]

                s_loss_mmd = get_MMD(features_l,features_us)

                del s_logits
                targets = torch.tensor(targets, dtype=torch.int64)
                s_loss_l = criterion(s_logits_l, targets)

                soft_pseudo_label_s = torch.softmax(s_logits_uw.detach() / args.temperature, dim=-1)
                max_probs_s, hard_pseudo_label_s = torch.max(soft_pseudo_label_s, dim=-1)

                mask_s = max_probs_s.ge(args.threshold).float()
                
                s_loss_u_no_mask = torch.mean(
                    -(soft_pseudo_label_s * torch.log_softmax(s_logits_us, dim=-1)).sum(dim=-1)
                )
                s_loss_u = torch.mean(
                    -(soft_pseudo_label_s * torch.log_softmax(s_logits_us, dim=-1)).sum(dim=-1) * mask_s
                )
                
                weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
                
                s_loss_uda = s_loss_l + weight_u * s_loss_u + args.lambda_mmd * s_loss_mmd


                
                ct_images = torch.cat((images_val, images_us))
                
                t_logits, _ = teacher_model(ct_images)
                
                t_logits_val = t_logits[:batch_size_val]
                
                t_logits_us = t_logits[batch_size_val:]
                
                
                del t_logits

                
                t_loss_l_old = F.cross_entropy(t_logits_val.detach(), val_targets)
                
                t_loss_ct = criterion(t_logits_us, hard_pseudo_label_s)

            t_scaler.scale(t_loss_ct).backward()
            if args.grad_clip > 0:
                t_scaler.unscale_(t_optimizer)
                nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
            t_scaler.step(t_optimizer)
            t_scaler.update()
            t_scheduler.step()

            
            with amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    
                    t_logits_val, _ = teacher_model(images_val)
                
                t_loss_l_new = F.cross_entropy(t_logits_val.detach(), val_targets)
            
                dot_product = t_loss_l_old - t_loss_l_new
                #             moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
                #             dot_product = dot_product - moving_dot_product
                _, hard_pseudo_label_s = torch.max(s_logits_us.detach(), dim=-1)
                
                s_loss_ad = max(0, dot_product) * F.cross_entropy(s_logits_us, hard_pseudo_label_s)- min(0, dot_product) * args.w_mmd2 * get_MMD(features_l,features_us)

                s_loss = s_loss_uda + s_loss_ad

            s_scaler.scale(s_loss).backward()
            if args.grad_clip > 0:
                s_scaler.unscale_(s_optimizer)
                nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
            s_scaler.step(s_optimizer)
            s_scaler.update()
            s_scheduler.step() 


            teacher_model.zero_grad()
            student_model.zero_grad()

            if args.world_size > 1:
                s_loss = reduce_tensor(s_loss.detach(), args.world_size)
                t_loss = reduce_tensor(t_loss_ct.detach(), args.world_size)
                s_loss_l = reduce_tensor(s_loss_l.detach(), args.world_size)
                s_loss_u = reduce_tensor(s_loss_u.detach(), args.world_size)
                s_loss_ad = reduce_tensor(s_loss_ad.detach(), args.world_size)
                mask = reduce_tensor(mask_s, args.world_size)

            s_losses.update(s_loss.item())
            t_losses.update(t_loss.item())
            s_losses_l.update(s_loss_l.item())
            s_losses_u.update(s_loss_u.item())
            s_losses_mpl.update(s_loss_ad.item())
            s_losses_mmd.update(s_loss_mmd.item())

            mean_mask.update(mask_s.mean().item())

            batch_time.update(time.time() - end)

            pbar.set_description(
                f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
                f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
                f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
            pbar.update()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("lr", get_lr(s_optimizer), step)

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                if step % 2 == 0:
                    args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                    args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                    args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                    args.writer.add_scalar("train/6.t_mmd", t_losses_mmd.avg, args.num_eval)
                else:
                    args.writer.add_scalar("train/3.s_labeled", s_losses_l.avg, args.num_eval)
                    args.writer.add_scalar("train/4.s_unlabeled", s_losses_u.avg, args.num_eval)
                    args.writer.add_scalar("train/5.s_mpl", s_losses_mpl.avg, args.num_eval)
                    args.writer.add_scalar("train/6.s_mmd", s_losses_mmd.avg, args.num_eval)

                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss_ema, top1_ema, top2_ema = evaluate(args, test_loader, test_model, criterion)

                test_loss, top1, top2 = evaluate(args, test_loader, student_model, criterion)
                
                teacher_test_model = avg_teacher_model if avg_teacher_model is not None else teacher_model
                teacher_loss_ema, t_top1_ema, t_top2_ema = evaluate(args, test_loader, teacher_test_model, criterion)

                teacher_loss, t_top1, t_top2 = evaluate(args, test_loader, teacher_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/ema_acc@1", top1_ema, args.num_eval)
                
                args.writer.add_scalar("test/acc@2", top2, args.num_eval)

                args.writer.add_scalar("teacher/acc@1", t_top1, args.num_eval)
                args.writer.add_scalar("teacher/ema_acc@1", t_top1_ema, args.num_eval)

                t_is_best = t_top1 > args.t_best_top1
                if t_is_best:
                    args.t_best_top1 = t_top1
                    args_t_best_top2 = t_top2

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top2 = top2

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")
                logger.info(f"teacher top-1 acc: {t_top1:.2f}")
                logger.info(f"Best teacher top-1 acc: {args.t_best_top1:.2f}")

                json_path = os.path.join(args.log, args.name+'.json')
                json_file = open(json_path, 'r')
                
                json_dict = json.load(json_file)
                json_file.close()

                # json_dict['top acc'] = max(args.best_top1, args.t_best_top1)
                
                top_acc = max(args.best_top1, args.t_best_top1).numpy()
                top_acc = str(top_acc)

                print("top acc", top_acc)
                json_dict['top acc'] = top_acc
                with open(json_path,'w') as write_f:
                    json.dump(json_dict,write_f,indent = 4, ensure_ascii=False)


                save_checkpoint2(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top2': args.best_top2,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)
                save_checkpoint2(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.t_best_top1,
                    'best_top2': args.best_top2,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, t_is_best, is_teacher=True)


                if step + 1 == 10000:
                    state = {
                        'step': step + 1,
                        'teacher_state_dict': teacher_model.state_dict(),
                        'student_state_dict': student_model.state_dict(),
                        'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                        'best_top1': args.best_top1,
                        'best_top2': args.best_top2,
                        'teacher_optimizer': t_optimizer.state_dict(),
                        'student_optimizer': s_optimizer.state_dict(),
                        'teacher_scheduler': t_scheduler.state_dict(),
                        'student_scheduler': s_scheduler.state_dict(),
                        'teacher_scaler': t_scaler.state_dict(),
                        'student_scaler': s_scaler.state_dict(),
                    }
                    # os.makedirs(args.save_path, exist_ok=True)
                    name = args.name
                    filename = f'{args.save_path}/{name}_{str(step+1)}_{str(int(top1))}.pth.tar'
                    torch.save(state, filename, _use_new_zipfile_serialization=False)
    
    # finetune
    del t_scaler, t_scheduler, t_optimizer, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(args, all_labeled_loader, test_loader, teacher_model, criterion)
    finetune(args, all_labeled_loader, test_loader, student_model, criterion)
    return


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs, _ = model(images)
                loss = criterion(outputs, targets)

            acc1, acc2 = accuracy(outputs, targets, (1, 2))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top2.update(acc2[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step + 1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top2: {top2.avg:.2f}. ")

        test_iter.close()
        return losses.avg, top1.avg, top2.avg


def finetune(args, train_loader, test_loader, model, criterion, is_teacher = False):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        #         num_workers=0,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader) * args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch + 1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            if is_teacher:
                args.writer.add_scalar("finetune_teacher/train_loss", losses.avg, epoch)
                test_loss, top1, top2 = evaluate(args, test_loader, model, criterion)
                args.writer.add_scalar("finetune_teacher/test_loss", test_loss, epoch)
                args.writer.add_scalar("finetune_teacher/acc@1", top1, epoch)
                args.writer.add_scalar("finetune_teacher/acc@2", top2, epoch)
            else:
                args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
                test_loss, top1, top2 = evaluate(args, test_loader, model, criterion)
                args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
                args.writer.add_scalar("finetune/acc@1", top1, epoch)
                args.writer.add_scalar("finetune/acc@2", top2, epoch)
            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top2 = top2

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint2(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top2': args.best_top2,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True,is_teacher=is_teacher)
    return


def create_model(num_classes, args):
    model = torchvision.models.resnet50(pretrained=True)

    loc = f'cuda:{args.gpu}'
    device = torch.device(loc)
    model.fc = nn.Linear(2048, num_classes)
    # model = nn.DataParallel(model)
    model= model.to(device)
    # model = model.cuda()
    return model

def create_model_res18(num_classes, args):

    loc = f'cuda:{args.gpu}'
    device = torch.device(loc)


    # model = torchvision.models.resnet18(pretrained=True)
    model = resnet18()
    
    checkpoint = torch.load(args.pretrain_model)
    # checkpoint = torch.load('/home/mnt/lyq/code/Face2Exp/pretrain_model/resnet18_msceleb.pth',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.fc = nn.Linear(512, num_classes)
    model= model.to(device)
    return model


def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top2 = 0.
    args.t_best_top1 = 0.
    args.t_best_top2 = 0.
    #args.local_rank ！= -1 （最好写args.local_rank = 0，而不是随便用一个值，否则会有warning）表示使用分布式训练
    if args.local_rank != -1:
        args.gpu = args.local_rank
        #         torch.distributed.init_process_group(backend='nccl')
        #         torch.distributed.init_process_group(backend='gloo')
        args.world_size = torch.distributed.get_world_size()
    else: 
        #args.local_rank = -1 表示使用不分布式训练，只有一个GPU
        args.gpu = 0 #TODO
        args.world_size = 1 #表示分布式训练的结点个数，有多个nodes

    args.device = torch.device('cuda', args.gpu)
    #     args.device = rch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)
    

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() #分布式训练，其他进程与主进程之间的数据同步


    labeled_dataset_t,labeled_dataset_s, unlabeled_dataset, test_dataset, val_dataset_t,val_dataset_s, all_labeled_dataset = DATASET_GETTERS['get_data'](args)
    
    if args.local_rank == 0:
        torch.distributed.barrier() #分布式训练，其他进程与主进程之间的数据同步

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader_t = DataLoader(
        labeled_dataset_t,
        sampler=train_sampler(labeled_dataset_t),
        batch_size=args.batch_size,
        num_workers=args.workers,
        #         num_workers=0,
        #         drop_last=True,
        pin_memory=True)
    labeled_loader_s = DataLoader(
        labeled_dataset_s,
        sampler=train_sampler(labeled_dataset_s),
        batch_size=args.batch_size,
        num_workers=args.workers,
        #         num_workers=0,
        #         drop_last=True,
        pin_memory=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        #         num_workers=0,
        #         drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             #                              num_workers=0
                             num_workers=args.workers,
                             pin_memory=True
                             )

    val_loader_t = DataLoader(val_dataset_t,
                             sampler=SequentialSampler(val_dataset_t),
                             batch_size=args.batch_size,
                             #                              num_workers=0
                             num_workers=args.workers,
                             pin_memory=True
                             )

    val_loader_s = DataLoader(val_dataset_s,
                            sampler=SequentialSampler(val_dataset_s),
                            batch_size=args.batch_size,
                            #                              num_workers=0
                            num_workers=args.workers,
                            pin_memory=True
                            )

    all_labeled_loader = DataLoader(
        all_labeled_dataset,
        sampler=train_sampler(all_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        #         num_workers=0,
        #         drop_last=True,
        pin_memory=True)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # if torch.cuda.device_count() > 1:
    #     print("We have ", torch.cuda.device_count(), " GPUs!")

    teacher_model = create_model_res18(num_classes=args.num_classes,args=args)
    student_model = create_model_res18(num_classes=args.num_classes, args=args)


    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info(f"Model: ResNet50")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)

    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(student_model, args.ema)
        avg_teacher_model = ModelEMA(teacher_model, args.ema)

    criterion = create_loss_fn(args)

    t_optimizer = optim.SGD(teacher_model.parameters(),
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_model.parameters(),
                            lr=args.student_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)

    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    config_save(args)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top2 = checkpoint['best_top2'].to(torch.device('cpu'))

            # args.t_best_top1 = checkpoint['t_best_top1'].to(torch.device('cpu'))
            # args.t_best_top2 = checkpoint['t_best_top2'].to(torch.device('cpu'))
            args.t_best_top1 = 0
            args.t_best_top2 = 0



            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                model_load_state_dict(student_model, checkpoint['student_state_dict'])
                '''
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])
                '''
                

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, all_labeled_loader, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()

    
    train_loop(args, labeled_loader_t,labeled_loader_s, unlabeled_loader, test_loader, val_loader_t, val_loader_s, all_labeled_loader,
                           teacher_model, student_model, avg_teacher_model,avg_student_model, criterion,
                           t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)

    return


if __name__ == '__main__':
    main()
