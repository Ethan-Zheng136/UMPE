from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
import torch
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import math

@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module() 
class AlphaScheduleHook(Hook):
    def __init__(self, 
                 start_iter=0,
                 end_iter=20000,
                 start_alpha=0.2, 
                 target_alpha=0.6,   
                 schedule_type='cosine',
                 update_mode='smooth'): 
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_alpha = start_alpha
        self.target_alpha = target_alpha
        self.schedule_type = schedule_type
        self.update_mode = update_mode
        self.momentum = 0.1 if update_mode == 'smooth' else 1.0 
        
    def get_current_alpha_target(self, current_iter):
        if current_iter < self.start_iter:
            return self.start_alpha
        elif current_iter >= self.end_iter:
            return self.target_alpha
        else:
            progress = (current_iter - self.start_iter) / (self.end_iter - self.start_iter)
            
            if self.schedule_type == 'cosine':
                progress = 0.5 * (1 - math.cos(math.pi * progress))

            return self.start_alpha + progress * (self.target_alpha - self.start_alpha)
    
    def after_train_iter(self, runner):
        current_iter = runner.iter
        target_alpha = self.get_current_alpha_target(current_iter)
        
        alpha_params_found = 0
        total_alpha_value = 0
        alpha_updated_count = 0
        
        for name, param in runner.model.named_parameters():
            if 'fusion_module.safe_fuse.alpha' in name:
                alpha_params_found += 1
                with torch.no_grad():
                    old_val = param.data.item()
                    
                    if self.update_mode == 'direct':

                        param.data.fill_(target_alpha)
                        new_val = target_alpha
                        alpha_updated_count += 1
                    elif self.update_mode == 'smooth':

                        new_val = (1 - self.momentum) * old_val + self.momentum * target_alpha
                        param.data.fill_(new_val)
                        if abs(new_val - old_val) > 1e-6:
                            alpha_updated_count += 1
                    
                    total_alpha_value += param.data.item()

                    if runner.iter % 1000 == 0:
                        runner.logger.info(f"FUSION_MODULE: {name}: {param.data.item():.6f} (target: {target_alpha:.6f})")
        
        if runner.iter % 100 == 0 and alpha_params_found > 0:
            avg_alpha = total_alpha_value / alpha_params_found
            progress = min((current_iter - self.start_iter) / (self.end_iter - self.start_iter), 1.0)
            progress = max(progress, 0.0)
            
            runner.log_buffer.output['fusion_alpha/target_alpha'] = target_alpha
            runner.log_buffer.output['fusion_alpha/current_alpha'] = avg_alpha
            runner.log_buffer.output['fusion_alpha/progress'] = progress
            runner.log_buffer.output['fusion_alpha/alpha_updated'] = alpha_updated_count > 0
            
            if target_alpha > 0:
                achievement_ratio = avg_alpha / target_alpha
                runner.log_buffer.output['fusion_alpha/achievement_ratio'] = achievement_ratio
            
            runner.logger.info(f"FUSION Alpha Schedule - Iter {current_iter}: "
                             f"Target: {target_alpha:.4f}, Current: {avg_alpha:.4f}, "
                             f"Progress: {progress:.1%}, Mode: {self.update_mode}")
        
        if runner.iter % 5000 == 0 and alpha_params_found == 0:
            runner.logger.warning("No fusion_module.alpha parameter found!")
            for name, param in runner.model.named_parameters():
                if 'alpha' in name:
                    runner.logger.info(f"   Found alpha param: {name}")


@HOOKS.register_module()
class FusionAlphaMonitorHook(Hook):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval

    def after_train_iter(self, runner):
        if runner.iter % self.log_interval != 0:
            return

        fusion_alpha = None
        for name, param in runner.model.named_parameters():
            if 'fusion_module.alpha' in name:
                alpha_val = param.data.item()
                fusion_alpha = alpha_val

                runner.log_buffer.output['fusion_alpha/value'] = alpha_val

                if alpha_val < -0.01:
                    runner.logger.warning(f'Negative fusion alpha: {alpha_val:.6f}')
                if alpha_val > 0.25:
                    runner.logger.warning(f'Large fusion alpha: {alpha_val:.6f}')
                break

        if fusion_alpha is not None:
            max_alpha_allowed = runner.log_buffer.output.get('fusion_alpha/max_alpha_allowed', 0.0)
            runner.log_buffer.output['fusion_alpha/utilization'] = (
                fusion_alpha / max(max_alpha_allowed, 1e-6)
            )
