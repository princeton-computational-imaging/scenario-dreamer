import torch
import torch.nn.functional as F 
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
torch.set_printoptions(sci_mode=False)

from nn_modules.ctrl_sim import CtRLSimEncoder, CtRLSimDecoder
from utils.train_helpers import create_lambda_lr_linear

class CtRLSim(pl.LightningModule):
    """ PyTorch Lightning module for CtRL-Sim model."""
    def __init__(self, cfg):
        super(CtRLSim, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg 
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        self.action_dim = self.cfg_dataset.vocab_size
        self.seq_len = self.cfg_dataset.train_context_length

        self.encoder = CtRLSimEncoder(self.cfg)
        self.decoder = CtRLSimDecoder(self.cfg)


    def forward(self, data, eval=False):
        """ Forward pass through the CtRL-Sim model."""
        scene_enc = self.encoder(data, eval)
        pred = self.decoder(data, scene_enc, eval)

        return pred 


    def compute_loss(self, data, preds):
        """ Compute the loss for CtRL-Sim model."""
        loss_dict = {}
        if self.cfg_model.trajeglish:
            logits = preds['action_preds']
            logits = logits[:, :, :-1, :]
            logits = logits.reshape(
                -1, 
                self.cfg_dataset.max_num_agents*(self.seq_len-1), 
                self.action_dim
            )
            B, T, C = logits.shape 
            existence_mask = torch.logical_and(
                data['agent'].agent_states[:, :, :-1, -1:].reshape(B, T, 1),
                data['agent'].agent_states[:, :, 1:, -1:].reshape(B, T, 1)
            )

            logits = logits.view(B * T, C)
            actions = data['agent'].actions[:, :, 1:].reshape(-1)
            existence_mask = existence_mask.view(-1)
            loss_actions = F.cross_entropy(logits.float(), actions.long(), reduction='none')
            loss_actions = loss_actions * existence_mask.float()
            loss_actions = (self.cfg_model.loss_action_coef * loss_actions.sum()) / existence_mask.sum()

        else:
            logits = preds['action_preds']
            logits = logits.reshape(
                -1, 
                self.cfg_dataset.max_num_agents*self.seq_len, 
                self.action_dim
            )
            B, T, C = logits.shape 
            existence_mask = data['agent'].agent_states[:, :, :, -1:].reshape(B, T, 1)
            rtg_mask = data['agent'].rtg_mask.reshape(B, T, 1)
            
            logits = logits.view(B * T, C)
            actions = data['agent'].actions.view(-1)
            existence_mask = existence_mask.view(-1)
            rtg_mask = rtg_mask.view(-1)
            loss_actions = F.cross_entropy(logits.float(), actions.long(), reduction='none')
            loss_actions = loss_actions * existence_mask.float()
            loss_actions = (self.cfg_model.loss_action_coef * loss_actions.sum()) / existence_mask.sum()

        loss_dict['loss_actions'] = loss_actions
        
        if self.cfg_model.predict_rtg:
            rtg_preds = preds['rtg_preds'].reshape(
                -1, 
                self.cfg_dataset.rtg_discretization, 
                self.cfg_model.num_reward_components
            )
            rtg_veh_logits = rtg_preds[:, :, 0]
            rtg_veh = data['agent'].rtgs[:, :, :, 0].reshape(-1)

            loss_rtg_veh = F.cross_entropy(rtg_veh_logits.float(), rtg_veh.long(), reduction='none')
            loss_rtg_veh = loss_rtg_veh * rtg_mask.float()
            loss_rtg_veh = loss_rtg_veh.sum() / rtg_mask.sum()

            loss_dict['loss_rtg_veh'] = loss_rtg_veh

        return loss_dict


    def training_step(self, data, batch_idx):
        """ Training step for CtRL-Sim model."""
        preds = self(data)
        loss_dict = self.compute_loss(data, preds)

        self.log(
            'loss', 
            loss_dict['loss_actions'], 
            prog_bar=True, 
            on_step=True, 
            on_epoch=False, 
            sync_dist=True
        )
        if self.cfg_model.predict_rtg:
            self.log(
                'loss_rtg_veh', 
                loss_dict['loss_rtg_veh'], 
                prog_bar=True, 
                on_step=True, 
                on_epoch=False, 
                sync_dist=True
            )
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(
            'lr', 
            cur_lr, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=False, 
            sync_dist=True
        )

        final_loss = loss_dict['loss_actions']
        if self.cfg_model.predict_rtg:
            final_loss = final_loss + loss_dict['loss_rtg_veh']
        
        return final_loss


    def validation_step(self, data, batch_idx):
        """ Validation step for CtRL-Sim model."""
        preds = self(data, eval=True)
        loss_dict = self.compute_loss(data, preds)
        B = preds['action_preds'].shape[0]

        self.log(
            'val_loss', 
            loss_dict['loss_actions'], 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=B
        )
        if self.cfg_model.predict_rtg:
            self.log(
                'val_rtg_veh_loss', 
                loss_dict['loss_rtg_veh'], 
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True, 
                batch_size=B
            ) 


    def on_before_optimizer_step(self, optimizer):
        """ Log gradient norms before optimizer step."""
        norms_encoder = grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms_encoder)

        norms_decoder = grad_norm(self.decoder, norm_type=2)
        self.log_dict(norms_decoder)


    ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        """ Configure optimizers and learning rate schedulers."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lambda_lr_linear(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}



