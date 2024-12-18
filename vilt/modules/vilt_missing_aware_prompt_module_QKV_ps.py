import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils, layers
import torch.nn.functional as F
import math                 
                    
 

class ViLTransformerSS_QKV_ps(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
# 
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")
  
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1
        from timm.models.layers import trunc_normal_

        # self.cal = ContextAwarePromptLayer(config["hidden_size"], prompt_length*3, prompt_num)
        prompt_idx_ten_0 = torch.tensor([i for i in range(prompt_length)])
        prompt_idx_ten_1 = torch.tensor([i for i in range(prompt_length)])+prompt_length
        prompt_idx_ten_2 = torch.tensor([i for i in range(prompt_length)])+prompt_length*2
        self.prompt_idx_dict = {
            0: prompt_idx_ten_0.long(),
            1: prompt_idx_ten_1.long(),
            2: prompt_idx_ten_2.long(),
        }
        self.pool_size = self.hparams.config["pool_size"]
        self.fixed_prompts = config["fixed_prompts"]
        self.fixed_prompt_layers = config["fixed_prompt_layers"]
        self.add_agnostic_prompts_layers = nn.ModuleList([layers.add_agnostic_prompts(config["hidden_size"], prompt_length+self.fixed_prompts) for _ in self.fixed_prompt_layers])
        self.QKV_ps_layers = nn.ModuleList([layers.QKV_ps(config["hidden_size"], prompt_length, self.pool_size, config["fixed_prompts"], return_att=True) for i in self.prompt_layers if i!=0])
        for i in self.prompt_layers:
            if i!= 0:   
                layer = self.QKV_ps_layers[i-self.prompt_layers[0]]  
                layer.W_Q = nn.Parameter(self.transformer.blocks[i].attn.qkv.weight.transpose(-1,-2)[:,:768].clone())
                layer.bias_Q = nn.Parameter(self.transformer.blocks[i].attn.qkv.bias[:768].clone())
                layer.layerNorm.weight = nn.Parameter(self.transformer.blocks[i].norm1.weight.clone())      
                layer.layerNorm.bias = nn.Parameter(self.transformer.blocks[i].norm1.bias.clone())      
                layer.W_Q.requires_grad = False   
                layer.bias_Q.requires_grad = False
                layer.layerNorm.weight.requires_grad = False   
                layer.layerNorm.bias.requires_grad = False     


        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}



    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        
        if image_embeds is None and image_masks is None:
                   
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
            


        else:
            patch_index, image_labels = (
                None,
                None,
            )
            
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        

        cuda_device = torch.cuda.current_device()
        missing_type = torch.tensor(batch["missing_type"])
        missing_type = missing_type.to(cuda_device)
        
        
        prompt_length = self.prompt_length + self.fixed_prompts
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(missing_type.shape[0], prompt_length//2, dtype=missing_type.dtype, device=missing_type.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(missing_type.shape[0], prompt_length*len(self.prompt_layers+self.fixed_prompt_layers), dtype=missing_type.dtype, device=missing_type.device).long()
        else:
            prompt_masks = torch.ones(missing_type.shape[0], prompt_length, dtype=missing_type.dtype, device=missing_type.device).long() 
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds.detach()
        
        prompts = None
        A_list = []
        CLS_A_list = []
        CLS_Agn_list = []
        prompt_adding = 0
        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers or i in self.fixed_prompt_layers:
                if self.multi_layer_prompt:                    
                    if i in self.fixed_prompt_layers:
                        selected_prompts = self.add_agnostic_prompts_layers[i](x)
                        prompt_adding = prompt_adding + 1
                    else:
                        CLS_idx = (prompt_adding)*(self.prompt_length+self.fixed_prompts)
                        CLS = x[:,CLS_idx:CLS_idx+1] 
                        selected_prompts, A = self.QKV_ps_layers[i-self.prompt_layers[0]](CLS, missing_type)
                        A_list.append(A.detach().cpu())
                        prompt_adding = prompt_adding + 1
                    
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=selected_prompts, 
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                    CLS_idx = (prompt_adding)*(self.prompt_length+self.fixed_prompts)
                    CLS_A_list.append(_attn[:, :, CLS_idx, :CLS_idx].detach().cpu())
                    

                    if i in self.fixed_prompt_layers:
                        CLS_Agn_list.append(_attn[:, :, CLS_idx, :CLS_idx].detach().cpu())
                    elif i in self.prompt_layers:
                        CLS_Agn_collect = []
                        last_fixed = (i-self.prompt_layers[0]+1)*(self.prompt_length+self.fixed_prompts)
                        agn_att = _attn[:, :, CLS_idx, last_fixed:CLS_idx].detach().cpu()
                        CLS_Agn_collect.append(agn_att)   
                        for j in range(self.prompt_layers[0], prompt_adding):                                                       
                            agn_att = _attn[:, :, CLS_idx, (j-self.prompt_layers[0])*(self.prompt_length+self.fixed_prompts):(j-self.prompt_layers[0])*(self.prompt_length+self.fixed_prompts)+self.fixed_prompts].detach().cpu()
                            CLS_Agn_collect.append(agn_att)   
                        CLS_Agn = torch.cat(CLS_Agn_collect, dim=2)   
                        CLS_Agn_list.append(CLS_Agn)                             
                    # selected_elements = torch.cat((tensor[10:15], tensor[20:25]))
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers+self.fixed_prompt_layers)* (self.prompt_length+self.fixed_prompts) #prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)

        if len(A_list)>0:
            A_list = torch.stack(A_list, dim=1) 
        # CLS_A_list = torch.stack(CLS_A_list, dim=1)   
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "A_list": A_list,
            "CLS_A_list": CLS_A_list,
            "CLS_Agn_list": CLS_Agn_list,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
