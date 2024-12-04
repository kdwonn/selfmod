import einops
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from functools import partial, reduce
from operator import mul
from einops import rearrange, repeat
import math
import re
'''
Borrowed from https://github.com/sagizty/VPT
'''

class PromptPool(nn.Module):
    def __init__(self, num_entries, depth, num_per_slot, embed_dim, patch_size):
        super().__init__()
        self.num_entries = num_entries
        self.depth = depth
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_per_slot = num_per_slot
        self.prompts = nn.Parameter(torch.zeros(num_entries, depth, num_per_slot, embed_dim))

        self._init_prompt()

    def _init_prompt(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.prompts.data, -val, val)

    def forward(self, indices):
        assert len(indices.shape) == 2
        b, k = indices.shape
        indices = rearrange(indices, 'b k -> (b k)')
        prompt = rearrange(self.prompts[indices], '(b k) ... -> b k ...', b=b)
        ret = rearrange(prompt, 'b k depth num dim -> depth b (k num) dim')
        return ret
    

class Prompt(nn.Module):
    def __init__(self, depth, num_prompt, embed_dim, patch_size):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_prompt = num_prompt
        self.prompts = nn.Parameter(torch.zeros(depth, num_prompt, embed_dim))

        self._init_prompt()

    def _init_prompt(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.prompts.data, -val, val)

    def forward(self):
        return self.prompts
    

class CodeToPrompt(nn.Module):
    def __init__(
        self, 
        code_dim, 
        num_per_slot, 
        embed_dim, 
        use_ln=True, 
        use_mlp=False,
        use_sa=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_per_slot = num_per_slot

        self.use_sa = use_sa
        if use_sa:
            self.sa = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=code_dim,
                    nhead=4,
                    dim_feedforward=code_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                nn.LayerNorm(code_dim),
            )

        if use_mlp:
            self.c2p = nn.Sequential(
                nn.Linear(code_dim, code_dim),
                nn.ReLU(),
                nn.Linear(code_dim, num_per_slot * embed_dim)
            )
        else:
            self.c2p = nn.Linear(code_dim, num_per_slot * embed_dim)
        self.ln = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()

        self.ema_decay = 0.9  # EMA decay rate
        self.cold = True
        self.register_buffer('running_prompt', torch.zeros(self.embed_dim))  # EMA prompts buffer

    def ema_update(self, prompts):
        with torch.no_grad():
            to_update = einops.reduce(prompts, 'b k dim -> dim', 'mean')
            self.running_prompt = self.ema_decay * self.running_prompt + (1 - self.ema_decay) * to_update

    def forward(self, code):
        assert len(code.shape) == 3
        b, k, in_dim = code.shape

        if self.use_sa:
            code = self.sa(code)
        
        prompts = rearrange(
            self.c2p(code),
            'b k (num dim) -> b (k num) dim',
            num=self.num_per_slot,
        )
        prompts = self.ln(prompts)

        if self.training:
            self.ema_update(prompts)

        return prompts


class VPT_ViT(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_deep=True,
        use_inter_share=False,
        basic_state_dict=None,
        num_entries=256,
        num_per_slot=3,
    ):
        # Recreate ViT
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.use_deep = use_deep
        self.use_inter_share = use_inter_share
        if use_deep:
            if use_inter_share:
                self.prompt_pool = PromptPool(num_entries, 1, num_per_slot, embed_dim, self.patch_embed.patch_size)
            else:
                self.prompt_pool = PromptPool(num_entries, depth, num_per_slot, embed_dim, self.patch_embed.patch_size)
        else:  # "Shallow"
            self.prompt_pool = PromptPool(num_entries, 1, num_per_slot, embed_dim, self.patch_embed.patch_size)

    def freeze_except_prompt(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.prompt_pool.parameters():
            param.requires_grad = True

    def forward_features(self, x, code, indices):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if indices is not None:
            prompt = self.prompt_pool(indices)
            prompt_token_num = prompt.shape[2]

            if self.use_deep:
                for i in range(len(self.blocks)):
                    if self.use_inter_share:
                        prompt_tokens = prompt.squeeze(0)
                    else:
                        prompt_tokens = prompt[i]
                    # firstly concatenate
                    x = torch.cat((x, prompt_tokens), dim=1)
                    num_tokens = x.shape[1]
                    # lastly remove, a genius trick
                    x = self.blocks[i](x)[:, :num_tokens - prompt_token_num]
            else:  # self.vpt_type == "Shallow"
                # concatenate prompt_tokens
                x = torch.cat((x, prompt.squeeze(0)), dim=1)
                num_tokens = x.shape[1]
                # Sequntially procees
                x = self.blocks(x)[:, :num_tokens - prompt_token_num]
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        raise NotImplementedError
    

class VPT_ViT_wo_fb(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_deep=True,
        basic_state_dict=None,
        num_prompt=20,
    ):
        # Recreate ViT
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.use_deep = use_deep
        if use_deep:
            self.prompt = Prompt(depth, num_prompt, embed_dim, self.patch_embed.patch_size)
        else:  # "Shallow"
            self.prompt = Prompt(1, num_prompt, embed_dim, self.patch_embed.patch_size)

    def freeze_except_prompt(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.prompt.parameters():
            param.requires_grad = True

    def forward_features(self, x, code, indices):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        prompt = self.prompt()
        prompt_token_num = prompt.shape[1]

        if self.use_deep:
            for i in range(len(self.blocks)):
                # concatenate prompt_tokens
                prompt_tokens = prompt[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                # firstly concatenate
                x = torch.cat((x, prompt_tokens), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - prompt_token_num]
        else:  # self.vpt_type == "Shallow"
            # concatenate prompt_tokens
            x = torch.cat((x, prompt.expand(x.shape[0], -1, -1)), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees
            x = self.blocks(x)[:, :num_tokens - prompt_token_num]

        x = self.norm(x)
        return x

    def forward(self, x):
        raise NotImplementedError
    

class VPT_ViT_code(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_deep=True,
        use_inter_share=False,
        basic_state_dict=None,
        num_per_slot=3,
        code_dim=256,
        use_mlp=False,
        use_ln=True,
        use_running_prompt=False,
        num_slot=7,
        prompt_layers=None,
        use_sa=False,
    ):
        # Recreate ViT
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.use_deep = use_deep
        self.use_inter_share = use_inter_share
        
        self.use_running_prompt = use_running_prompt
        self.code_dim = code_dim
        self.num_per_slot = num_per_slot
        self.num_slot = num_slot
        
        if use_deep:
            if use_inter_share:
                self.prompt_pool = CodeToPrompt(code_dim, num_per_slot, embed_dim, use_ln=use_ln, use_mlp=use_mlp, use_sa=use_sa)
            else:
                raise NotImplementedError
        else:  # "Shallow"
            self.prompt_pool = CodeToPrompt(code_dim, num_per_slot, embed_dim, use_ln=use_ln, use_mlp=use_mlp, use_sa=use_sa)

        if prompt_layers is not None:
            match = re.match(r'^(\d+)-(\d+)$', prompt_layers)
            assert match, "Invalid format: 'a-b'"
            self.prompting_start_layer = int(match.group(1))
            self.prompting_end_layer = int(match.group(2))
        else:
            self.prompting_start_layer = 0
            self.prompting_end_layer = depth -1

    def freeze_except_prompt(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.prompt_pool.parameters():
            param.requires_grad = True

    def forward_features(self, x, code, indices):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if code is not None or self.use_running_prompt:
            if code is None:
                prompt = self.prompt_pool.running_prompt
                prompt = repeat(
                    prompt, 
                    'dim -> b k dim', 
                    b=x.shape[0], k=self.num_per_slot*self.num_slot
                )
                prompt_token_num = prompt.shape[1]
            else:
                prompt = self.prompt_pool(code)
                prompt_token_num = prompt.shape[1]

            if self.use_deep:
                for i in range(len(self.blocks)):
                    if self.prompting_start_layer <= i <= self.prompting_end_layer:
                        # firstly concatenate
                        x = torch.cat((x, prompt), dim=1)
                        num_tokens = x.shape[1]
                        # lastly remove, a genius trick
                        x = self.blocks[i](x)[:, :num_tokens - prompt_token_num]
                    else:
                        x = self.blocks[i](x)
            else:  # self.vpt_type == "Shallow"
                for i in range(self.depth):
                    if i == self.prompting_start_layer:
                        x = torch.cat((x, prompt), dim=1)
                        
                    x = self.blocks[i](x)
                    
                    if i == self.prompting_end_layer:
                        num_tokens = x.shape[1]
                        x = x[:, :num_tokens - prompt_token_num]
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        raise NotImplementedError