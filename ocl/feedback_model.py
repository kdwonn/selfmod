import dataclasses
from functools import reduce
import math
from operator import mul
from typing import Optional
from einops import repeat
import torch
import torch.nn as nn
from ema_pytorch import EMA
from ocl.feature_extractors import TimmFeatureExtractorPrompt
from ocl.conditioning import RandomConditioning
from ocl.decoding import AutoregressivePatchDecoder, PatchReconstructionOutput
from ocl.typing import FeatureExtractorOutput, PerceptualGroupingOutput


@dataclasses.dataclass
class ModelOutput:
    feature_extractor: FeatureExtractorOutput
    post_feature_extractor: FeatureExtractorOutput
    conditioning: torch.Tensor
    perceptual_grouping: PerceptualGroupingOutput
    post_perceptual_grouping: PerceptualGroupingOutput
    object_decoder: PatchReconstructionOutput
    pool_indices: torch.Tensor
    commit_loss: torch.Tensor
    feedback_prob: float
    code_to_feat_attn: Optional[torch.Tensor] = None


class FeedbackVPT(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: RandomConditioning,
        perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: AutoregressivePatchDecoder,
        feedback_prob: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

    def forward(self, image, batch_size):
        feature_extractor = self.feature_extractor(image)

        conditioning = self.conditioning(batch_size)
        perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor, 
            conditioning=conditioning
        )
        _, indices, commit_loss = self.pool(perceptual_grouping.objects)

        if self.training:
            do_feedback = torch.rand(1) < self.feedback_prob
        else:
            do_feedback = self.feedback_prob > 0

        if do_feedback:
            post_feature_extractor = self.feature_extractor(image, indices=indices)
            post_perceptual_grouping = self.perceptual_grouping(
                feature=post_feature_extractor, 
                conditioning=conditioning
            )
        else:
            post_feature_extractor = feature_extractor
            post_perceptual_grouping = perceptual_grouping

        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
            target=post_feature_extractor.features,
            image=image
        )

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=post_feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob
        )


class FeedbackCodeVPT(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: AutoregressivePatchDecoder,
        feedback_prob: float = 0.5,
        use_initial_code: bool = False,
        use_prompt_as_slot: bool = False,
        use_pre_target: bool = False,
        pre_masking_prob: float = 0.,
        use_masking_token: bool = False,
        pre_slot_num: Optional[int] = None,
        td_synth: Optional[nn.Module] = None,
        code_to_feat_attn: Optional[nn.Module] = None,
        use_attn_mod: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

        self.use_initial_code = use_initial_code
        self.use_running_prompt = self.feature_extractor.model.use_running_prompt

        self.use_prompt_as_slot = use_prompt_as_slot
        self.use_pre_target = use_pre_target

        if self.use_initial_code and (not self.use_running_prompt):
            code_dim = self.conditioning.object_dim
            num_code = self.conditioning.n_slots
            self.initial_code = nn.Parameter(torch.randn(1, num_code, code_dim))
            self._init_prompt()

        self.p2s = None
        if self.use_prompt_as_slot:
            embed_dim = self.feature_extractor.model.embed_dim
            self.p2s = nn.Linear(embed_dim, self.conditioning.object_dim)

        self.pre_masking_prob = pre_masking_prob
        if pre_masking_prob > 0:
            if use_masking_token:
                self.mask_token = nn.Parameter(torch.zeros(self.feature_extractor.model.embed_dim))
            else:
                self.register_buffer("mask_token", torch.zeros(self.feature_extractor.model.embed_dim))

        self.pre_slot_num = pre_slot_num

        if td_synth is not None:
            self.td_synth = td_synth

        if code_to_feat_attn is not None:
            self.code_to_feat_attn = code_to_feat_attn

        self.use_attn_mod = use_attn_mod
        if use_attn_mod:
            assert code_to_feat_attn is not None
            assert td_synth is not None

    def _init_prompt(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.feature_extractor.model.patch_embed.patch_size, 1) + self.conditioning.object_dim))
        nn.init.uniform_(self.initial_code.data, -val, val)

    def get_initial_prompt(self, batch_size):
        if self.use_initial_code and (not self.use_running_prompt):
            return self.initial_code.expand(batch_size, -1, -1)
        else:
            return None
        
    def mask_features(self, feature: FeatureExtractorOutput, prob: float):
        # replace the embeddings of 2nd dimension with mask token
        if prob == 0:
            return feature
        
        b, n, d = feature.features.shape
        
        mask = torch.rand_like(feature.features[:, :, 0]) < prob
        mask = repeat(mask, 'b n -> b n d', d=d)
        
        masked_features = torch.where(
            mask, 
            repeat(self.mask_token, 'd -> b n d', b=b, n=n),
            feature.features
        )

        return FeatureExtractorOutput(
            features=masked_features,
            positions=feature.positions,
            aux_features=feature.aux_features,
            prompt=feature.prompt
        )

    def forward(self, image, batch_size):
        initial_code = self.get_initial_prompt(batch_size)

        feature_extractor = self.feature_extractor(image, code=initial_code)

        conditioning = self.conditioning(batch_size=batch_size)
        
        if self.training:
            do_feedback = torch.rand(1) < self.feedback_prob
        else:
            do_feedback = self.feedback_prob > 0

        if self.pre_slot_num is not None and do_feedback:
            pre_conditioning = self.conditioning(batch_size=batch_size, num_slots=self.pre_slot_num)
        else:
            pre_conditioning = conditioning

        masked_feature_extractor = self.mask_features(feature_extractor, self.pre_masking_prob)

        #  pre slot attention
        perceptual_grouping = self.perceptual_grouping(
            feature=masked_feature_extractor, 
            conditioning=pre_conditioning
        )

        # compute td feature using slot attn map and enc feature
        td_feat = perceptual_grouping.objects
        if self.td_synth is not None:
            td_feat = self.td_synth(
                perceptual_grouping.objects,
                perceptual_grouping.feature_attributions,
                masked_feature_extractor.features
            )

        # online clustering via vector quantization
        code, indices, commit_loss = self.pool(td_feat)

        # compute cosine map between code and enc feature
        code_to_feat_attn = None
        if self.code_to_feat_attn is not None:
            code_to_feat_attn = self.code_to_feat_attn(code, masked_feature_extractor.features)

        if do_feedback:
            post_feature_extractor = self.feature_extractor(image, code=code)
            pg_input = {
                "feature": post_feature_extractor,
                "conditioning": self.p2s(feature_extractor.prompt) if self.use_prompt_as_slot else conditioning
            }
            if self.use_attn_mod:
                pg_input["attn_mod"] = code_to_feat_attn
            post_perceptual_grouping = self.perceptual_grouping(**pg_input)
        else:
            post_feature_extractor = feature_extractor
            post_perceptual_grouping = perceptual_grouping

        # reconstuct original feature using decoder
        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
            target=feature_extractor.features if self.use_pre_target else post_feature_extractor.features,
            image=image
        )

        if torch.isnan(object_decoder.reconstruction).any():
            import ipdb; ipdb.set_trace()

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=post_feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob,
            code_to_feat_attn=code_to_feat_attn
        )
    

class FeedbackCodeAttn(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: AutoregressivePatchDecoder,
        feedback_prob: float = 0.5,
        td_synth: Optional[nn.Module] = None,
        code_to_feat_attn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

        if td_synth is not None:
            self.td_synth = td_synth

        if code_to_feat_attn is not None:
            self.code_to_feat_attn = code_to_feat_attn

    def forward(self, image, batch_size):

        feature_extractor = self.feature_extractor(image)

        conditioning = self.conditioning(batch_size=batch_size)
        
        if self.training:
            do_feedback = torch.rand(1) < self.feedback_prob
        else:
            do_feedback = self.feedback_prob > 0

        pre_conditioning = conditioning

        #  pre slot attention
        perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor, 
            conditioning=pre_conditioning
        )

        # compute td feature using slot attn map and enc feature
        td_feat = perceptual_grouping.objects
        if self.td_synth is not None:
            td_feat = self.td_synth(
                perceptual_grouping.objects,
                perceptual_grouping.feature_attributions,
                feature_extractor.features
            )

        # online clustering via vector quantization
        code, indices, commit_loss = self.pool(td_feat)

        # compute cosine map between code and enc feature
        code_to_feat_attn = None
        if self.code_to_feat_attn is not None:
            code_to_feat_attn = self.code_to_feat_attn(code, feature_extractor.features)

        if do_feedback:
            pg_input = {
                "feature": feature_extractor,
                "conditioning": conditioning,
                "attn_mod": code_to_feat_attn
            }
            post_perceptual_grouping = self.perceptual_grouping(**pg_input)
        else:
            post_perceptual_grouping = perceptual_grouping

        # reconstuct original feature using decoder
        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
            target=feature_extractor.features,
            image=image
        )

        if torch.isnan(object_decoder.reconstruction).any():
            import ipdb; ipdb.set_trace()

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob,
            code_to_feat_attn=code_to_feat_attn
        )
    

class FeedbackCodeIA3(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: AutoregressivePatchDecoder,
        feedback_prob: float = 1,
        td_synth: Optional[nn.Module] = None,
        pre_implicit: bool = False,
        post_implicit: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

        self.td_synth = td_synth

        self.pre_implicit = pre_implicit
        self.post_implicit = post_implicit

    def forward(self, image, batch_size=1):

        feature_extractor = self.feature_extractor(image)

        conditioning = self.conditioning(batch_size=batch_size)
        
        #  pre slot attention
        perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor, 
            conditioning=conditioning,
            use_implicit_differentiation=self.pre_implicit
        )

        # compute td feature using slot attn map and enc feature
        td_feat = perceptual_grouping.objects
        if self.td_synth is not None:
            td_feat = self.td_synth(
                perceptual_grouping.objects,
                perceptual_grouping.feature_attributions,
                feature_extractor.features
            )

        # online clustering via vector quantization
        code, indices, commit_loss = self.pool(td_feat)
        
        if self.training:
            do_feedback = torch.rand(1) < self.feedback_prob
        else:
            do_feedback = self.feedback_prob > 0

        if do_feedback:
            post_perceptual_grouping = self.perceptual_grouping(
                feature=feature_extractor,
                conditioning=conditioning,
                code=code,
                pre_attn=perceptual_grouping.feature_attributions,
                use_implicit_differentiation=self.post_implicit
            )
        else:
            post_perceptual_grouping = perceptual_grouping

        # reconstuct original feature using decoder
        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
            target=feature_extractor.features,
            image=image
        )

        if torch.isnan(object_decoder.reconstruction).any():
            import ipdb; ipdb.set_trace()

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob,
        )
    

class FeedbackCodeIA3NoSharing(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: nn.Module,
        pre_conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        pre_perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: AutoregressivePatchDecoder,
        feedback_prob: float = 1,
        td_synth: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

        self.td_synth = td_synth

        self.pre_conditioning = pre_conditioning
        self.pre_perceptual_grouping = pre_perceptual_grouping

    def forward(self, image, batch_size):

        feature_extractor = self.feature_extractor(image)

        pre_conditioning = self.pre_conditioning(batch_size=batch_size)
        
        #  pre slot attention
        pre_perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor, 
            conditioning=pre_conditioning
        )

        # compute td feature using slot attn map and enc feature
        td_feat = pre_perceptual_grouping.objects
        if self.td_synth is not None:
            td_feat = self.td_synth(
                pre_perceptual_grouping.objects,
                pre_perceptual_grouping.feature_attributions,
                feature_extractor.features
            )

        # online clustering via vector quantization
        code, indices, commit_loss = self.pool(td_feat)
        

        conditioning = self.conditioning(batch_size=batch_size)
        post_perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor,
            conditioning=conditioning,
            code=code,
            pre_attn=pre_perceptual_grouping.feature_attributions
        )

        # reconstuct original feature using decoder
        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
            target=feature_extractor.features,
            image=image
        )

        if torch.isnan(object_decoder.reconstruction).any():
            import ipdb; ipdb.set_trace()

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=pre_perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob,
        )


class FeedbackCodeIA3PatchDec(nn.Module):
    def __init__(
        self,
        feature_extractor: TimmFeatureExtractorPrompt,
        conditioning: nn.Module,
        perceptual_grouping: nn.Module,
        pool: nn.Module,
        object_decoder: nn.Module,
        feedback_prob: float = 1,
        td_synth: Optional[nn.Module] = None,
        pre_implicit: bool = False,
        post_implicit: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conditioning = conditioning
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.pool = pool

        self.feedback_prob = feedback_prob

        self.td_synth = td_synth

        self.pre_implicit = pre_implicit
        self.post_implicit = post_implicit

    def forward(self, image, batch_size):

        feature_extractor = self.feature_extractor(image)

        conditioning = self.conditioning(batch_size=batch_size)
        
        #  pre slot attention
        perceptual_grouping = self.perceptual_grouping(
            feature=feature_extractor, 
            conditioning=conditioning,
            # use_implicit_differentiation=self.pre_implicit
        )

        # compute td feature using slot attn map and enc feature
        td_feat = perceptual_grouping.objects
        if self.td_synth is not None:
            td_feat = self.td_synth(
                perceptual_grouping.objects,
                perceptual_grouping.feature_attributions,
                feature_extractor.features
            )

        # online clustering via vector quantization
        code, indices, commit_loss = self.pool(td_feat)
        
        if self.training:
            do_feedback = torch.rand(1) < self.feedback_prob
        else:
            do_feedback = self.feedback_prob > 0

        if do_feedback:
            post_perceptual_grouping = self.perceptual_grouping(
                feature=feature_extractor,
                conditioning=conditioning,
                code=code,
                pre_attn=perceptual_grouping.feature_attributions,
                use_implicit_differentiation=self.post_implicit
            )
        else:
            post_perceptual_grouping = perceptual_grouping

        # reconstuct original feature using decoder
        object_decoder = self.object_decoder(
            object_features=post_perceptual_grouping.objects,
        )

        if torch.isnan(object_decoder.reconstruction).any():
            import ipdb; ipdb.set_trace()

        return ModelOutput(
            feature_extractor=feature_extractor,
            post_feature_extractor=feature_extractor,
            conditioning=conditioning,
            perceptual_grouping=perceptual_grouping,
            post_perceptual_grouping=post_perceptual_grouping,
            object_decoder=object_decoder,
            pool_indices=indices,
            commit_loss=commit_loss.sum(),
            feedback_prob=self.feedback_prob,
        )