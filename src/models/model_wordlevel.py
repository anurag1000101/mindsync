import sys
sys.path.append('/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/src')
from config import ModelConfig

from transformers import BartForConditionalGeneration, BartTokenizer,AutoTokenizer
import torch
from torch import nn
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import warnings
from models.mae_for_eeg import *

warnings.filterwarnings("ignore")

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings
    
# Create Encoder
class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Encoder, self).__init__()

        self.embedding = nn.Linear(config.encoder_input_dim, config.encoder_output_dim)
        self.positional_encoding = PositionalEncoding(config.encoder_output_dim, config.max_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_output_dim,
            nhead=config.encoder_num_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.encoder_dropout,
            batch_first=True
        )

        # Add residual connections
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         config.encoder_num_layers, 
                                                         norm=nn.LayerNorm(config.encoder_output_dim))

    def forward(self, inputs, input_mask_invert):
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(src=x, src_key_padding_mask=input_mask_invert)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_tokens=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_tokens, d_model)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Create Decoder
class Decoder(nn.Module):
    def __init__(self,config: ModelConfig):
        super(Decoder, self).__init__()

        self.embedding = nn.Linear(config.decoder_input_dim, 
                                   config.decoder_input_dim)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.relu = nn.ReLU()

        
        self.positional_encoding = PositionalEncoding(config.decoder_output_dim, config.max_tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_output_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.decoder_dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.decoder_num_layers, norm=nn.LayerNorm(config.encoder_output_dim))
        # self.conv1 = nn.Conv1d(config.decoder_input_channels, 
        #                        config.decoder_output_channels, 
        #                        kernel_size=3, stride=2, padding=1)
        # self.upsample_ff=nn.Linear(config.decoder_input_dim,
        #                            1024)
        # Deconvolution operations 

    def forward(self, inputs, input_mask_invert):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(src = x, src_key_padding_mask = input_mask_invert)
        # x = self.upsample_ff(x)
        return x

class EEG_Model(nn.Module):
    def __init__(self,config: ModelConfig,decay=0):
        super(EEG_Model, self).__init__()
        self.d_eeg= config.d_eeg
        self._encoder = Encoder(config)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(config.vqvae_num_embeddings, 
                                              config.vqvae_embedding_dim, 
                                              config.vqvae_commitment_cost,
                                              config.vqvae_decay,
                                              config.vqvae_epsilon)
        else:
            self._vq_vae = VectorQuantizer(   config.vqvae_num_embeddings, 
                                              config.vqvae_embedding_dim, 
                                              config.vqvae_commitment_cost)
        self._decoder = Decoder(config)
        
    def forward(self, x, input_mask_invert):
        # print(f"Inside EEG_Model: {x.shape}")
        z = self._encoder(x, input_mask_invert) ## (1, N, 512)
        assert z is not None
        # print(f'x: {x}')
        # print(f'z: {z}')
        # print(f"Encoder output: {z.shape}")
        loss_vae, quantized, perplexity, _ = self._vq_vae(z) ## (1, N, 512)
        # print(f'loss_vae, quantized, perplexity: {loss_vae}, {quantized}, {perplexity}')
        x_recon = self._decoder(quantized, input_mask_invert) ## (1, N, 1024)
        # print(f"Out of decoder: {x_recon.shape}")
        loss_recon=F.mse_loss(z, x_recon)
        # print(f'loss_recon: {loss_recon}')
        # print(f"Exiting EEG_Model\n")

        return loss_vae,loss_recon, quantized, perplexity


## BART Model for textual training
class NeuroBART(nn.Module):
    def __init__(self, config: ModelConfig):
        super(NeuroBART, self).__init__()

        self.pretrained = BartForConditionalGeneration.from_pretrained(config.neurobart_model_name)
        self.decoder_embedding_size = self.pretrained.config.d_model

        # additional transformer encoder
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=config.neurobart_in_feature, 
                                                                   nhead=config.neurobart_additional_encoder_nhead,
                                                                   dim_feedforward=config.neurobart_additional_encoder_dim_feedforward,
                                                                   batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=config.neurobart_encoder_layers,
                                                                norm=nn.LayerNorm(config.encoder_output_dim))

        self.fc1 = nn.Linear(config.neurobart_in_feature, self.decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=None,
            assistant_model=None,
            streamer=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            beams: int = 10,
            **kwargs,
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            labels=target_ids_batch_converted,
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs, 
            num_beams=beams  # Set beam search size to 14
        )

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        # print(f"Input to bart: {encoded_embedding.shape} \n {encoded_embedding}")
        encoded_embedding = torch.tensor(encoded_embedding).float()
        self.pretrained = self.pretrained.to(encoded_embedding.device)
        out = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch,
                              return_dict=True, labels=target_ids_batch_converted)
        # out = self.pretrained.base_model.encoder(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch,
        #                       return_dict=True)
        # out=self.pretrained.generate(encoder_outputs=out)
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        # tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return out

class MindSync(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MindSync, self).__init__()
        ## make projection Layer for projection frm 840 to 512
        config.encoder_input_dim = 840
        self.alpha = config.alpha
        self.config = config
        self.max_length = config.max_tokens
        ## featuren extraction: mae for eeg 
        self.mae = eeg_encoder(time_len= config.time_len ,in_chans = config.d_eeg, path = config.mae_chkpt)
        ## freeze all the params except the intial patch_embed and pos_embed as we changed the 'time_len')
        eeg_encoder.freeze_parameters(self.mae)

        ## defining the encoder-decoder model for feature extraction
        self.eegEncoder = EEG_Model(config)
        ## making the BART model
        self.neuro_bart  = NeuroBART(config)

    def freeze_bart_parameters(self, freeze: bool = True):
        for param in self.neuro_bart.parameters():
            param.requires_grad = not freeze

    def unfreeze_bart_parameters(self):
        self.freeze_bart_parameters(False)

    def forward(self, x , target_ids_batch_converted, input_attention_mask, input_masks_invert, freezeBart: bool = False):
        B, _, _ = x.shape
        ## Step 1: feed the Encoder
        # print('\nBefore self.eegEncoder(x...)\n')
        loss_vae,loss_recon, quantized, perplexity = self.eegEncoder(x, input_masks_invert)
        # print(f'quantized, perplexity: {quantized}, {perplexity}')
        if freezeBart:
            loss = loss_vae + loss_recon
            return loss, None
        else: 
             # print(f"AFter encode: loss: {loss_vae}, loss_recon: {loss_recon}m quantized: {quantized.shape}")
            out = self.neuro_bart(quantized, input_attention_mask, input_masks_invert, target_ids_batch_converted)
            ## compiling final loss
            # print(f'loss_vae + loss_recon + (self.alpha*out.loss): {loss_vae} + {loss_recon} + ({self.alpha} * {out.loss})')

            loss = loss_vae + loss_recon + (self.alpha*out.loss)
            return loss, out

    @torch.no_grad()
    def generate(self, x, target_ids_batch_converted, input_attention_mask, input_masks_invert, **kwargs):
        B, _, _ = x.shape

        ## Step 3: feed the Encoder
        loss_vae,loss_recon, quantized, perplexity = self.eegEncoder(x, input_masks_invert)
        out = self.neuro_bart.generate(quantized, input_attention_mask, input_masks_invert, target_ids_batch_converted, **kwargs)
        return out



# '''sanity test'''
# if __name__ == '__main__':
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print(f"Device in use: {device}")
 
#     config = ModelConfig()
#     B = 8
#     # input=torch.randn(B, 105,4200)
#     input=torch.randn(B, 56,840)
#     mask = torch.zeros(B, 56).to(device)
#     mask_invert = torch.ones(B, 56).to(device)
#     model=MindSync(config, False)
#     target_id=torch.randint(low=0, high=100, size=(B, 20), dtype=torch.long)

#     input = input.to(device)
#     model = model.to(device)
#     target_id = target_id.to(device)

#     # Process the wave ## make the processor
#     loss ,out = model(input, target_id, mask_invert, mask)

#     print(f"The final loss: {loss}\n out: {out}")

#     # print("For generate")
#     # print(model.generate(input, target_id, mask_invert, mask))
#     print("Done sanity check!")