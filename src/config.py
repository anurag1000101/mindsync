##################################################################################################
## Important Intializations
##################################################################################################
class ModelConfig_wordlevel:
    def __init__(self):
        self.alpha = 0.35  ## wieght to BART loss
        # self.window_size = 750
        self.d_eeg = 105 ## input channel
        self.max_tokens = 1024
        # self.model_conformer = "facebook/wav2vec2-conformer-rope-large-960h-ft"

        # Encoder configuration
        self.encoder_input_dim = 1024
        self.encoder_output_dim = 512
        self.encoder_num_layers = 3
        self.encoder_num_heads = 8
        self.encoder_ff_dim = 1024
        self.encoder_dropout = 0.1

        # VQ-VAE configuration
        self.vqvae_num_embeddings = 2048
        self.vqvae_embedding_dim = 512
        self.vqvae_commitment_cost = 0.25
        self.vqvae_decay = 0
        self.vqvae_epsilon = 0.00001

        # Decoder configuration
        self.decoder_input_dim = 512
        self.decoder_output_dim = 512
        self.decoder_num_layers = 3
        self.decoder_num_heads =  8
        self.decoder_ff_dim = 1024
        self.decoder_dropout = 0.1
        self.decoder_input_channels = 512
        self.decoder_output_channels = 1024

        # NeuroBART configuration
        self.neurobart_in_feature = 512
        self.neurobart_encoder_layers =  6
        self.neurobart_additional_encoder_nhead = 8
        self.neurobart_additional_encoder_dim_feedforward = 2048
        self.neurobart_model_name = 'facebook/bart-large' 

        ## for mae-eeg encoder
        self.mae_chkpt = "/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/models/mae_eeg_chkpt.pth"
        self.time_len =  3500 #2939 ## mean of all time-len in Zuco2.0

        ## training specific args
        self.epochs = 100
        self.batch_size = 64
        self.preload = "/nlsasfs/home/nltm-st/sujitk/temp/eeg2text/models/final-eeg2text_wordLevel__task1_task2_taskNRv2_20240423_234616/best_chkpt/bestModel"
        self.lr = 1e-4
        self.lr1 = 1e-4
        self.lr2 =  1e-6
        self.stage1 = 20
        # self.task_name = ["task1", "task2", "task3", "taskNRv2"]
        self.task_name = ["task1", "task2", "taskNRv2"]
