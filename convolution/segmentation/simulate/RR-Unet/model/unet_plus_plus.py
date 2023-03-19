import segmentation_models_pytorch as smp
from .hf_config import UnetConfig
from transformers import PreTrainedModel


class HFUnetPlusPlus(PreTrainedModel):
    config_class = UnetConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights="imagenet",
            decoder_channels=config.decoder_channels,
            in_channels=config.input_channels,
            classes=config.num_classes,
            decoder_attention_type="scse")

    def forward(self, tensor):
        return self.model(tensor)
