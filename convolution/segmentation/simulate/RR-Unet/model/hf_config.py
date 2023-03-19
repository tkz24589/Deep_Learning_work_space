from transformers import PretrainedConfig

class UnetConfig(PretrainedConfig):

    def __init__(
        self,
        encoder_name: str = "resnet18",
        num_classes: int = 1,
        input_channels: int = 3,
        decoder_channels: tuple = (1024, 512, 256, 128, 64),
        **kwargs
    ):
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.decoder_channels = decoder_channels
        super().__init__(**kwargs)
