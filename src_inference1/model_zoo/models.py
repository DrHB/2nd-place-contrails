# TODO : DOC

import torch
import torch.nn as nn
import segmentation_models_pytorch

from .unet import Unet
from .transformer import Tmixer

import os,json


DECODERS = [
    "Unet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
    "UnetPlusPlus",
]


def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    encoder_weights="imagenet",
    pretrained=True,
    n_channels=3,
    pretrained_weights=None,
    reduce_stride=False,
    upsample=False,
    use_pixel_shuffle=False,
    use_hypercolumns=False,
    center="none",
    use_cls=False,
    frames=4,
    use_lstm=False,
    bidirectional=False,
    use_cnn=False,
    kernel_size=5,
    use_transfo=False,
    two_layers=False,
    verbose=0,
):
    """
    Loads a segmentation architecture.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
        encoder_weights (str, optional): Pretrained weights. Defaults to "imagenet".

    Returns:
        torch model: Segmentation model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"

    if decoder_name == "Unet":
        model = Unet(
            encoder_name="tu-" + encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=n_channels,
            classes=num_classes,
            use_pixel_shuffle=use_pixel_shuffle,
            use_hypercolumns=use_hypercolumns,
            center=center,
            use_lstm=use_lstm,
            aux_params={"dropout": 0.2, "classes": num_classes} if use_cls else None,
        )
    elif decoder_name == "FPN" and "nextvit" in encoder_name:
        from mmseg.models import build_segmentor
        from mmcv.utils import Config
        import sys
        sys.path.append('../nextvit/segmentation/')
        # from nextvit import nextvit_small

        cfg = Config.fromfile(f"../nextvit/segmentation/configs/fpn_512_{encoder_name}_80k.py")
        model = build_segmentor(cfg.model)
        if pretrained:
            state_dict = torch.load(f'../input/fpn_80k_{encoder_name}_1n1k6m_pretrained.pth')["state_dict"]
            del state_dict['decode_head.conv_seg.weight'], state_dict['decode_head.conv_seg.bias']
            model.load_state_dict(state_dict, strict=False)
        model.backbone.stem[0].conv.stride = (1, 1)
        model.backbone.stem[3].conv.stride = (1, 1)
        model = nn.Sequential(model.backbone, model.neck, model.decode_head)

    else:
        decoder = getattr(segmentation_models_pytorch, decoder_name)
        model = decoder(
            encoder_name="tu-" + encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=n_channels,
            classes=num_classes,
            aux_params={"dropout": 0.2, "classes": 1} if use_cls else None,
            upsampling=int(4 // 2 ** reduce_stride),
        )

    model.num_classes = num_classes

    model = SegWrapper(
        model,
        use_cls,
        frames=frames,
        use_lstm=use_lstm,
        bidirectional=bidirectional,
        use_cnn=use_cnn,
        kernel_size=kernel_size,
        use_transfo=use_transfo,
        two_layers=two_layers,
    )
    
    model.upsample = 2 ** reduce_stride if upsample else 0
    model.reduce_stride(encoder_name, decoder_name, reduce_stride)

    if pretrained_weights is not None:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
        del (
            state_dict['model.segmentation_head.0.weight'],
            state_dict['model.segmentation_head.0.bias'],
        )
        model.load_state_dict(state_dict, strict=False)

    return model


class SegWrapper(nn.Module):
    def __init__(
        self,
        model,
        use_cls=False,
        frames=4,
        use_lstm=False,
        bidirectional=False,
        use_cnn=False,
        kernel_size=3,
        use_transfo=False,
        two_layers=False,
    ):
        """
        Constructor.
        TODO

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.model = model
        self.num_classes = model.num_classes
        self.use_cls = use_cls
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.use_transfo = use_transfo
        self.frames = frames
        self.two_layers = two_layers

        if use_lstm or use_cnn:
            assert isinstance(frames, (tuple, list)), "frames must be tuple or int"
            assert (len(frames) > 1) and (4 in frames), "several frames expected, 4 has to be included"
        if use_transfo:
            assert not use_lstm and not use_cnn, "Cannot use transformer and lstm/cnn"

        if self.use_lstm:
            self.lstm = nn.LSTM(
                model.encoder.out_channels[-1],
                model.encoder.out_channels[-1] // (1 + bidirectional),
                batch_first=True,
                bidirectional=bidirectional
            )
            if self.two_layers:
                self.lstm_2 = nn.LSTM(
                    model.encoder.out_channels[-2],
                    model.encoder.out_channels[-2] // (1 + bidirectional),
                    batch_first=True,
                    bidirectional=bidirectional
                )
        if self.use_cnn:
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)

            self.cnn = nn.Sequential(
                nn.Conv3d(
                    model.encoder.out_channels[-1],
                    model.encoder.out_channels[-1],
                    kernel_size=kernel_size,
                    padding=(kernel_size[0] // 2 if use_lstm else 0, kernel_size[1] // 2, kernel_size[2] // 2)
                ),
                nn.BatchNorm3d(model.encoder.out_channels[-1]),
                nn.ReLU(),
            )
            if self.two_layers:
                self.cnn_2 = nn.Sequential(
                    nn.Conv3d(
                        model.encoder.out_channels[-2],
                        model.encoder.out_channels[-2],
                        kernel_size=kernel_size,
                        padding=(kernel_size[0] // 2 if use_lstm else 0, kernel_size[1] // 2, kernel_size[2] // 2)
                    ),
                    nn.BatchNorm3d(model.encoder.out_channels[-2]),
                    nn.ReLU(),
                )

        if self.use_transfo:
            self.transfo = Tmixer(model.encoder.out_channels[-1])
            if self.two_layers:
                self.transfo_2 = Tmixer(model.encoder.out_channels[-2])

    def reduce_stride(self, encoder_name, decoder_name="Unet", reduce_stride=0):
        if "nextvit" in encoder_name:
            return

        if "swinv2" in encoder_name:
            assert reduce_stride == 2
            if decoder_name == "Unet":
                if len(self.model.decoder.blocks) >= 4:
                    self.model.decoder.blocks[3].upscale = False
                    self.model.decoder.blocks[3].pixel_shuffle = nn.Identity()
            return
        
        if reduce_stride == 0:
            return

        if not self.upsample:
            if "nfnet" in encoder_name:
                self.model.encoder.model.stem_conv1.stride = (1, 1)
            elif "efficientnet" in encoder_name:
                self.model.encoder.model.conv_stem.stride = (1, 1)
            elif "resnet" in encoder_name or "resnext" in encoder_name:
                try:
                    self.model.encoder.model.conv1[0].stride = (1, 1)
                except:
                    self.model.encoder.model.conv1.stride = (1, 1)
            elif "convnext" in encoder_name:
                self.model.encoder.stem[0].stride = (2, 2)
                self.model.encoder.stem[0].padding = (1, 1)
            else:
                raise NotImplementedError

        if decoder_name == "Unet":
            if len(self.model.decoder.blocks) >= 5:
                self.model.decoder.blocks[4].upscale = False
                self.model.decoder.blocks[4].pixel_shuffle = nn.Identity()

        if reduce_stride >= 2:
            if not self.upsample:
                if "efficientnetv2" in encoder_name:
                    self.model.encoder.model.blocks[1][0].conv_exp.stride = (1, 1)
                elif "efficientnet" in encoder_name:
                    self.model.encoder.model.blocks[1][0].conv_dw.stride = (1, 1)
                elif "convnext" in encoder_name:
                    self.model.encoder.stem[0].stride = (1, 1)
                elif "resnet" in encoder_name or "resnext" in encoder_name:
                    self.model.encoder.model.maxpool.stride = 1
            if decoder_name == "Unet":
                if len(self.model.decoder.blocks) >= 4:
                    self.model.decoder.blocks[3].upscale = False
                    self.model.decoder.blocks[3].pixel_shuffle = nn.Identity()

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        x = x.permute(0,2,1,3,4)[:,self.frames]
        if len(x.size()) == 5:
            bs, n_frames, c, h, w = x.size()
            x = x.reshape(bs * n_frames, c, h, w)
        else:
            assert len(x.size()) == 4, "Length of input size not supported"
            bs, c, h, w = x.size()
            n_frames = 1

        if self.upsample > 1:
#             print(x.size())
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="bilinear")
#             print(x.size())

        features = self.model.encoder(x)
        
#         for ft in features:
#             print(ft.size())

        if self.use_lstm or self.use_cnn or self.use_transfo:
            assert n_frames > 1, "Only one frame, cannot use LSTM / CNN"
            features_ = []
            frame_idx = self.frames.index(4)

            for i, ft in enumerate(features):
#                 print(ft.size())

                if i != len(features) - 1:  # not last layer
                    if self.two_layers and (i == len(features) - 2):
#                         print(i)
                        pass
                    else:
                        ft = ft.view(bs, n_frames, ft.size(1), ft.size(2), ft.size(3))[:, frame_idx]
                        features_.append(ft)
#                         print(f'skip {i}')
                        continue
                
#                 print('2.5D !')
                _, n_fts, h, w = ft.size()
                ft = ft.view(bs, n_frames, n_fts, h, w)

                if self.use_transfo:
                    if i == len(features) - 2:
                        ft = self.transfo_2(ft, frame_idx=frame_idx)
                    else:
                        ft = self.transfo(ft, frame_idx=frame_idx)

                if self.use_cnn:
                    ft = ft.permute(0, 2, 1, 3, 4).contiguous()  # bs x n_fts x n_frames h x w
                    if i == len(features) - 2:
                        ft = self.cnn_2(ft)  # bs x n_fts x h x w
                    else:
                        ft = self.cnn(ft)  # bs x n_fts x h x w

                if self.use_lstm:
                    ft = ft.permute(0, 3, 4, 2, 1).contiguous()  # bs x h x w x n_frames x n_fts
                    ft = ft.view(bs * h * w, n_frames, n_fts)

                    if i == len(features) - 2:
                        ft = self.lstm_2(ft)[0][:, frame_idx]  # bs x h x w x n_fts
                    else:
                        ft = self.lstm(ft)[0][:, frame_idx]  # bs x h x w x n_fts

                    ft = ft.view(bs, h, w, n_fts).permute(0, 3, 1, 2)  # bs x n_fts x h x w

                features_.append(ft.view(bs, n_fts, h, w))

            features = features_

        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
        else:
            labels = torch.zeros(bs, 1).to(x.device)

        return masks[:,:1]#, labels
        
class Config:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

def build_model_theo(exp_folder):
    config = Config(json.load(open(os.path.join(exp_folder,"config.json"), "r")))
    models_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in config.selected_folds:
            model = define_model(
                config.decoder_name,
                config.encoder_name,
                num_classes=config.num_classes,
                n_channels=config.n_channels,
                reduce_stride=config.reduce_stride,
                use_pixel_shuffle=config.use_pixel_shuffle,
                use_hypercolumns=config.use_hypercolumns,
                center=config.center,
                use_cls=config.loss_config['aux_loss_weight'] > 0,
                frames=config.frames if hasattr(config, "use_lstm") else 4,
                use_lstm=config.use_lstm if hasattr(config, "use_lstm") else False,
                bidirectional=config.bidirectional if hasattr(config, "bidirectional") else False,
                use_cnn=config.use_cnn if hasattr(config, "use_cnn") else False,
                kernel_size=config.kernel_size if hasattr(config, "kernel_size") else 1,
                use_transfo=config.use_transfo if hasattr(config, "use_transfo") else False,
                two_layers=config.two_layers if hasattr(config, "two_layers") else False,
                pretrained=False,
            )
            model = model.to(device).eval()

            weights = os.path.join(exp_folder, f"{config.decoder_name}_{config.encoder_name}_{fold}.pt")
            model = load_model_weights(model, weights, verbose=1)

            models_list.append(model)
    return models_list

def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".
        strict (str, optional): Whether to use strict weight loading. Defaults to True.

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        try:
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            state_dict_ = {}
            for k, v in state_dict.items():
                state_dict_[re.sub("module.", "", k)] = v
            model.load_state_dict(state_dict_, strict=strict)

    except BaseException:
        try:  # REMOVE CLASSIFIER
            state_dict_ = copy.deepcopy(state_dict)
            try:
                del (
                    state_dict_["encoder.classifier.weight"],
                    state_dict_["encoder.classifier.bias"],
                )
            except KeyError:
                del (
                    state_dict_["encoder.head.fc.weight"],
                    state_dict_["encoder.head.fc.bias"],
                )
            model.load_state_dict(state_dict_, strict=strict)
        except BaseException:  # REMOVE LOGITS
            try:
                for k in ["logits.weight", "logits.bias"]:
                    try:
                        del state_dict[k]
                    except KeyError:
                        pass
                model.load_state_dict(state_dict, strict=strict)
            except BaseException:
                del state_dict["encoder.conv_stem.weight"]
                model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(
            f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n"
        )

    return model
    
