import argparse

_HELP_INPUT_DATA = r"""
Save your input image or video on the same path of parent folder.
"""

_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3')
_METHOD_NAMES = ('cam', 'adl', 'acol', 'spg', 'has', 'cutmix', 'fpn', 'crop')
_NORM_NAMES = ('minmax', 'max', 'pas', 'ivr')
_CODEC_TYPES = ('xvid', 'divx', 'fmp4', 'x264', 'mjpg')


def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(wsol_method):
    if wsol_method in ('has', 'cutmix'):
        architecture_type = 'cam'
    else:
        architecture_type = wsol_method
    return architecture_type


def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--architecture', default='resnet18',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet18)')
    parser.add_argument('--wsol_method', type=str, default='base',
                        choices=_METHOD_NAMES)
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Use pre_trained model.')
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--acol_threshold', type=float, default=0.7)
    parser.add_argument('--original_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--norm_method', type=str, default='minmax',
                        choices=_NORM_NAMES)
    parser.add_argument('--percentile', type=float, default=0.6)
    parser.add_argument('--cam_threshold', type=float, default=0.15)
    parser.add_argument('--checkpoint_path', type=str, default='ckpt/')
    parser.add_argument('--input_name', type=str, default='test.avi', help=_HELP_INPUT_DATA)
    parser.add_argument('--output_name', type=str, default='result.avi')
    parser.add_argument('--video_codec', type=str, default='xvid',
                        choices=_CODEC_TYPES)
    parser.add_argument('--concatenate', type=str2bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()
    args.architecture_type = get_architecture_type(args.wsol_method)
    args.pretrained_path = configure_pretrained_path(args)

    return args

