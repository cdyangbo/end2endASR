'''@file asr_factory
contains the asr factory'''

from . import ds2_classifier, las_classifier
from six.moves import configparser

def proc_las_args(args):
    ad = args.__dict__

    ad['input_dim'] = args.features
    ad['output_dim'] = args.classes

    parsedlas_cfg = configparser.ConfigParser()
    parsedlas_cfg.read('conf/las_network.conf')
    las_cfg = dict(parsedlas_cfg.items('las'))

    if not ad.has_key('add_labels'):
        ad['add_labels'] = las_cfg['add_labels']

    if not ad.has_key('listener_numlayers'):
        ad['listener_numlayers'] = las_cfg['listener_numlayers']

    if not ad.has_key('listener_numunits'):
        ad['listener_numunits'] = las_cfg['listener_numunits']

    if not ad.has_key('listener_dropout'):
        ad['listener_dropout'] = las_cfg['listener_dropout']

    return ad


def model_factory(args, name=None, server = None, device = None):
    '''
    create an asr classifier
    Args:
        args: the classifier config as a dictionary
        name: the classifier name
    Returns:
        A Classifier object
    '''

    if args.model == 'ds2':
        return ds2_classifier.DeepSpeech2(args, name or args.model, server=server, device=device)

    elif args.model == 'las':
        return las_classifier.LAS_Classifier(proc_las_args(args), output_dim=args.classes, name=name or args.model)

    else:
        raise Exception('undefined asr type: %s' % args.model)
