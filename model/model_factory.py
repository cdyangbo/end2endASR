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

    newdict = dict(ad.items() + las_cfg.items())
    print newdict
    return newdict


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
        classifier =  las_classifier.LAS_Classifier(proc_las_args(args),
                                                    output_dim=args.classes,
                                                    name=name or args.model,
                                                    server=server,
                                                    device=device)
        classifier.build_graph()
        return classifier

    else:
        raise Exception('undefined asr type: %s' % args.model)
