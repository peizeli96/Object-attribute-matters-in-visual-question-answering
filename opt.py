import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument("--pretrain", action="store_true", help='use gqa2.0 or not')

    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/GQA4.29')

    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
        
    parser.add_argument('--epochs', type=int, default=100, help='the number of epoches')


    parser.add_argument('--batch_size', type=int, default=320,
                    help='minibatch size')
    parser.add_argument('--print_interval', default=200, type=int, metavar='N',
                        help='print per certain number of steps')

    parser.add_argument('--dataset', type=str, default='GQA4.29', choices=['GQA4.29'],
                        help='Dataset to train and evaluate')

#################### Qembedding
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint language features')

    parser.add_argument('--activation', type=str, default='swish', choices=['relu', 'swish'],
                        help='the activation to use for final classifier')


    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
                      

    # optimation
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')
    parser.add_argument('--weight_init', type=str, default='none', choices=['none', 'kaiming_normal'],
                        help='dynamic weighting with Kaiming normalization')


    # Optimization: General
    parser.add_argument('--lxmert_lr', default=1e-5, type=float, metavar='lr',
                        help='initial learning rate')

    # parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=8)
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')
    




    args = parser.parse_args()

    return args
