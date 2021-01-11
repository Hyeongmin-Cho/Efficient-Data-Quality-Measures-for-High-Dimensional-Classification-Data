from measure import get_lda_object, load_data
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, help='Dataset name')
parser.add_argument('--root', type=str, help='Dataset folder')
parser.add_argument('--ratio', type=float, help='Sampling ratio')
parser.add_argument('--sampling_count', type=int, help='Sampling count')
parser.add_argument('--vec', type=int, help='Number of projection vectors')
parser.add_argument('--is_training', type=bool, default=True ,help='Quality evaluation type (for train or test dataset)')
args = parser.parse_args()

if __name__=='__main__':
    stime=time.time()
    indicator = get_lda_object(args.root, args.dataset_name, args.is_training, args.ratio, args.sampling_count, args.vec)
    Msep, Mvar = indicator.coherence(args.vec)
    etime=time.time()
    
    print('Msep :', Msep)
    print('Mvar :', Mvar)
    print('Elapsed time : %.2f sec' %(etime-stime))