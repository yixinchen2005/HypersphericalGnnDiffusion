
from trainer_ner import Trainer as Trainer_ner
from options import get_parser
import os
from utils import ensure_reproducibility

def main():
    ensure_reproducibility(3407)
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.base == 'ner':
        trainer = Trainer_ner(args)

    trainer.load("best_f1_0.9658")
    trainer.eval_epoch('test')

if __name__ == '__main__':
    main()
