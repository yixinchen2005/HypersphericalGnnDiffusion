from trainer_ner import Trainer as Trainer_ner
from options import get_parser
import os


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.base == 'ner':
        trainer = Trainer_ner(args)

    trainer.train()
    trainer.save()
    trainer.eval_epoch('test')


if __name__ == '__main__':
    main()