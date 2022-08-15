import argparse
from datetime import datetime

from src.trainer import Trainer
from src.dataloader import DonutDataset, loader
from src.utils import get_transforms

RUN_START_TIME = datetime.datetime.now()
RUN_TIME_STR = str(RUN_START_TIME).replace(" ", "")
RUN_NAME = "donut_ocr"
RUN_TIMESTAMP = RUN_START_TIME.timestamp()
run_name = f"{RUN_NAME}-{RUN_TIMESTAMP}"

def main(args):
    trainset, testset = DonutDataset(
        args.image_path,
        args.annotation_path,
        transforms=get_transforms(),
    ).split(test_size=args.train_test_split)

    train_loader, test_loader = loader(trainset,
        testset, 
        args.batch_size,
        args.shuffle,
        args.num_workers)
    
    trainer = Trainer(train_loader, args)

    if args.train_type.lower() == 'one':
        trainer.trainOne()
    elif args.train_type.lower() == 'all':
        trainer.train()
    else:
        raise ValueError(f"train_type! must be 'one' or 'all' "
                        f"not {args.train_type}")

if __name__ == '__main__':    

    parse = argparse.ArgumentParser()
    parse.add_argument('--annotation_path', type=str, required=True)
    parse.add_argument('--image_path', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--train_type', type=str, default='one', \
                        help="`one` for single epoch; `all` for `n` epochs")
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--run_name', type=str, default=run_name)
    parse.add_argument('--train_test_split', type=float, default=0.2)
    parse.add_argument('--shuffle', type=bool, default=False)
    parse.add_argument('--num_workers', type=int, default=0)

    args = parse.parse_args()

    main(args)