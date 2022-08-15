from datetime import datetime
from src.trainer import trainOne, train

if __name__ == '__main__':
    import argparse
    
    RUN_START_TIME = datetime.datetime.now()
    RUN_TIME_STR = str(RUN_START_TIME).replace(" ", "")
    RUN_NAME = "donut_ocr"
    RUN_TIMESTAMP = RUN_START_TIME.timestamp()
    run_name = RUN_NAME + '_' + RUN_TIMESTAMP

    parse = argparse.ArgumentParser()
    parse.add_argument('--annotation_path', type=str, required=True)
    parse.add_argument('--image_path', type=str, required=True)
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--train_type', type=str, default='all')
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--run_name', type=str, default=run_name)
