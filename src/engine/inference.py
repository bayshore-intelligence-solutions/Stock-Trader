from pathlib import Path
import yfinance as yf
from config import Config
import click
from past.builtins import filter
import silence_tensorflow
import tensorflow.compat.v1 as tf
from utils import get_tensorflow_config
from data import StockDataset


def _get_correct_checkpoint(root, checkpoint):
    ckptPath = root.joinpath("checkpoints")
    if ckptPath.is_dir():
        if checkpoint != 'latest':
            allCheckPoints = filter(lambda x: x.name == checkpoint, ckptPath.iterdir())
            relevant_checkpoint = None if len(allCheckPoints) == 0 else allCheckPoints[0]
        else:
            allCheckPoints = list(ckptPath.iterdir())
            relevant_checkpoint = None if len(allCheckPoints) == 0 else sorted(allCheckPoints,
                                                                               key=lambda x: int(x.name),
                                                                               reverse=True)[0]
        return relevant_checkpoint
    else:
        raise IOError("Checkpoint path does not exist!!")


def _get_meta_file(ckptPath):
    # Check the presense of meta file
    metaFile = list(ckptPath.glob("*.meta"))
    if len(metaFile) < 1:
        raise FileNotFoundError("Can not find meta file to restore the graph!!")
    elif len(metaFile) > 1:
        raise RuntimeError("Found multiple meta files to restore the graph!!")
    else:
        metaFile = metaFile[0]
    return metaFile


def _get_live_data():
    data = yf.download('V', period="2d", interval="1m",
                       threads=1, progress=False)
    data = data.tail(31)[["Open", "Close"]]
    return data


@click.command()
@click.option('--config', '-c', default='config.yaml',
              help="Config file that needs to be loaded",
              show_default=True,
              type=click.Path(exists=True))
@click.option(
    '--checkpoint',
    '-ckpt',
    default='latest',
    help="Checkpoint to be loaded",
    type=str,
    show_default=True
)
def main(config, checkpoint):
    """
    Loads a tensorflow pretrained model
    and use it to infer from the pseudo
    live intraday data.
    """

    # Get the live data
    data = _get_live_data()

    # Prepare for the model
    X_test, y_test = StockDataset.prepare_for_test_single(data)

    print(X_test.shape)
    print(y_test.shape)

    conf = Config(config)
    root = conf.root
    ckptPath = _get_correct_checkpoint(root, checkpoint)

    if ckptPath:
        # Get meta file to restore the graph
        metaFile = _get_meta_file(ckptPath)
        # Restore the graph form the meta file
        gconf = get_tensorflow_config()
        with tf.Session(config=gconf) as sess:
            saver = tf.train.import_meta_graph(str(metaFile))
            saver.restore(sess, tf.train.latest_checkpoint(str(ckptPath)))

            # Get all the graph placeholders, namely, X, y, keep_prob
            ts, iS = conf.cell_dim
            inputs = tf.placeholder(tf.float32,
                                    [1, ts, iS],
                                    name="inputs")
            targets = tf.placeholder(tf.float32, [1, iS],
                                     name="targets")
            kp = tf.placeholder(tf.float32, None, name="keep_prob")

            # Prepate the feed dict
            test_data_feed = {
                inputs: X_test,
                targets: y_test,
                kp: 1.0
            }

            test_loss, test_pred = sess.run(feed_dict=test_data_feed)
            print(test_loss)
            print(test_pred)


if __name__ == '__main__':
    main()
