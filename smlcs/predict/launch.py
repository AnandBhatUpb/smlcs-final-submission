"""This script launches the GUI to select the programs for prediction/ you can give list of programs as command line input.

Usage:
  launch.py --gui=<gui> --programs=<filepath>
  launch.py (-h | --help)
  launch.py

Options:
  -h --help                             Show this screen.
  --gui=<gui>                           Open GUI or not. yes/no
  --programs=<filepath>                 Path for file containing program features
"""

from docopt import docopt
import datetime
import logging


class Launch:
    if __name__ == '__main__':
        try:
            arguments = docopt(__doc__, version=None)
            if arguments['--gui'] is None:
                gui = 'yes'
            else:
                gui = arguments['--gui']

            if arguments['--programs'] is None:
                program_path = ''
            else:
                program_path = arguments['--programs']

            logging.basicConfig(filename='../../logs/prediction.log', filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
            logger = logging.getLogger('Prediction')

            logger.info('Prediction date: {}'.format(datetime.datetime.now()))
            logger.info('GUI : {}'.format(gui))
            logger.info('Program path: {}'.format(program_path))

            if gui == 'yes':
                logger.info('nyi')
            else:
                logger.info('Non GI part is not yet implemented')

        except Exception as e:
            logger.error('Failed in the main of classifier.py: ' + str(e))
