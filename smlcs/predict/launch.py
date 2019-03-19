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
from smlcs.predict.predictor import Predictor

from tkinter import filedialog
from tkinter import *
from functools import partial

button_identities = []


class Launch:
    try:
        def lauch_gui(self, path_name):
            predictor = Predictor(self.logger)
            path = path_name
            logger.info('Path to the program feature file: {}'.format(path))
            programs, program_count, config_count, clf_correct, rt_correct, clf_all = predictor.predictor(path)
            list_1 = []
            list_2 = []
            list_3 = []
            list_4 = []
            count = 0
            least_runtime = 300
            for p in programs:
                if p in program_count:
                    while count < len(program_count) and p == program_count[count]:
                        if least_runtime > rt_correct[count]:
                            least_runtime = rt_correct[count]
                            index = count
                        count += 1
                    list_1.append(p)
                    list_2.append(config_count[index])
                    list_3.append(clf_correct[index][0])
                    list_4.append(rt_correct[index])
                    least_runtime = 300
                else:
                    list_1.append(p)
                    list_2.append('--')
                    list_3.append('--')
                    list_4.append('--')

            res = Label(root, text='Program', font=("Arial Bold", 10))
            res.grid(row=4, column=0)
            res = Label(root, text='Configurations', font=("Arial Bold", 10))
            res.grid(row=4, column=1)
            res = Label(root, text='Predict_prob', font=("Arial Bold", 10))
            res.grid(row=4, column=2)
            res = Label(root, text='Runtime', font=("Arial Bold", 10))
            res.grid(row=4, column=3)

            for i in range(0, len(list_1)):
                res = Label(root, text=list_1[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=0)

                res = Label(root, text=list_2[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=1)

                res = Label(root, text=list_3[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=2)

                res = Label(root, text=str(list_4[i]), font=("Times New Roman", 10))
                res.grid(row=i + 5, column=3)

                verify_button = Button(text="Verify", command=partial(verify, list_1[i], list_2[i], i + 5))
                verify_button.grid(row=i + 5, column=4)

                button_identities.append(verify_button)

    except Exception as e:
        print('Error in launch.py')

    def __init__(self, log, ro):
        self.logger = log
        self.root = ro


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


        def verify(program, configuration, row):
            varification_result = Label(root, text='True', font=("Arial Bold", 10), fg="green")
            varification_result.grid(row=row, column=5)

        def browse_button(roots):
            global folder_path
            global filename
            filename = filedialog.askopenfilename()
            folder_path.set('Selected file: ' + str(filename))
            launch = Launch(logger, roots)
            predict_button = Button(text="Predict", command=lambda: launch.lauch_gui(filename))
            predict_button.grid(row=2, column=1)

        root = Tk()
        rows = 0
        while rows < 150:
            root.rowconfigure(rows, weight=1)
            rows += 1

        columns = 0
        while columns < 11:
            root.columnconfigure(columns, weight=1)
            columns += 1

        root.title("smlcs")
        root.geometry('1000x1000')
        lbl = Label(root, text="Supervised learning approach for CPAchecker Configuration Selection",
                    font=("Arial Bold", 20))
        lbl.grid(row=0, column=4)

        select = Label(root, text="Please select the program feature file", font=("Arial Bold", 10))
        select.grid(row=1, column=0)

        folder_path = StringVar()
        lbl1 = Label(master=root, textvariable=folder_path)
        lbl1.grid(row=2, column=0)
        button2 = Button(text="Browse", command=lambda: browse_button(root))
        button2.grid(row=1, column=1)

        root.mainloop()

    except Exception as e:
        logger.error('Failed in the main of launch.py: ' + str(e))
