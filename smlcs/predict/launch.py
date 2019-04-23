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

frames = []
widgets = []
global o


class AutoScrollbar(Scrollbar):
    # A scrollbar that hides itself if it's not needed.
    # Only works if you use the grid geometry manager!
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise TclError("cannot use pack with this widget")

    def place(self, **kw):
        raise TclError("cannot use place with this widget")


class ScrollFrame:
    def __init__(self, master):
        self.vscrollbar = AutoScrollbar(master)
        self.vscrollbar.grid(row=0, column=1, sticky=N + S)
        self.hscrollbar = AutoScrollbar(master, orient=HORIZONTAL)
        self.hscrollbar.grid(row=1, column=0, sticky=E + W)

        self.canvas = Canvas(master, yscrollcommand=self.vscrollbar.set,
                             xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)

        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)

        # make the canvas expandable
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # create frame inside canvas
        self.frame = Frame(self.canvas)
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(1, weight=1)

        self.frame.bind("<Configure>", self.reset_scrollregion)

    def update(self):
        self.canvas.create_window(0, 0, anchor=NW, window=self.frame)
        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        if self.frame.winfo_reqwidth() != self.canvas.winfo_width():
            # update the canvas's width to fit the inner frame
            self.canvas.config(width=self.frame.winfo_reqwidth())
        if self.frame.winfo_reqheight() != self.canvas.winfo_height():
            # update the canvas's width to fit the inner frame
            self.canvas.config(height=self.frame.winfo_reqheight())

    def reset_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class Launch:
    try:
        def lauch_gui(self, path_name):
            predictor = Predictor(self.logger)
            path = path_name
            logger.info('Path to the program feature file: {}'.format(path))
            programs, program_count, config_count, clf_correct, rt_correct, clf_all, all_rtime= predictor.predictor(path)
            print(len(all_rtime))
            print(len(clf_all))
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

            res = Label(o.frame, text='Program', font=("Arial Bold", 10))
            res.grid(row=4, column=0)
            res = Label(o.frame, text='Configurations', font=("Arial Bold", 10))
            res.grid(row=4, column=1)
            res = Label(o.frame, text='Predict_prob', font=("Arial Bold", 10))
            res.grid(row=4, column=2)
            res = Label(o.frame, text='Runtime', font=("Arial Bold", 10))
            res.grid(row=4, column=3)

            for i in range(0, len(list_1)):
                #print('{} -- {} -- {}'.format(list_1[i], list_2[i], list_3[i]))
                res = Label(o.frame, text=list_1[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=0)

                res = Label(o.frame, text=list_2[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=1)

                res = Label(o.frame, text=list_3[i], font=("Times New Roman", 10))
                res.grid(row=i + 5, column=2)

                res = Label(o.frame, text=str(list_4[i]), font=("Times New Roman", 10))
                res.grid(row=i + 5, column=3)

                verify_button = Button(o.frame, text="Verify", command=partial(verify, list_1[i], list_2[i], i + 5))
                verify_button.grid(row=i + 5, column=4)

                button_identities.append(verify_button)

            for i in all_rtime:
                print(i)

            print('===================')
            print('===================')
            for j in clf_all:
                print(j[0])
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
            varification_result = Label(o.frame, text='True', font=("Arial Bold", 10), fg="green")
            varification_result.grid(row=row, column=5)

        def browse_button(roots):
            global folder_path
            global filename
            filename = filedialog.askopenfilename()
            folder_path.set('Selected file: ' + str(filename))
            launch = Launch(logger, roots)
            predict_button = Button(o.frame, text="Predict", command=lambda: launch.lauch_gui(filename))
            predict_button.grid(row=2, column=1)

        root = Tk()

        o = ScrollFrame(root)
        root.title("smlcs")
        label = Label(o.frame, text="Supervised learning approach for CPAchecker Configuration Selection",
                      font=("Arial Bold", 20))
        label.grid(row=0, column=4)

        select = Label(o.frame, text="Please select the program feature file", font=("Arial Bold", 10))
        select.grid(row=1, column=0)

        folder_path = StringVar()
        lbl1 = Label(master=o.frame, textvariable=folder_path)
        lbl1.grid(row=2, column=0)
        button2 = Button(o.frame, text="Browse", command=lambda: browse_button(root))
        button2.grid(row=1, column=1)
        o.update()
        root.mainloop()

    except Exception as e:
        logger.error('Failed in the main of launch.py: ' + str(e))
