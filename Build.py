import threading
from threading import Lock
import time

import random
import numpy as np

from tkinter import *
from tkinter import filedialog

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as mFigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as mNavigationToolbar2Tk
from matplotlib.figure import Figure as mFigure
from matplotlib import style as mstyle

from stuff.create_theta import random_thetas_sigmoid, random_thetas_reLU
from stuff.download_data import download_data
from stuff.fully_connected import gradient_descent, feed_forward
from stuff.load_save_ai import load_ai, save_ai
from stuff.obj_AI import AI
from stuff.obj_Dataset import Dataset
from stuff.visualize_learning import cost_function, accuracy

matplotlib.use("TkAgg")
mstyle.use("ggplot")


class Build(LabelFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.grid(row=0, column=0, sticky="news")
        self.parent = parent

        #
        self.labelf_shape = LabelFrame(self, text="Shape (neurons by layers)", labelanchor="n")
        self.labelf_shape.grid(row=0, column=0, sticky="nsew")

        self.entry_shape = Entry(self.labelf_shape)
        self.entry_shape.insert(0, "784, 10")
        self.entry_shape.pack()

        #
        self.labelf_af = LabelFrame(self, text="Activation function", labelanchor="n")
        self.labelf_af.grid(row=0, column=1, sticky="news")

        self.spinbox_af = Spinbox(self.labelf_af, justify="center", values=("sigmoid", "reLU"))
        self.spinbox_af.pack()

        #
        self.labelf_epoch = LabelFrame(self, text="Epoch", labelanchor="n")
        self.labelf_epoch.grid(row=1, column=0, sticky="news")

        self.spinbox_epoch = Spinbox(self.labelf_epoch, justify="center", from_=1, to=1000)
        self.spinbox_epoch.pack()

        #
        self.labelf_batch = LabelFrame(self, text="Batch", labelanchor="n")
        self.labelf_batch.grid(row=1, column=1, sticky="news")

        self.spinbox_batch = Spinbox(self.labelf_batch, justify="center", from_=0, to=60000, increment=50)
        self.spinbox_batch.insert(0, "5")
        self.spinbox_batch.pack()

        #
        self.labelf_lambda = LabelFrame(self, text="Regularization strength", labelanchor="n")
        self.labelf_lambda.grid(row=0, column=2, sticky="nsew")

        self.spinbox_lambda = Spinbox(self.labelf_lambda, justify="center", from_=0, to=1, increment=0.01)
        self.spinbox_lambda.pack()

        #
        self.labelf_lr = LabelFrame(self, text="Learning rate", labelanchor="n")
        self.labelf_lr.grid(row=0, column=3, sticky="nsew")

        self.spinbox_lr = Spinbox(self.labelf_lr, justify="center", from_=0, to=1, increment=0.01)
        self.spinbox_lr.pack()

        #
        self.labelf_ntr = LabelFrame(self, text="Nb of training example", labelanchor="n")
        self.labelf_ntr.grid(row=1, column=2, sticky="nsew")

        self.spinbox_ntr = Spinbox(self.labelf_ntr, justify="center", from_=0, to=60000, increment=1000)
        self.spinbox_ntr.delete(0)
        self.spinbox_ntr.insert(0, "60000")
        self.spinbox_ntr.pack()

        #
        self.labelf_nte = LabelFrame(self, text="Nb of test example", labelanchor="n")
        self.labelf_nte.grid(row=1, column=3, sticky="nsew")

        self.spinbox_nte = Spinbox(self.labelf_nte, justify="center", from_=0, to=10000, increment=1000)
        self.spinbox_nte.delete(0)
        self.spinbox_nte.insert(0, "10000")
        self.spinbox_nte.pack()

        #
        self.labelf_ftr = LabelFrame(self, text="Training set path", labelanchor="n")
        self.labelf_ftr.columnconfigure(0, weight=1)
        self.labelf_ftr.grid(row=2, column=0, sticky='news')

        self.entry_ftr = Entry(self.labelf_ftr)
        self.entry_ftr.grid(row=0, column=0, columnspan=4, sticky="news")
        self.entry_ftr.insert(0, "MNIST_data/mnist_train")

        self.button_ftr = Button(self.labelf_ftr, text="D",
                                 command=lambda entry=self.entry_ftr: self.choose_file_entry(entry))
        self.button_ftr.grid(row=0, column=5, sticky="news")

        #
        self.labelf_fte = LabelFrame(self, text="Test set path", labelanchor="n")
        self.labelf_fte.grid(row=2, column=1, sticky='news')
        self.labelf_fte.columnconfigure(0, weight=1)

        self.entry_fte = Entry(self.labelf_fte)
        self.entry_fte.grid(row=0, column=0, columnspan=4, sticky="news")
        self.entry_fte.insert(0, "MNIST_data/mnist_test")

        self.button_fte = Button(self.labelf_fte, text="D",
                                 command=lambda entry=self.entry_fte: self.choose_file_entry(entry))
        self.button_fte.grid(row=0, column=5, sticky="news")

        #
        self.labelf_data = LabelFrame(self, text="Use what on dataset", labelanchor="n")
        self.labelf_data.grid(row=2, column=2, sticky="news")

        self.spinbox_data = Spinbox(self.labelf_data, justify="center",
                                    values=("Nothing", "feature scaling", "mean normalization"))
        self.spinbox_data.pack()

        #
        self.labelf_ai = LabelFrame(self, text="Preexisting neural network", labelanchor="n")
        self.labelf_ai.grid(row=2, column=3, sticky='news')
        self.labelf_ai.columnconfigure(0, weight=1)

        self.entry_ai = Entry(self.labelf_ai)
        self.entry_ai.grid(row=0, column=0, columnspan=4, sticky="news")
        self.button_ai = Button(self.labelf_ai, text="D",
                                command=lambda entry=self.entry_ai: self.choose_file_entry(entry))
        self.button_ai.grid(row=0, column=5, sticky="news")

        #
        self.labelf_graph = LabelFrame(self, text="Graph to visualize learning", labelanchor="n")
        self.labelf_graph.grid(row=3, column=0, columnspan=2, sticky="news")

        self.spinbox_graph = Spinbox(self.labelf_graph, justify="center",
                                     values=("ENABLED", "DISABLED"))
        self.spinbox_graph.pack()

        #
        self.labelf_x = LabelFrame(self, text="How often should it compute a point", labelanchor="n")
        self.labelf_x.grid(row=3, column=2, columnspan=2, sticky="nsew")

        self.spinbox_x = Spinbox(self.labelf_x, justify="center", from_=0, to=60000, increment=1000)
        self.spinbox_x.insert(0, "1000")
        self.spinbox_x.pack()

        #
        self.button_go = Button(self, text="Go train", command=self.go_train)
        self.button_go.grid(row=4, column=0, columnspan=4, sticky="news")

        for cl in range(4):
            self.columnconfigure(cl, weight=4)

    def choose_file_entry(self, entry):

        new_path = filedialog.askopenfilename()
        if new_path:
            entry.delete(first=0, last=len(entry.get()))
            entry.insert(0, new_path)

    def go_train(self):
        """ Download data and create a neural network if needed """

        file_ai = self.entry_ai.get()
        if file_ai:
            ai = load_ai(file_ai)
            ai.epoch += int(self.spinbox_epoch.get())

        else:
            list_shape = self.entry_shape.get().split(",")
            list_shape = [i.replace(" ", "") for i in list_shape]
            shape = tuple(map(int, list_shape))
            activation_function = self.spinbox_af.get().lower()

            regularization_strength = float(self.spinbox_lambda.get())
            learning_rate = float(self.spinbox_lr.get())

            epoch = int(self.spinbox_epoch.get())
            mini_batch = int(self.spinbox_batch.get())

            n_train = int(self.spinbox_ntr.get())
            n_test = int(self.spinbox_nte.get())

            ai = AI(shape, activation_function, regularization_strength,
                    learning_rate, epoch, mini_batch, n_train, n_test)

            if activation_function == "reLU":
                ai.param = random_thetas_reLU(shape)
            else:
                ai.param = random_thetas_sigmoid(shape)

        n_train = int(self.spinbox_ntr.get())
        n_test = int(self.spinbox_nte.get())

        path_train = 'MNIST_data/mnist_train.csv'
        path_test = 'MNIST_data/mnist_test.csv'

        path_train_obj = self.entry_ftr.get()
        path_test_obj = self.entry_fte.get()

        mean = False
        feature = False

        if self.spinbox_data.get() == "mean normalization":
            mean = True

        elif self.spinbox_data.get() == "feature scaling":
            feature = True

        data_train_p, label_train_p = download_data(path_train_obj, n_train, path=path_train, n_total=60000)
        data_test_p, label_test_p = download_data(path_test_obj, n_test, n_total=10000, path=path_test)

        data_train = Dataset(data_train_p, label_train_p,
                             feature_scaling=feature, mean_normalization=mean, one_dimensional=True)
        data_test = Dataset(data_test_p, label_test_p,
                            feature_scaling=feature, mean_normalization=mean, one_dimensional=True)


        Train(self.parent, ai, int(self.spinbox_batch.get()), int(self.spinbox_epoch.get()),
              (data_train, data_test), (self.spinbox_graph.get(), self.spinbox_x.get()),
              bg="white")



class Train(LabelFrame):

    def __init__(self, parent, ai, mini_batch, epoch, database, graph, **kwargs):

        super().__init__(parent, **kwargs)
        self.grid(row=1, column=0, sticky="news")

        self.parent = parent
        self.finish = False

        self.parent.rowconfigure(0, weight=0)
        self.parent.rowconfigure(1, weight=1)

        self.mini_batch = mini_batch
        self.nb_epoch = epoch
        self.configure(text=f"{self.nb_epoch} epoch(s) remaining...", labelanchor="n")

        self.ai = ai
        self.database_train, self.database_test = database

        self.graph, self.how_often = graph

        if self.graph == "ENABLED":
            self.graph, self.how_often = True, int(self.how_often)

            self.f = mFigure()

            self.accuracy = self.f.add_subplot(121)
            self.cost = self.f.add_subplot(122)

            self.plot_x = []

            self.accuracy_plot_tr_y, self.accuracy_plot_te_y = [], []
            self.cost_plot_tr_y, self.cost_plot_te_y = [], []

            self.canvas = mFigureCanvasTkAgg(self.f, self)
            self.canvas.draw()

            self.toolbar = mNavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()

            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


        else:
            self.graph = False

        self.lock = Lock()

        self.training = threading.Thread(target=self._train)
        self.training.start()

        if self.graph:
            self.after(1000, self.draw)

        # Will check every 5 sec if the training is over
        self.after(5000, self.check_over)

    def draw(self):
        """ Update the graph each 10 sec"""

        # Doesn't allow to totally avoid global value problem
        self.look_lock(timeout=3000)

        self.accuracy.clear()
        self.cost.clear()

        self.lock.acquire()

        self.accuracy.plot(self.plot_x, self.accuracy_plot_te_y, color='red', label="Test-set")
        self.accuracy.plot(self.plot_x, self.accuracy_plot_tr_y, color='black', label='Training-set')

        self.cost.plot(self.plot_x, self.cost_plot_te_y, color='red', label="Test-set")
        self.cost.plot(self.plot_x, self.cost_plot_tr_y, color='black', label='Training-set')

        self.accuracy.set_xlabel("number of epoch")
        self.accuracy.set_ylabel("% of succes")

        self.cost.set_title("Cost function")
        self.cost.set_xlabel("number of epoch")
        self.cost.set_ylabel("cost function value")

        self.accuracy.set_title("Accuracy")
        self.accuracy.legend(loc="best")
        self.cost.legend(loc="best")

        self.canvas.draw()
        self.lock.release()

        if not self.finish:
            self.after(5000, self.draw)


    def _train(self):
        """ Compute gradient de   scent, to update the GUI need to be in a thread """

        for epoch in range(self.nb_epoch):
            self["text"] = f"{self.nb_epoch-epoch} epoch(s) remaining..."

            nb_batch_in_one_epoch = int(len(self.database_train) / self.mini_batch)

            for current_batch in range(nb_batch_in_one_epoch):

                from_ = current_batch * self.mini_batch
                to = (current_batch + 1) * self.mini_batch

                mini_train = self.database_train.data[from_:to]
                mini_train_l = self.database_train.label[:, from_:to]

                self.ai.param = gradient_descent(mini_train, mini_train_l, feed_forward, self.ai.activation_function,
                                                 self.ai.derivative_af, self.ai.param,
                                                 self.ai.learning_rate, self.ai.lamb)

                if self.graph and (current_batch * self.mini_batch) % self.how_often == 0:
                    self._compute_point(current_batch, epoch, entire=False, n_test=10000, n_train=10000)

        self.finish = True

    def _compute_point(self, current_batch, current_epoch, entire=True, n_test=5000, n_train=5000):
        """ Compute a point, called"""

        if not entire:
            data_test, label_test = self.only_a_few(self.database_test, n_test)
            data_train, label_train = self.only_a_few(self.database_train, n_train)
        else:
            data_test, label_test = self.database_test.data, self.database_test.label
            data_train, label_train = self.database_train.data, self.database_train.label

        # Compute point ; test set
        result_test, _ = feed_forward(data_test, self.ai.param, self.ai.activation_function)

        c_te = cost_function(self.ai.lamb, self.ai.param, n_test,
                             result_test, label_test)
        a_te = accuracy(result_test, label_test, n_test)

        # Compute point ; training set
        result_train, _ = feed_forward(data_train, self.ai.param, self.ai.activation_function)
        c_tr = cost_function(self.ai.lamb, self.ai.param, n_train,
                             result_train, label_train)
        a_tr = accuracy(result_train, label_train, n_train)

        self.look_lock(timeout=3000)
        self.lock.acquire()

        self.plot_x.append((current_batch*self.mini_batch)/int(len(self.database_train)) + current_epoch)

        self.accuracy_plot_te_y.append(a_te)
        self.cost_plot_te_y.append(c_te)
        self.accuracy_plot_tr_y.append(a_tr)
        self.cost_plot_tr_y.append(c_tr)

        self.lock.release()

    def look_lock(self, timeout):
        """
        Look if the lock is locked or not

        Parameter
        ---------

        timeout : int
            override the condition after this time in MS
            (in case the lock stay on locked)

        Return
        ------
        """
        what_time_is_it = time.time()
        while self.lock.locked():
            what_time_is_it_now = time.time()  # Most annoying machine
            if what_time_is_it-what_time_is_it_now >= timeout:
                break

    def only_a_few(self, database, a_few):
        """
        Take a certain number of data from a Dataset object

        Parameters
        ----------

        database : Dataset obj
            The data
        a_few : int
            how many examples from the data obj

        Returns
        -------
            data : numpy array
                a_few data
            label : numpy array
                a_few data's label
        """
        index = random.sample(list(range(len(database))), k=a_few)

        # Fastest way (np.append is much slower)
        data = np.zeros((a_few, database.data.shape[1]))
        label = np.zeros((database.label.shape[0], a_few))

        for counter, k in enumerate(index):
            data[counter], label[:, counter] = database[k]
        return data, label

    def check_over(self):
        if not self.finish:
            self.after(5, self.check_over)
            return

        final_r, _ = feed_forward(self.database_test.data, self.ai.param, self.ai.activation_function)
        final_a = accuracy(final_r, self.database_test.label, len(self.database_test))
        self.ai.accuracy = final_a

        self["text"] = f"[TRAINING OVER]\n Final accuracy on test set : {round(final_a*100, 2)}% "
        save_button = Button(self.parent, text=f"Save this neural network", command=self.save_ai)
        save_button.grid(row=2, column=0, sticky="news")

    def save_ai(self):
        ai_path = filedialog.asksaveasfilename()
        save_ai(ai_path, self.ai)


if __name__ == "__main__":
    root = Tk()
    root.title("Build your neural network")
    root.geometry("800x600")

    build = Build(root, text="Hyperparameters", labelanchor="n")

    root.columnconfigure(0, weight=1)
    # Will be changed further in class Train
    root.rowconfigure(0, weight=1)

    root.mainloop()
