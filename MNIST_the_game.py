from tkinter import *
import os
from stuff.obj_Dataset import Dataset
from stuff.load_save_ai import load_ai
import numpy as np
from PIL import ImageTk
from stuff.fully_connected import feed_forward
from stuff.download_data import download_data


class Interface(Frame):

    def __init__(self, main_f, **kwargs):

        Frame.__init__(self, main_f, **kwargs)

        self.main_f = main_f

        self.main_f.geometry("450x400")
        self.main_f.title("Choose your opponent")
        self.main_f.config(bg="black")

        self.list_ai = []
        for ai in os.listdir("saved_ai"):
            self.list_ai.append(load_ai(f"saved_ai/{ai}"))

        self.list_ai.sort(key=lambda x: x.accuracy, reverse=True)

        self.nice_label = Label(self.main_f, font=("bold", 12, "bold"), bg="white",
                                text='You have {} neural network(s)'.format(len(self.list_ai)))
        self.nice_label.grid(column=0, row=0, columnspan=2, pady=2, sticky="nsew")

        self.listbox = Listbox(self.main_f, bg="white")
        self.listbox.grid(column=0, row=1, sticky="nsew")

        for i, ai in enumerate(self.list_ai):
            self.listbox.insert(END, ai.name)

        self.listbox.bind('<<ListboxSelect>>', self.information)

        self.labelframe = LabelFrame(self.main_f, text="Selected neural network info", labelanchor="n")
        self.labelframe.grid(row=1, column=1, sticky="nsew", )

        self.label_spinbox = Label(self.main_f, text="Number of round :")
        self.label_spinbox.grid(row=2, column=0, sticky="nsew")

        self.spinbox = Spinbox(self.main_f, from_=1, to=10)
        self.spinbox.delete(0)
        self.spinbox.insert(0, "3")
        self.spinbox.grid(row=2, column=1, sticky="nsew")

        self.button_v = Button(self.main_f, text="Go", bg="chartreuse4", font=("bold", 12), command=self.validate)
        self.button_v.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.main_f.columnconfigure(1, weight=1)
        self.main_f.rowconfigure(1, weight=1)

    def information(self, event):

        listbox = event.widget
        i = listbox.curselection()

        if i != ():
            i = i[0]

            for widget in self.labelframe.winfo_children():
                widget.destroy()

            self.labelframe.config(text=self.list_ai[i].name)

            label_accuracy = Label(self.labelframe, text=f"Accuracy : {self.list_ai[i].accuracy}\n")
            label_accuracy.grid(row=0, sticky="w")

            label_shape = Label(self.labelframe, text=f"Shape : {self.list_ai[i].shape}")
            label_shape.grid(row=1, sticky="w")

            label_af = Label(self.labelframe,
                             text=f"Activation function : {self.list_ai[i].activation_function_name}\n")
            label_af.grid(row=2, sticky="w")

            label_learning = Label(self.labelframe, text=f"Learning rate : {self.list_ai[i].learning_rate}")
            label_learning.grid(row=3, sticky="w")

            label_lambda = Label(self.labelframe, text=f"Regularization strength : {self.list_ai[i].lamb}\n")
            label_lambda.grid(row=4, sticky="w")

            label_epoch = Label(self.labelframe, text=f"Number of epochs: {self.list_ai[i].epoch}")
            label_epoch.grid(row=5, sticky="w")

            label_mini_batch = Label(self.labelframe, text=f"Mini-Batch size : {self.list_ai[i].mini_batch}")
            label_mini_batch.grid(row=6, sticky="w")

            label_n_train = Label(self.labelframe, text=f"\nNumber of training examples : {self.list_ai[i].n_train}")
            label_n_train.grid(row=7, sticky="w")

            label_n_train = Label(self.labelframe, text=f"Number of test examples : {self.list_ai[i].n_test}")
            label_n_train.grid(row=8, sticky="w")

    def validate(self):

        i = self.listbox.curselection()

        try:
            i = i[0]
            ai_choice = self.list_ai[i]
            nb_round = int(self.spinbox.get())
        except IndexError:
            print('Pls select a neural network')
        except ValueError:
            print('Pls select a NUMBER of round')
        else:
            TheGame(ai_choice, nb_round)


class TheGame(Toplevel):

    def __init__(self, ai, max_round):

        self.ai = ai
        self.max_round = max_round
        self.round = 1

        self.score = [0, 0]
        self.correct_guess = [0, 0]

        Toplevel.__init__(self)

        path_pickle_obj = "MNIST_data/mnist_test"
        path_MNIST = 'MNIST_data/mnist_test.csv'

        data, label = download_data(path_pickle_obj, 10000, path_MNIST, 10000)
        self.dataset = Dataset(data, label)

        self.geometry("225x300")
        self.title("YOU vs " + ai.name.upper())
        self.resizable(width=False, height=False)

        # Round score
        self.labelframe = LabelFrame(self, text=f"ROUND 1/{self.max_round}", labelanchor="n")
        self.labelframe.grid(row=0, column=1, sticky="nsew")

        self.label_j = Label(self.labelframe)
        self.round_score_j = StringVar(self.label_j, value="   ")
        self.label_j.config(textvariable=self.round_score_j)
        self.label_j.grid(row=0)

        self.label_ai = Label(self.labelframe)
        self.round_score_ai = StringVar(self.label_ai, value="   ")
        self.label_ai.config(textvariable=self.round_score_ai)
        self.label_ai.grid(row=1)

        # All score
        self.labelframe_score = LabelFrame(self, text='SCORE', labelanchor="n")
        self.labelframe_score.grid(row=0, sticky="nsew")

        self.label_j_s = Label(self.labelframe_score)
        self.total_score_j = StringVar(self.label_j_s, value="You           : 0")
        self.label_j_s.config(textvariable=self.total_score_j)
        self.label_j_s.grid(row=0, sticky="w")

        self.label_ai_s = Label(self.labelframe_score)
        self.total_score_ai = StringVar(self.label_ai_s, value="Opponent : 0")
        self.label_ai_s.config(textvariable=self.total_score_ai)
        self.label_ai_s.grid(row=1, sticky="w")

        self.canvas = Canvas(self, bg="black")
        self.canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.entry = Entry(self)
        self.entry.grid(row=2, column=0, columnspan=2)

        self.entry.bind('<Return>', self.entry_event)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.new_image()

    def entry_event(self, event):

        entry = event.widget
        choice = entry.get()

        image_numpy = self.dataset.data[self.i]
        image_numpy_f = np.array([image_numpy])  # Convert into the correct format

        image_numpy_f = self.dataset.one_dimensional(image_numpy_f)
        image_numpy_f = self.dataset.mean_normalization(image_numpy_f)

        result_ai, _ = feed_forward(image_numpy_f, self.ai.param, self.ai.activation_function)
        digit_displayed = np.argmax(self.dataset.label[:, self.i])

        # Player
        actual_round_score_j = self.round_score_j.get()
        if choice == str(digit_displayed):
            self.round_score_j.set(actual_round_score_j + "O")
            self.correct_guess[0] += 1
        else:
            self.round_score_j.set(actual_round_score_j + "X")

        # Not the player
        guess_digit = np.argmax(result_ai)
        actual_round_score_ai = self.round_score_ai.get()
        if guess_digit == digit_displayed:
            self.round_score_ai.set(actual_round_score_ai + "O")
            self.correct_guess[1] += 1
        else:
            self.round_score_ai.set(actual_round_score_ai + "X")

        self.check_round()

        if self.canvas.winfo_exists():
            self.new_image(event)

    def check_round(self):

        if len(self.round_score_j.get()) < 10 + 3:
            return

        self.label_score_update()
        if self.round >= self.max_round:
            self.end_game()
            return

        self.round += 1
        self.label_round_update()
        self.clean_label_score()

    def label_score_update(self):

        score_ai = 0
        for result in self.round_score_ai.get():
            if result == "O":
                score_ai += 1

        score_j = 0
        for result in self.round_score_j.get():
            if result == "O":
                score_j += 1

        if score_j >= score_ai:
            self.score[0] += 1
            self.total_score_j.set(f'You           : {str(self.score[0])}')

        if score_ai >= score_j:
            self.score[1] += 1
            self.total_score_ai.set(f'Opponent : {str(self.score[1])}')

    def label_round_update(self):
        self.labelframe.config(text=f"ROUND {self.round}/{self.max_round}")

    def clean_label_score(self):
        self.round_score_j.set("   ")
        self.round_score_ai.set("   ")

    def new_image(self, event=None):

        self.entry.delete(first=0, last=len(self.entry.get()))
        imagePIL, label, self.i = self.dataset.random_image()

        imagePIL = imagePIL.resize((112, 112))
        self._imageTK = ImageTk.PhotoImage(imagePIL)
        self.canvas.delete("all")
        self.canvas.create_image((int(225 / 2), int(325 / 2) - 70), image=self._imageTK)

    def end_game(self):

        self.canvas.destroy()
        self.entry.destroy()
        button_quit = Button(self, text="QUIT GAME", command=self.destroy)
        button_quit.grid(row=2, column=0, columnspan=2, sticky="news")

        j_score = self.correct_guess[0]
        ai_score = self.correct_guess[1]
        if ai_score < j_score:
            text_winner = "YOU"
        elif ai_score > j_score:
            text_winner = self.ai.name
        else:
            text_winner = "NOBODY"

        end_game_f = LabelFrame(self, text=text_winner.upper() + " WON", font=("bold", "11", "bold"),
                                foreground="gold", labelanchor="n", bg="black")
        end_game_f.grid(row=1, column=0, columnspan=2, sticky="news")

        accuracy_ai = round(self.correct_guess[1] / (self.max_round * 10), 2)
        label_accuracy_ai = Label(end_game_f, foreground="white", bg="black",
                                  text=f"\nOpponent accuracy : {str(accuracy_ai)}"
                                       f" ({str(self.correct_guess[1])}/{str(self.max_round * 10)})")
        label_accuracy_ai.grid(row=0, sticky="w")

        accuracy_j = round(self.correct_guess[0] / (self.max_round * 10), 2)
        label_accuracy_j = Label(end_game_f, foreground="white", bg="black",
                                 text=f"\nYour accuracy : {str(accuracy_j)}"
                                      f" ({str(self.correct_guess[0])}/{str(self.max_round * 10)})")
        label_accuracy_j.grid(row=1, sticky="w")


if __name__ == '__main__':
    root = Tk()
    main_frame = Interface(root, bg="black")
    root.mainloop()
