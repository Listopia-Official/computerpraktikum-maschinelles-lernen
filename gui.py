import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import matplotlib as plt
import collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import visual
import dataset
import numpy as np
import sys
import time


class Gui:

    def __init__(self, classify_function):

        self.classify_function = classify_function

        # UI components
        self.frame = Tk()
        self.frame.title("CP_ML-Classificator")
        self.frame.protocol('WM_DELETE_WINDOW', self.close)

        self.data_folder_label = Label(self.frame, text="Data directory:")
        self.data_folder_textfield = Entry(self.frame, width = 100, state="readonly")
        self.data_folder_button = Button(self.frame,
                        text="Select data directory",
                        command=self.select_data)

        self.dataset_label = Label(self.frame, text="Datasets:")
        self.dataset_combobox = ttk.Combobox(self.frame, state="readonly")

        self.k_label = Label(self.frame, text="Max k:")
        self.k_slider = Scale(self.frame, from_=1, to=200, resolution = 1,length = 600, orient=HORIZONTAL, tickinterval = 20)

        self.l_label = Label(self.frame, text="Partition count (l):")
        self.l_slider = Scale(self.frame, from_=2, to=20, resolution=1, length=600, orient=HORIZONTAL, tickinterval=2)

        self.kd_tree_checkbox_var = BooleanVar()
        self.kd_tree_checkbox = Checkbutton(self.frame, var=self.kd_tree_checkbox_var, text="Use KD-Tree-Search (otherwise brute-searching)")

        self.train_button = Button(self.frame, text = "Classify selected", command = self.train, width = 50)
        self.train_all_button = Button(self.frame, text="Classify all (print on console)", command=self.train_all, width=50)

        self.train_data_label = Label(self.frame, text="Training data:")
        self.train_data_zoom_button = Button(self.frame, text="Detailed view", command = self.display_train_data)

        self.test_data_label = Label(self.frame, text="Test data:")
        self.test_data_zoom_button = Button(self.frame, text="Detailed view", command=self.display_test_data)

        self.result_data_label = Label(self.frame, text="Result data:")
        self.result_data_zoom_button = Button(self.frame, text="Detailed view", command=self.display_result_data)

        self.data_folder_textfield.configure(state=tk.NORMAL)
        self.data_folder_textfield.insert(0, os.path.abspath("./data")) # Default data dir
        self.data_folder_textfield.configure(state="readonly")

        self.populate_datasets()

        self.k_slider.set(50)

        self.l_slider.set(5)

        self.data_folder_label.grid(column=0,row=0, padx = 5, pady = 2, sticky="W")
        self.data_folder_textfield.grid(column=0, columnspan=3, row=1, padx = 10, pady = 2, sticky = "WE")
        self.data_folder_button.grid(column=3, row=1, padx = 10, pady = 2)

        self.dataset_label.grid(column = 0, row = 2, padx = 10, pady = 2, sticky="W")
        self.dataset_combobox.grid(column = 0, row = 3, columnspan=3, padx = 10, pady = 2, sticky="WE")

        self.k_label.grid(column=0, row = 4, padx = 5, pady = 2, sticky="W")
        self.k_slider.grid(column = 0,columnspan=3, row = 5, padx = 10, pady = 2, sticky="WE")

        self.l_label.grid(column=0, row=6, padx=5, pady=2, sticky="W")
        self.l_slider.grid(column=0,columnspan=3, row=7, padx=10, pady=2, sticky="WE")

        self.kd_tree_checkbox.grid(column = 0, row = 8, padx = 10, pady = 2, sticky = "W")

        self.train_button.grid(column = 0,columnspan = 3, row = 9, padx = 10, pady = 2, sticky="WE")
        self.train_all_button.grid(column=0, columnspan=3, row=10, padx=10, pady=2, sticky="WE")

        self.train_data_label.grid(column = 0, row = 11, padx = 10, pady = 2)
        self.train_data_zoom_button.grid(column=0, row=12, padx=10, pady=2)

        self.test_data_label.grid(column=1, row=11, padx=10, pady=2)
        self.test_data_zoom_button.grid(column=1, row=12, padx=10, pady=2)

        self.result_data_label.grid(column=2, row=11, padx=10, pady=2)
        self.result_data_zoom_button.grid(column=2, row=12, padx=10, pady=2)

        self.train_data = None
        self.test_data = None
        self.result_data = None

    def close(self):
        self.frame.destroy()
        sys.exit()

    def show(self):
        self.frame.mainloop()

    def select_data(self):
        chosen_dir = filedialog.askdirectory()

        if chosen_dir != "":
            self.data_folder_textfield.configure(state=tk.NORMAL)
            self.data_folder_textfield.delete(0, tk.END)
            self.data_folder_textfield.insert(0, chosen_dir)
            self.data_folder_textfield.configure(state="readonly")

        self.populate_datasets()

    def validate_data_dir(self, widget, mode, validator):

        return False

    def populate_datasets(self):
        data_dir = self.data_folder_textfield.get()

        data_values = []

        for file in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, file)) and file.endswith(".csv"):
                data_values.append(os.path.splitext(file)[0].rsplit(".", 1)[0])

        self.dataset_combobox['values'] = np.unique(np.array(data_values)).tolist()

        if len(data_values) > 0:
            self.dataset_combobox.set(data_values[0])
        else:
            self.dataset_combobox.set("")

    def train(self):
        if len(self.dataset_combobox['values']) == 0:
            messagebox.showerror("Error:", "No dataset was selected.")
            return

        dataset_name = self.dataset_combobox.get()
        data_dir = self.data_folder_textfield.get()

        self.train_data = dataset.parse(data_dir + "/" + dataset_name + ".train.csv")
        self.test_data = dataset.parse(data_dir + "/" + dataset_name + ".test.csv")

        output_path = data_dir + "/" + dataset_name + ".result.csv"

        self.train_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.train_data, "Training data:", micro = True), master=self.frame)
        self.test_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.test_data, "Test data:", micro = True),
                                                 master=self.frame)

        self.train_data_plot._tkcanvas.grid(column = 0, row = 13, padx = 10, pady = 4)
        self.test_data_plot._tkcanvas.grid(column=1, row=13, padx=10, pady=4)

        start_time = time.time()

        k_best, f_rate, self.result_data = self.classify_function(self.train_data, self.test_data, output_path,
                                                                  kset=np.arange(self.k_slider.get()),
                                                                  l=self.l_slider.get(), algorithm='k-d_tree')

        end_time = time.time() - start_time

        self.result_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.result_data, "Result data:", micro = True),
                                                  master=self.frame)

        self.result_data_plot._tkcanvas.grid(column=2, row=13, padx=10, pady=4)

        self.data_label = Message(self.frame, anchor = "w",text="Time: {:.4f}s \nFailure rate: {:.4f}\n k*: {}".format(end_time, f_rate, k_best), width = 125)

        self.data_label.grid(column = 3, row = 13, padx = 10, pady = 4, sticky="NW")

        messagebox.showinfo("Information:", "The simulation was done in and the results were saved at " + output_path)

    def train_all(self):
        if len(self.dataset_combobox['values']) == 0:
            messagebox.showerror("Error:", "No data are there to be used.")
            return

        data_dir = self.data_folder_textfield.get()

        kset_val = np.arange(self.k_slider.get())
        l_val = self.l_slider.get()
        bs_val = not self.kd_tree_checkbox_var.get()

        for dataset_name in self.dataset_combobox['values']:
            print('## Running dataset', dataset_name, ' ##')

            train_data = dataset.parse(data_dir + "/" + dataset_name + ".train.csv")
            test_data = dataset.parse(data_dir + "/" + dataset_name + ".test.csv")

            output_path = data_dir + "/" + dataset_name + ".result.csv"

            start_time = time.time()

            k_best, f_rate, self.result_data = self.classify_function(train_data, test_data, output_path,
                                                                      kset=kset_val,
                                                                      l=l_val,
                                                                      brute_sort=bs_val)

            end_time = time.time() - start_time


            print('Elapsed time:', end_time)

            print("## -------- ##\n")

    def display_train_data(self):
        self.display_data(self.train_data, "Detailed train data view:")

    def display_test_data(self):
        self.display_data(self.test_data, "Detailed test data view:")

    def display_result_data(self):
        self.display_data(self.result_data, "Detailed result data view:")

    def display_data(self, data, title):
        if data is None:
            messagebox.showerror("Error:", "No data to display.")
            return

        fig = visual.display_2d_dataset(data, title)
        fig.show()
