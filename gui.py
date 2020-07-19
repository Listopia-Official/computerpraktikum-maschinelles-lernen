import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import visual
import dataset
import numpy as np

class Gui:

    def __init__(self):

        # UI components
        self.frame = Tk()
        self.frame.title("CP_ML-Classificator")

        self.data_folder_label = Label(self.frame, text="Data directory:")
        self.data_folder_textfield = Entry(self.frame, width = 100, state="readonly")
        self.data_folder_button = Button(self.frame,
                        text="Select data directory",
                        command=self.select_data)

        self.dataset_label = Label(self.frame, text="Datasets:")
        self.dataset_combobox = ttk.Combobox(self.frame, state="readonly")

        self.k_label = Label(self.frame, text="K:")
        self.k_slider = Scale(self.frame, from_=0, to=200, resolution = 1,length = 600, orient=HORIZONTAL, tickinterval = 20)

        self.l_label = Label(self.frame, text="l:")
        self.l_slider = Scale(self.frame, from_=2, to=20, resolution=1, length=600, orient=HORIZONTAL, tickinterval=2)

        self.train_button = Button(self.frame, text = "Train", command = self.train, width = 110)

        #self.plot = FigureCanvasTkAgg(visual.display_2d_dataset(dataset.parse('data/bananas-2-2d.test.csv'), "ere"), master=self.frame)

        self.data_folder_textfield.configure(state=tk.NORMAL)
        self.data_folder_textfield.insert(0, os.path.abspath("./data")) # Default data dir
        self.data_folder_textfield.configure(state="readonly")

        self.populate_datasets()

        self.k_slider.set(50)

        self.l_slider.set(5)

        self.data_folder_label.grid(column=0,row=0, padx = 5, pady = 2)
        self.data_folder_textfield.grid(column=0, row=1, padx = 10, pady = 2)
        self.data_folder_button.grid(column=1, row=1, padx = 10, pady = 2)

        self.dataset_label.grid(column = 0, row = 2, padx = 10, pady = 2)
        self.dataset_combobox.grid(column = 0, row = 3, padx = 10, pady = 2)

        self.k_label.grid(column=0, row = 4, padx = 5, pady = 2)
        self.k_slider.grid(column = 0, row = 5, padx = 10, pady = 2)

        self.l_label.grid(column=0, row=6, padx=5, pady=2)
        self.l_slider.grid(column=0, row=7, padx=10, pady=2)

        #self.plot._tkcanvas.grid(column=0, row=7, padx=10, pady=2)

        self.train_button.grid(column = 0,columnspan = 2, row = 8, padx = 10, pady = 2)

    def show(self):
        self.frame.mainloop()


    def select_data(self):
        choosen_dir = filedialog.askdirectory()

        if choosen_dir != "":
            self.data_folder_textfield.configure(state=tk.NORMAL)
            self.data_folder_textfield.delete(0, tk.END)
            self.data_folder_textfield.insert(0, choosen_dir)
            self.data_folder_textfield.configure(state="readonly")

        self.populate_datasets()

    def validate_data_dir(self, widget, mode, validator):


        return False

    def populate_datasets(self):
        data_dir = self.data_folder_textfield.get()

        data_values = []

        for file in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, file)) and file.endswith(".csv"):
                data_values.append(os.path.splitext(file)[0].split(".")[0])

        self.dataset_combobox['values'] = np.unique(np.array(data_values)).tolist()

        if len(data_values) > 0:
            self.dataset_combobox.set(data_values[0])
        else:
            self.dataset_combobox.set("")


    def train(self):
        if len(self.dataset_combobox['values']) == 0:
            messagebox.showerror("Error:", "No dataset was selected.")
