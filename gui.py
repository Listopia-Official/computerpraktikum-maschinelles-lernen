import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import visual
import dataset
import numpy as np
import sys
import time

"""
The GUI module, which wraps the code in main.py to a nice-to-use interface.
"""

"""
A class modeling our main application window.
"""
class Gui:

    # Classify function reference is passed so we don't have circular dependencies
    def __init__(self, classify_function, grid_function):

        self.classify_function = classify_function
        self.grid_function = grid_function

        # Configure main frame
        self.frame = Tk()
        self.frame.title("CP_ML-Classificator")
        self.frame.protocol('WM_DELETE_WINDOW', self.close) # Close-listener

        # UI components

        self.data_folder_label = Label(self.frame, text="Data directory:")
        self.data_folder_textfield = Entry(self.frame, width = 100, state="readonly") # Textfield isn't editable
        self.data_folder_button = Button(self.frame,
                        text="Select data directory",
                        command=self.select_data)

        self.dataset_label = Label(self.frame, text="Datasets:")
        self.dataset_combobox = ttk.Combobox(self.frame, state="readonly") # Also not editable

        self.k_label = Label(self.frame, text="Max k:")
        self.k_slider = Scale(self.frame, from_=1, to=200, resolution = 1,length = 600, orient=HORIZONTAL, tickinterval = 20)

        self.l_label = Label(self.frame, text="Partition count (l):")
        self.l_slider = Scale(self.frame, from_=2, to=20, resolution=1, length=600, orient=HORIZONTAL, tickinterval=2)

        self.algorithm_label = Label(self.frame, text="Algorithm:")
        self.algorithm_combobox = ttk.Combobox(self.frame, values = ["brute_sort", "k-d_tree", "sklearn"], text="Search algorithm:", state="readonly")

        self.grid_checkbox_var = BooleanVar()
        self.grid_checkbox = Checkbutton(self.frame, text="Display grid (instead of the test dataset)", var = self.grid_checkbox_var)

        self.train_button = Button(self.frame, text = "Classify selected", command = self.train, width = 50)
        self.train_all_button = Button(self.frame, text="Classify all (print on console)", command=self.train_all, width=50)

        self.train_data_label = Label(self.frame, text="Training data:")
        self.train_data_zoom_button = Button(self.frame, text="Detailed view", command = self.display_train_data)

        self.test_data_label = Label(self.frame, text="Test data:")
        self.test_data_zoom_button = Button(self.frame, text="Detailed view", command=self.display_test_data)

        self.result_data_label = Label(self.frame, text="Result data:")
        self.result_data_zoom_button = Button(self.frame, text="Detailed view", command=self.display_result_data)

        self.data_folder_textfield.configure(state=tk.NORMAL) # Make textfield temporarily writeable
        self.data_folder_textfield.insert(0, os.path.abspath("./data")) # Default data dir
        self.data_folder_textfield.configure(state="readonly") # Make textfield read-only again

        self.populate_datasets() # Scan the default directory for dataset files

        self.algorithm_combobox.set("brute_sort")

        self.k_slider.set(50)

        self.l_slider.set(5)

        # Place the components in the layout grid
        self.data_folder_label.grid(column=0,row=0, padx = 5, pady = 2, sticky="W")
        self.data_folder_textfield.grid(column=0, columnspan=3, row=1, padx = 10, pady = 2, sticky = "WE")
        self.data_folder_button.grid(column=3, row=1, padx = 10, pady = 2)

        self.dataset_label.grid(column = 0, row = 2, padx = 10, pady = 2, sticky="W")
        self.dataset_combobox.grid(column = 0, row = 3, columnspan=3, padx = 10, pady = 2, sticky="WE")

        self.k_label.grid(column=0, row = 4, padx = 5, pady = 2, sticky="W")
        self.k_slider.grid(column = 0,columnspan=3, row = 5, padx = 10, pady = 2, sticky="WE")

        self.l_label.grid(column=0, row=6, padx=5, pady=2, sticky="W")
        self.l_slider.grid(column=0,columnspan=3, row=7, padx=10, pady=2, sticky="WE")

        self.algorithm_label.grid(column=0, row=8, padx=5, pady=2, sticky="W")
        self.algorithm_combobox.grid(column = 0, row = 9, padx = 10, pady = 2, sticky = "W")

        self.grid_checkbox.grid(column=0, row=10, padx=5, pady=2, sticky="W")

        self.train_button.grid(column = 0,columnspan = 3, row = 11, padx = 10, pady = 2, sticky="WE")
        self.train_all_button.grid(column=0, columnspan=3, row=12, padx=10, pady=2, sticky="WE")

        self.train_data_label.grid(column = 0, row = 13, padx = 10, pady = 2)
        self.train_data_zoom_button.grid(column=0, row=14, padx=10, pady=2)

        self.test_data_label.grid(column=1, row=13, padx=10, pady=2)
        self.test_data_zoom_button.grid(column=1, row=14, padx=10, pady=2)

        self.result_data_label.grid(column=2, row=13, padx=10, pady=2)
        self.result_data_zoom_button.grid(column=2, row=14, padx=10, pady=2)

        # By default those are None
        self.train_data = None
        self.test_data = None
        self.result_data = None

    """
    Listener invoked of the Frame is closed, makes sure that the application completely terminates.
    """
    def close(self):
        self.frame.destroy()
        sys.exit()

    """
    Invoked if the window should be displayed - this blocks until the window is closed!
    """
    def show(self):
        self.frame.mainloop()

    """
    Invoked if the user presses the "Select data dir" button.
    """
    def select_data(self):
        chosen_dir = filedialog.askdirectory() # Open file chooser

        # Set new dir
        if chosen_dir != "":
            self.data_folder_textfield.configure(state=tk.NORMAL)
            self.data_folder_textfield.delete(0, tk.END)
            self.data_folder_textfield.insert(0, chosen_dir)
            self.data_folder_textfield.configure(state="readonly")

        self.populate_datasets() # Update dataset list

    """
    Reads the dataset files from the specified data dir and adds them to the combobox.
    """
    def populate_datasets(self):
        data_dir = self.data_folder_textfield.get()

        data_values = []

        for file in os.listdir(data_dir): # Iterate files in data dir

            # We're only interested in .csv files
            if os.path.isfile(os.path.join(data_dir, file)) and file.endswith(".csv"):
                data_values.append(os.path.splitext(file)[0].rsplit(".", 1)[0]) # Remove the .csv, .train and .test parts

        # Set the combobox entries to the computed data
        self.dataset_combobox['values'] = np.unique(np.array(data_values)).tolist()

        # If we have found datasets, select the first entry in the list, otherwise none
        if len(data_values) > 0:
            self.dataset_combobox.set(data_values[0])
        else:
            self.dataset_combobox.set("")

    # Invokes the classification algorithm for one selected dataset
    def train(self):

        # Only execute if there are datasets
        if len(self.dataset_combobox['values']) == 0:
            messagebox.showerror("Error:", "No dataset was selected.")
            return

        # Extract some information from the GUI and compute the dataset file names (assuming they follow the canonical name scheme)
        dataset_name = self.dataset_combobox.get()
        data_dir = self.data_folder_textfield.get()
        algo_val = self.algorithm_combobox.get()
        display_grid = self.grid_checkbox_var.get()

        # Grid is only supported for brute_sort and 2D datasets
        if display_grid and algo_val != "brute_sort":
            messagebox.showerror("Error:", "The grid is only supported for brutesort!")
            return

        if display_grid and not dataset_name.endswith("2d"):
            messagebox.showerror("Error:", "The grid is only supported for 2D datasets.")
            return

        self.train_data = dataset.parse(data_dir + "/" + dataset_name + ".train.csv")
        self.test_data = dataset.parse(data_dir + "/" + dataset_name + ".test.csv")

        # The result data will be stored here
        output_path = data_dir + "/" + dataset_name + ".result.csv"

        # Plot the training and test data with matplotlib and embedd them into the window
        self.train_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.train_data, "Training data:", micro = True), master=self.frame)

        # Display training data of grid is disabled
        if not display_grid:
            self.test_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.test_data, "Test data:", micro = True),
                                                 master=self.frame)

        # Position the components
        self.train_data_plot._tkcanvas.grid(column = 0, row = 15, padx = 10, pady = 4)

        start_time = time.time()

        # Actually run the algorithm with the parameters from the GUI
        if algo_val != "sklearn": # Take grid into consideration
            k_best, f_rate, self.result_data, dd = self.classify_function(self.train_data, self.test_data, output_path,
                                                                  kset=np.arange(1, self.k_slider.get() + 1),
                                                                  l=self.l_slider.get(), algorithm=algo_val)
            end_time = time.time() - start_time  # The time the algorithm did take - don't measure the grid time

            if algo_val == "brute_sort" and display_grid: # If displayed, plot it into the test data plot
                grid = self.grid_function(dd, k_best, 100) # Hardcoded grid-size of 100

                self.test_data_plot = FigureCanvasTkAgg(
                    visual.display_2d_dataset(grid, "Grid:", micro=True),
                    master=self.frame)
                self.test_data = grid # Set test data to grid

        else: # Else plot as normal
            k_best, f_rate, self.result_data = self.classify_function(self.train_data, self.test_data, output_path,
                                                                          kset=np.arange(1, self.k_slider.get() + 1),
                                                                          l=self.l_slider.get(), algorithm=algo_val)
            end_time = time.time() - start_time  # The time the algorithm did take

        self.test_data_plot._tkcanvas.grid(column=1, row=15, padx=10, pady=4)

        print("Elapsed time:", end_time,"s")

        # Plot the result data
        self.result_data_plot = FigureCanvasTkAgg(visual.display_2d_dataset(self.result_data, "Result data:", micro = True),
                                                  master=self.frame)

        self.result_data_plot._tkcanvas.grid(column=2, row=15, padx=10, pady=4)


        # Plot some stats about the current run
        self.data_label = Message(self.frame, anchor = "w",text="Time: {:.4f}s \nFailure rate: {:.4f}\n k*: {}".format(end_time, f_rate, k_best), width = 125)

        self.data_label.grid(column = 3, row = 15, padx = 10, pady = 4, sticky="NW")

        # Inform the user that the plot was finished
        messagebox.showinfo("Information:", "The classification was done and the results were saved at " + output_path+".")

    # Invokes the algorithm for all datasets, and prints the results on the console
    def train_all(self):

        # Only execute if we have data
        if len(self.dataset_combobox['values']) == 0:
            messagebox.showerror("Error:", "No data are there to be used.")
            return

        # Get some params from the GUI
        data_dir = self.data_folder_textfield.get()

        kset_val = np.arange(self.k_slider.get())
        l_val = self.l_slider.get()
        algorithm_val = self.algorithm_combobox.get()

        for dataset_name in self.dataset_combobox['values']:
            print('## Running dataset', dataset_name, ' ##')

            train_data = dataset.parse(data_dir + "/" + dataset_name + ".train.csv")
            test_data = dataset.parse(data_dir + "/" + dataset_name + ".test.csv")

            output_path = data_dir + "/" + dataset_name + ".result.csv"

            start_time = time.time()

            # Execute per dataset
            if algorithm_val == "sklearn":
                k_best, f_rate, self.result_data = self.classify_function(train_data, test_data, output_path,
                                                                              kset=kset_val,
                                                                              l=l_val,
                                                                              algorithm=algorithm_val)
            else:
                k_best, f_rate, self.result_data, dd = self.classify_function(train_data, test_data, output_path,
                                                                      kset=kset_val,
                                                                      l=l_val,
                                                                      algorithm = algorithm_val)

            end_time = time.time() - start_time

            # Other data will be print to the console via the print(...) statements in the algorithm impl


            print('Elapsed time:', end_time) # Print the elapsed time

            print("## -------- ##\n")

        # Inform the user that the plot was finished
        messagebox.showinfo("Information:",
                                "The classifications were done and the results were saved at the canonical locations.")

    # Displays the train data plot in the matplotlib window, allowing zoom, saving the image and closer inspection
    def display_train_data(self):
        self.display_data(self.train_data, "Detailed train data view:")

    def display_test_data(self):
        self.display_data(self.test_data, "Detailed test data view:")

    def display_result_data(self):
        self.display_data(self.result_data, "Detailed result data view:")

    # Helper function for the display_... functions
    def display_data(self, data, title):
        # Only display data if they're there
        if data is None:
            messagebox.showerror("Error:", "No data to display.")
            return

        fig = visual.display_2d_dataset(data, title) # Use the large plot
        fig.show() # Show via matplotlib window
