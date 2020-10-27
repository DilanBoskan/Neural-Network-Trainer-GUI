# GUI modules
import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.font
import tkinter.ttk as ttk
# Neural Network
import FeedforwardNeuralNetwork as fnn
import numpy as np
# Plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# File Reading
import xlrd
# File Writing
import xlsxwriter
# System Modules
import sys
import os
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_path)

# -Global Variables-
neural_network: fnn.Feedforward_Neural_Network = None
figure = plt.Figure(figsize=(6, 5), dpi=100)
axis = figure.add_subplot(111)
line, = axis.plot([1, 2], [0, 0])


class MainWindow(tk.Tk):
    # --Constants--
    # None

    def __init__(self):
        # Run the __init__ method on the tk.Tk class
        super().__init__()
        window_width = 1920 * 0.55
        window_height = 1080 * 0.43

        # --Window Settings--
        self.title('Neural Network Trainer/Tester')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=int(window_width),
            height=int(window_height),
            xpad=int(self.winfo_screenwidth()/2 -
                     window_width/2),
            ypad=int(self.winfo_screenheight()/2 -
                     window_height/2)))
        self.configure(bg='#FFFFFF')  # Set background color
        self.update()

        # --Variables--
        self.currentModel_var = tk.StringVar(value='N/A')
        self.inputs = [[]]
        self.outputs = [[]]

        # --Widgets--
        self.create_widgets()
        self.configure_widgets()
        self.place_widgets()

    # -Widget Methods-
    def create_widgets(self):
        """Create window widgets"""
        font = tk.font.Font(size=12)
        self.currentModel_Label = tk.Label(self,
                                           textvariable=self.currentModel_var,
                                           font=font)

        self.training_Frame = tk.Frame(self,
                                       bg='#FFF',
                                       highlightthickness=2,
                                       highlightcolor='#EEE',
                                       highlightbackground='#EEE')
        self.fill_training_Frame()
        self.graph_Frame = tk.Frame(self,
                                    bg='#FFF',)
        self.fill_graph_Frame()

        self.bottom_Frame = tk.Frame(self,
                                     bg='#EEE')
        self.fill_bottom_Frame()

    def configure_widgets(self):
        """Change widget styling and appearance"""

    def place_widgets(self):
        """Place main widgets"""
        self.currentModel_Label.place(x=0, y=0, width=0, height=40,
                                      relx=0, rely=0, relwidth=1, relheight=0)

        self.training_Frame.place(x=20, y=60, width=-20, height=-60*2,
                                  relx=0, rely=0, relwidth=0.3, relheight=1)
        self.graph_Frame.place(x=20, y=60, width=-20*2, height=-60*2,
                               relx=0.3, rely=0, relwidth=0.7, relheight=1)

        self.bottom_Frame.place(x=0, y=-40, width=0, height=40,
                                relx=0, rely=1, relwidth=1, relheight=0)

    def fill_training_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
        # Title
        font = tk.font.Font(size=11)
        self.trainingTitle_Label = tk.Label(self.training_Frame,
                                            text='Train/Test Neural Network',
                                            bg='#EEE', font=font)
        # Inputs
        self.trainingDataInput_Button = ttk.Button(self.training_Frame,
                                                   text='Load Inputs',
                                                   command=self.open_inputFile_filedialog)
        self.trainingDataInput_Label = tk.Label(self.training_Frame,
                                                text='',
                                                bg='#FBFBFB')
        # Outputs
        self.trainingDataOutput_Button = ttk.Button(self.training_Frame,
                                                    text='Load Training Outputs',
                                                    command=self.open_outputFile_filedialog)
        self.trainingDataOutput_Label = tk.Label(self.training_Frame,
                                                 text='',
                                                 bg='#FBFBFB')
        # Epochs
        self.trainingDataEpochs_Label = tk.Label(self.training_Frame,
                                                 text='Epochs',
                                                 anchor=tk.E,
                                                 bg='#FFF')
        self.trainingDataEpochs_Combobox = ttk.Spinbox(self.training_Frame,
                                                       from_=1, increment=1, to=9999999,)
        self.trainingDataEpochs_Combobox.set(1000)
        # Learning Rate
        self.trainingDataLearningRate_Label = tk.Label(self.training_Frame,
                                                       text='Learning Rate',
                                                       anchor=tk.E,
                                                       bg='#FFF')
        self.trainingDataLearningRate_Combobox = ttk.Spinbox(self.training_Frame,
                                                             from_=0.01, increment=0.05, to=100000,)
        self.trainingDataLearningRate_Combobox.set(0.1)
        # Seperator
        self.seperator = ttk.Separator(self.training_Frame,
                                       orient='horizontal')
        # Buttons
        self.trainData_Button = ttk.Button(self.training_Frame,
                                           text='Train Neural Network',
                                           command=self.clicked_trainNN)
        self.testData_Button = ttk.Button(self.training_Frame,
                                          text='Test Neural Network',
                                          command=self.clicked_testNN)

        # -Place Widgets-
        # Title
        self.trainingTitle_Label.place(x=0, y=0, width=0, height=35,
                                       relx=0, rely=0, relwidth=1, relheight=0)
        # Inputs
        self.trainingDataInput_Button.place(x=10, y=35 + 10, width=135, height=33,
                                            relx=0, rely=0, relwidth=0, relheight=0)
        self.trainingDataInput_Label.place(x=10 + 135 + 5, y=35 + 10, width=- (10 + 135 + 5*2), height=33,
                                           relx=0, rely=0, relwidth=1, relheight=0)
        # Outputs
        self.trainingDataOutput_Button.place(x=10, y=35 + 33*1 + 10*2, width=135, height=33,
                                             relx=0, rely=0, relwidth=0, relheight=0)
        self.trainingDataOutput_Label.place(x=10 + 135 + 5, y=35 + 33*1 + 10*2, width=- (10 + 135 + 5*2), height=33,
                                            relx=0, rely=0, relwidth=1, relheight=0)
        # Epochs
        self.trainingDataEpochs_Label.place(x=10, y=35 + 10*3 + 33*2, width=125, height=25,
                                            relx=0, rely=0, relwidth=0, relheight=0)
        self.trainingDataEpochs_Combobox.place(x=10 + 135 + 5, y=35 + 10*3 + 33*2 + 5/2, width=- (10 + 135 + 5*2), height=20,
                                               relx=0, rely=0, relwidth=0.75, relheight=0)
        # Learning Rate
        self.trainingDataLearningRate_Label.place(x=10, y=35 + 10*3 + 33*2 + 25 + 5, width=125, height=25,
                                                  relx=0, rely=0, relwidth=0, relheight=0)
        self.trainingDataLearningRate_Combobox.place(x=10 + 135 + 5, y=35 + 10*3 + 33*2 + 25 + 5 + 5/2, width=- (10 + 135 + 5*2), height=20,
                                                     relx=0, rely=0, relwidth=0.75, relheight=0)
        # Seperator
        self.seperator.place(x=20, y=35 + 10*3 + 33*2 + 25 + 5 + 35 + 5, width=-20*2, height=0,
                             relx=0, rely=0, relwidth=1, relheight=0)
        # Buttons
        self.trainData_Button.place(x=10, y=-38 - 38, width=-20, height=33,
                                    relx=0, rely=1, relwidth=1, relheight=0)
        self.testData_Button.place(x=10, y=-38, width=-20, height=33,
                                   relx=0, rely=1, relwidth=1, relheight=0)

    def fill_graph_Frame(self):
        """
        Fill the matplotlib graph
        """
        chart_type = FigureCanvasTkAgg(figure, self.graph_Frame)
        chart_type.get_tk_widget().pack()

        axis.set_ylim(-0.04, 1.04)
        axis.set_xlim(-10, 1000 + 10)
        figure.canvas.draw()
        figure.canvas.flush_events()

        axis.grid()
        axis.set_ylabel('MSE')
        axis.set_xlabel('Epoch')

    def fill_bottom_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
        self.weightsAndBiasesNN_Label = tk.Label(self.bottom_Frame,
                                                 text='Weights and Biases',
                                                 bg='#EEE')
        self.loadNN_Button = ttk.Button(self.bottom_Frame,
                                        text='Load',
                                        command=self.clicked_loadNN)
        self.saveNN_Button = ttk.Button(self.bottom_Frame,
                                        text='Save',
                                        command=self.clicked_saveNN)
        self.resetNN_Button = ttk.Button(self.bottom_Frame,
                                         text='Reset',
                                         command=self.clicked_resetNN)
        self.createNN_Button = ttk.Button(self.bottom_Frame,
                                          text='Create a Neural Network',
                                          command=self.clicked_createNN)
        # -Place Widgets-
        self.weightsAndBiasesNN_Label.place(x=0, y=0, width=140, height=0,
                                            relx=0, rely=0, relwidth=0, relheight=1)
        self.loadNN_Button.place(x=140, y=7/2, width=55, height=-7,
                                 relx=0, rely=0, relwidth=0, relheight=1)
        self.saveNN_Button.place(x=140 + 55, y=7/2, width=55, height=-7,
                                 relx=0, rely=0, relwidth=0, relheight=1)
        self.resetNN_Button.place(x=140 + 55*2 + 5, y=7/2, width=55, height=-7,
                                  relx=0, rely=0, relwidth=0, relheight=1)
        self.createNN_Button.place(x=-170 - 5, y=7/2, width=170, height=-7,
                                   relx=1, rely=0, relwidth=0, relheight=1)

    def open_inputFile_filedialog(self):
        """
        Ask the user to select an excel file representing the input
        """
        # -Check for valid inputs-
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return
        # -Get Path-
        path = tk.filedialog.askopenfilename(parent=self,
                                             title=f'Select Input File',
                                             initialfile='',
                                             filetypes=[
                                                 ('Excel', '*.xlsx'),
                                             ])
        if path:
            # Open Workbook
            wb = xlrd.open_workbook(path)
            sheet = wb.sheet_by_index(0)

            self.inputs.clear()
            for row in range(sheet.nrows):
                data_set = []
                for col in range(neural_network.num_inputs):
                    try:
                        # Extract Value of cell
                        value = sheet.cell(row, col).value
                        value = float(value)
                    except:
                        tk.messagebox.showerror(parent=self,
                                                title='Invalid Input Encountered',
                                                message='Please check if your inputs in the excel file are valid!',
                                                detail=f'Encountered Invalid Input:\n"{value}"',)
                        self.trainingDataInput_Label.configure(text='')
                        return
                    data_set.append(value)
                self.inputs.append(data_set)

            tk.messagebox.showinfo(parent=self,
                                   title='Success',
                                   message='Successfully read the input file!',)
            self.trainingDataInput_Label.configure(text=os.path.basename(path))

    def open_outputFile_filedialog(self):
        """
        Ask the user to select an excel file representing the output
        """
        # -Check for valid inputs-
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return
        # -Get Path-
        path = tk.filedialog.askopenfilename(parent=self,
                                             title=f'Select Output File',
                                             initialfile='',
                                             filetypes=[
                                                 ('Excel', '*.xlsx'),
                                             ])
        if path:
            # Open Workbook
            wb = xlrd.open_workbook(path)
            sheet = wb.sheet_by_index(0)

            self.outputs.clear()
            for row in range(sheet.nrows):
                data_set = []
                for col in range(neural_network.num_outputs):
                    try:
                        # Extract Value of cell
                        value = sheet.cell(row, col).value
                        value = float(value)
                    except:
                        tk.messagebox.showerror(parent=self,
                                                title='Invalid Output Encountered',
                                                message='Please check if your outputs in the excel file are valid!',
                                                detail=f'Encountered Invalid Output:\n"{value}"',)
                        self.trainingDataOutput_Label.configure(text='')
                        return
                    data_set.append(value)
                self.outputs.append(data_set)

            tk.messagebox.showinfo(parent=self,
                                   title='Success',
                                   message='Successfully read the output file!',)
            self.trainingDataOutput_Label.configure(text=os.path.basename(path))  # nopep8

    def clicked_trainNN(self):
        """
        Train the Neural Network and show the MSE (progress)
        on the graph next to it
        """
        # -Check for valid inputs-
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return
        try:
            epochs = int(self.trainingDataEpochs_Combobox.get().strip())
            learning_rate = float(self.trainingDataLearningRate_Combobox.get().strip())  # nopep8
        except ValueError:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Epoch Or Learning Rate',
                                    message='Please input number values for the epoch and learning rate!',)
            return
        if not self.inputs[0]:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Input',
                                    message='Please select an input file for training!',)
            return
        if not self.outputs[0]:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Output',
                                    message='Please select an output file for training!',)
            return
        # Check Dimensions
        if len(self.inputs[0]) != neural_network.num_inputs:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Input Dimension',
                                    message='The number of inputs in the data does not match the number of input neurons in the neural network!',
                                    detail=f'Neural Network Inputs: {neural_network.num_inputs}\nData Inputs: {len(self.inputs[0])}')
            return
        if len(self.outputs[0]) != neural_network.num_outputs:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Output Dimension',
                                    message='The number of outputs in the data does not match the number of output neurons in the neural network!',
                                    detail=f'Neural Network Outputs: {neural_network.num_outputs}\nData Outputs: {len(self.outputs[0])}')
            return

        # -Update Graphs x-ticker-
        axis.set_xlim(0, epochs)
        axis.set_xlim(-(epochs/100), epochs + (epochs/100))
        figure.canvas.draw()
        figure.canvas.flush_events()

        line.set_xdata(range(epochs))
        # -Train NN-
        self.trainData_Button.configure(state=tk.DISABLED)
        neural_network.perform_training(training_inputs=self.inputs,
                                        training_labels=self.outputs,
                                        epochs=epochs,
                                        lr=learning_rate,
                                        graph_data={
                                            'figure': figure,
                                            'axis': axis,
                                            'line': line},
                                        )
        self.trainData_Button.configure(state=tk.NORMAL)

    def clicked_testNN(self):
        """
        Test the Neural Network and save the outputs in an excel
        file and ask for the saving location
        """
        # -Check for valid inputs-
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return
        if not self.inputs[0]:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Input',
                                    message='Please select an input file for testing!',)

        outputs = neural_network.perform_test(self.inputs)

        # -Save Output-
        path = tk.filedialog.asksaveasfilename(parent=self,
                                               title='Save Outputs',
                                               filetypes=[('Excel', '.xlsx')],
                                               initialfile='Generated Output',
                                               defaultextension='.xlsx')
        if path:
            workbook = xlsxwriter.Workbook(path)
            worksheet = workbook.add_worksheet()
            worksheet.set_column(0, len(outputs[0]), 15)

            for row, row_data in enumerate(outputs):
                worksheet.write_row(row, 0, row_data)
            workbook.close()

    def clicked_loadNN(self):
        """
        Load Weights and Biases from given path
        """
        global neural_network
        path = tk.filedialog.askopenfilename(parent=self,
                                             title=f'Select Weights and Biases File',
                                             initialfile='',
                                             filetypes=[
                                                 ('numpy', '*.npy'),
                                             ])
        if path:
            # -Get Data of Neural Networks Dimensions of the Data-
            input_neurons = 0
            hidden_neurons = []
            output_neurons = 0

            array = np.load(path, allow_pickle=True)
            for weight_layer in array[0]:
                hidden_neurons.append(weight_layer.shape[0])
            else:
                hidden_neurons.append(weight_layer.shape[1])

            # hidden_neurons.append(len(layer))

            input_neurons = hidden_neurons.pop(0)
            output_neurons = hidden_neurons.pop(-1)

            if neural_network is None:
                if tk.messagebox.askyesno(parent=self,
                                          title='No Neural Network Available',
                                          message='No Neural network to load the Weights and Biases into.',
                                          detail='Do you want to automatically create a new Neural Network based on the data?'):

                    # -Create NN-
                    neural_network = fnn.Feedforward_Neural_Network(inputs=input_neurons,
                                                                    hidden=hidden_neurons,
                                                                    outputs=output_neurons)
                    self.currentModel_var.set(neural_network.generate_modelStructure())  # nopep8
                    tk.messagebox.showinfo(parent=self,
                                           title='Success',
                                           message='Successfully created a new Neural Network!')
                else:
                    return
            else:
                if not (input_neurons == neural_network.num_inputs and
                        set(hidden_neurons) == set(neural_network.num_hidden) and
                        output_neurons == neural_network.num_outputs):
                    if tk.messagebox.askyesno(parent=self,
                                              title='Invalid Dimensions',
                                              message='The Neural Networks Dimensions do not match the Dimensions of the Weights and Biases!',
                                              detail='Do you want to automatically create a new Neural Network based on the data?',
                                              icon='warning'):
                        # -Create NN-
                        neural_network = fnn.Feedforward_Neural_Network(inputs=input_neurons,
                                                                        hidden=hidden_neurons,
                                                                        outputs=output_neurons)
                        self.currentModel_var.set(neural_network.generate_modelStructure())  # nopep8
                        tk.messagebox.showinfo(parent=self,
                                               title='Success',
                                               message='Successfully created a new Neural Network!')
                    else:
                        return

            neural_network.load_WB(path)

    def clicked_saveNN(self):
        """
        Save Weights and Biases in given directory
        """
        # -Check for valid inputs-
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return

        # -Ask for Save Location-
        initialfile = neural_network.generate_fileName()
        path = tk.filedialog.asksaveasfilename(parent=self,
                                               title='Save Weights and Biases',
                                               filetypes=[('numpy', '.npy')],
                                               initialfile=initialfile,
                                               defaultextension='.npy')
        if path:
            # Save File
            neural_network.save_WB(path)

    def clicked_resetNN(self):
        """
        Reset Weights and Biases after asking for confirmation
        """
        if neural_network is None:
            tk.messagebox.showerror(parent=self,
                                    title='No Neural Network Available',
                                    message='Please first create a Neural Network!',)
            return

        if not tk.messagebox.askyesno(parent=self,
                                      title='Confirmation',
                                      message='Reseting the Weights and Biases is unreversible if you have not saved the current status of the neural network!',
                                      detail='Are you sure you want to proceed?',
                                      icon='warning'):
            return

        neural_network.randomize_WB()

    def clicked_createNN(self):
        """
        Reset Weights and Biases after asking for confirmation
        """
        NewNNWindow(self)


class NewNNWindow(tk.Toplevel):
    # --Constants--
    # None

    def __init__(self, master):
        # Run the __init__ method on the tk.Toplevel class
        super().__init__(master=master)
        window_width = 1920 * 0.15
        window_height = 1080 * 0.19

        # --Window Settings--
        self.title('Choose Dimensions')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=int(window_width),
            height=int(window_height),
            xpad=int(self.winfo_screenwidth()/2 -
                     window_width/2),
            ypad=int(self.winfo_screenheight()/2 -
                     window_height/2)))
        self.configure(bg='#FFFFFF')  # Set background color
        self.grab_set()
        self.update()

        # --Variables--
        self.hidden_var = tk.StringVar(value='10; 5')
        self.yourModel_var = tk.StringVar(value='N/A')

        # --Widgets--
        self.create_widgets()
        self.configure_widgets()
        self.place_widgets()

        # -Other-
        self.update_yourModel()

    # -Widget Methods-
    def create_widgets(self):
        """Create window widgets"""
        # -Input-
        self.input_Label = tk.Label(self, text='Input Neurons',
                                    bg='white')
        self.input_Combobox = ttk.Spinbox(self,
                                          from_=1, increment=1, to=999,
                                          command=self.update_yourModel)
        # -Hidden-
        self.hidden_Label = tk.Label(self, text='Hidden Neurons',
                                     bg='white')
        self.hidden_Entry = ttk.Entry(self,
                                      textvariable=self.hidden_var,)
        # -Output-
        self.output_Label = tk.Label(self, text='Output Neurons',
                                     bg='white')
        self.output_Combobox = ttk.Spinbox(self,
                                           from_=1, increment=1, to=999,
                                           command=self.update_yourModel)
        # -Info-
        self.yourModel_Label = tk.Label(self, text='Your Model',
                                        anchor=tk.CENTER, justify=tk.CENTER,
                                        bg='#EEE',
                                        textvariable=self.yourModel_var)
        # -Create NN-
        self.createNN_Button = ttk.Button(self, text='Create Neural Network',
                                          command=self.clicked_createNN)

        # -Bind Widgets-
        self.input_Combobox.bind('<FocusOut>', self.update_yourModel)
        self.hidden_Entry.bind('<FocusOut>', self.update_yourModel)
        self.output_Combobox.bind('<FocusOut>', self.update_yourModel)
        self.input_Combobox.bind('<KeyRelease>', self.update_yourModel)
        self.hidden_Entry.bind('<KeyRelease>', self.update_yourModel)
        self.output_Combobox.bind('<KeyRelease>', self.update_yourModel)

        # -Set Default Values-
        self.input_Combobox.set(25)
        self.output_Combobox.set(1)

    def configure_widgets(self):
        """Change widget styling and appearance"""

    def place_widgets(self):
        """Place main widgets"""
        # -Input-
        self.input_Label.place(x=15, y=15, width=-30 - 100, height=20,
                               relx=0, rely=0, relwidth=1, relheight=0)
        self.input_Combobox.place(x=-15 - 100, y=15, width=100, height=20,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        # -Hidden-
        self.hidden_Label.place(x=15, y=15 + 30*1, width=-30 - 100, height=20,
                                relx=0, rely=0, relwidth=1, relheight=0)
        self.hidden_Entry.place(x=-15 - 100, y=15 + 30*1, width=100, height=20,
                                relx=1, rely=0, relwidth=0, relheight=0)
        # -Output-
        self.output_Label.place(x=15, y=15 + 30*2, width=-30 - 100, height=20,
                                relx=0, rely=0, relwidth=1, relheight=0)
        self.output_Combobox.place(x=-15 - 100, y=15 + 30*2, width=100, height=20,
                                   relx=1, rely=0, relwidth=0, relheight=0)
        # -Info-
        self.yourModel_Label.place(x=30, y=15 + 30*3 + 5, width=-30*2, height=35,
                                   relx=0, rely=0, relwidth=1, relheight=0)
        # -Create NN-
        self.createNN_Button.place(x=15, y=-45, width=-15*2, height=35,
                                   relx=0, rely=1, relwidth=1, relheight=0)

    def update_yourModel(self, event=None):
        """Update the yourModel variable"""
        model = ''

        try:
            model += '%s >' % int(self.input_Combobox.get().strip())
            for hidden_neuron in self.hidden_var.get().split(';'):
                hidden_neuron = hidden_neuron.strip()
                if hidden_neuron == '':
                    continue
                model += '%s >' % int(hidden_neuron)
            model += '%s' % int(self.output_Combobox.get().strip())
        except ValueError:
            model = 'N/A'

        self.yourModel_var.set(model)

    def clicked_createNN(self):
        try:
            input_neurons = int(self.input_Combobox.get().strip())
            hidden_neurons = [int(neuron.strip()) for neuron in self.hidden_var.get().split(';') if neuron.strip() != '']  # nopep8
            output_neurons = int(self.output_Combobox.get().strip())
        except ValueError:
            tk.messagebox.showerror(parent=self,
                                    title='Invalid Inputs',
                                    message='Please input integer number values for the neurons!',
                                    detail='Hidden Neuron Layers are seperated with a ";"!')
            return

        global neural_network
        neural_network = fnn.Feedforward_Neural_Network(inputs=input_neurons,
                                                        hidden=hidden_neurons,
                                                        outputs=output_neurons)
        parent = self.nametowidget(self.winfo_parent())
        parent.currentModel_var.set(neural_network.generate_modelStructure())  # nopep8

        tk.messagebox.showinfo(parent=self,
                               title='Success',
                               message='Successfully created a new Neural Network!',)
        self.destroy()

    def destroy(self):
        """
        Run this before destruction
        """
        super().destroy()


if __name__ == "__main__":
    root = MainWindow()

    root.mainloop()
