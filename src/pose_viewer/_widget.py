"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from re import T
from socket import TCP_KEEPINTVL
import sys
sys.path.insert(1, "./")
from typing import TYPE_CHECKING


from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox
from magicgui.widgets import create_widget, Widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt


from scipy.ndimage import gaussian_filter1d as gaussian_filter1d
from scipy.signal import find_peaks
import os

from ._loader import ZebData, HyperParams
from torch.utils.data import DataLoader, random_split, Subset
import torch
from .models import st_gcn_aaai18_pylightning_3block, c3d

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import torch.nn as nn

import pims
import tables as tb
#from skimage.segmentation import watershed
#from skimage.feature import peak_local_max
#from scipy import ndimage as ndi
#from sklearn.manifold import TSNE
#from matplotlib.animation import FuncAnimation
import time
import scipy.stats as st

import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
try:
    from napari_video.napari_video import VideoReaderNP
except:
    print("no module named napari_video. pip install napari_video")
    import time

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
except:
    print("no cuda support")


if TYPE_CHECKING:
    import napari


class ExampleQWidget(Container):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Add behaviour labels to a list 
        
        self.title_label = Label(label = "Settings")

        self.decoder_dir_picker = FileEdit(label ="Select data folder", value='./', tooltip = "Select data folder", mode = "d")
        self.decoder_dir_picker.changed.connect(self.decoder_dir_changed)

        self.add_behaviour_text = TextEdit(label = "Enter new behavioural label")
        self.add_behaviour_button = PushButton(label = "Add new behaviour label")
        self.add_behaviour_button.clicked.connect(self.add_behaviour)

        self.label_menu = ComboBox(label='Behaviour labels', choices = [], tooltip = "Select behaviour label")
        push_button = PushButton(label = "Save Labels")
        self.extend([self.decoder_dir_picker,  self.add_behaviour_text, self.add_behaviour_button, self.label_menu, push_button]) #self.title_label,

        # Number of nodes in pose estimation
        self.n_node_select = SpinBox(label = "Number of nodes")
        self.n_node_select.changed.connect(self.set_n_nodes)
        # Center node
        self.center_node_select = SpinBox(label = "Center node")
        self.center_node_select.changed.connect(self.set_center_node)

        # file pickers
        label_txt_picker = FileEdit(label ="Select a labeled txt file", value='./', tooltip = "Select labeled txt file")
        label_h5_picker = FileEdit(label ="Select a labeled h5 file", value='./', tooltip = "Select labeled h5 file")
        h5_picker = FileEdit(label = "Select a DLC h5 file", value='./', tooltip = "Select h5 file")
        vid_picker = FileEdit(label = "Select the corresponding raw video", value='./', tooltip = "Select corresponding raw video")
        self.extend([h5_picker, vid_picker, label_h5_picker, label_txt_picker])

        # Behavioural extraction method
        self.behavioural_extract_method = ComboBox(label='Behaviour extraction method', choices = ["orth", "egocentric"],
                                                  tooltip = "Select preferred method for extracting behaviours (orth is best for zebrafish)")

        self.extract_method = self.behavioural_extract_method.value
        self.behavioural_extract_method.changed.connect(self.update_extract_method)
        self.extract_behaviour_button = PushButton(label = "Extract behaviour bouts")
        self.extract_behaviour_button.clicked.connect(self.extract_behaviours)
        self.add_behaviour_from_selected_area_button = PushButton(label = "Add behaviour from selected area")
        self.add_behaviour_from_selected_area_button.clicked.connect(self.add_behaviour_from_selected_area)
        self.confidence_threshold_spinbox = FloatSpinBox(label = "Confidence Threshold", tooltip = "Change confidence threshold for pose estimation") 
        self.amd_threshold_spinbox = FloatSpinBox(label = "Movement Threshold", tooltip = "Change movement threshold for pose estimation") 

        self.amd_threshold = 2
        self.confidence_threshold = 0.8
        self.confidence_threshold_spinbox.value = self.confidence_threshold
        self.amd_threshold_spinbox.value = self.amd_threshold

        self.confidence_threshold_spinbox.changed.connect(self.extract_behaviours)
        self.amd_threshold_spinbox.changed.connect(self.extract_behaviours)


        
        
        
        ### Variables to define
        self.classification_data = {}
        self.point_subset = np.array([])
        
        self.coords_data = {}
        self.spinbox = SpinBox(label = "Behaviour Number", tooltip = "Change behaviour")
        self.ind_spinbox = SpinBox(label = "Individual Number", tooltip = "Change individual")
        
        self.extend([self.confidence_threshold_spinbox, #self.n_node_select, self.center_node_select,
                     self.amd_threshold_spinbox, self.behavioural_extract_method, self.ind_spinbox,
                     self.extract_behaviour_button, self.add_behaviour_from_selected_area_button,
                     self.spinbox])
        

        
        
        
        self.labeled_txt = label_txt_picker.value
        self.labeled_h5 = label_h5_picker.value
        self.h5_file = h5_picker.value
        self.video_file = vid_picker.value

            
        h5_picker.changed.connect(self.h5_picker_changed)
        vid_picker.changed.connect(self.vid_picker_changed)
        label_h5_picker.changed.connect(self.convert_h5_todict)
        label_txt_picker.changed.connect(self.convert_txt_todict)
        self.ind_spinbox.changed.connect(self.individual_changed)
        self.spinbox.changed.connect(self.behaviour_changed)
        push_button.changed.connect(self.save_to_h5)
            
        self.ind = 0
        self.behaviour_no = 0
        self.clean = None # old function may be useful in future
        self.im_subset = None
        self.labeled = False
        self.behaviours = []
        self.choices = []
        self.b_labels = None
        self.decoder_data_dir = None

        self.add_1d_widget()
        self.viewer.dims.events.current_step.connect(self.update_slider)

        ### infererence
        self.batch_size_spinbox = SpinBox(label = "Batch Size", value = 16)
        self.num_workers_spinbox = SpinBox(label = "Num Workers", value = 8)
        self.lr_spinbox = FloatSpinBox(label = "Learning Rate", value = 0.01, step = 0.0000)
        self.dropout_spinbox = FloatSpinBox(label = "Dropout", value = 0)
        self.num_labels_spinbox = SpinBox(label = "Num labels", value = 0)
        self.num_channels_spinbox = SpinBox(label = "Num channels", value = 3)

        self.model_dropdown = ComboBox(label='Model type', choices = ["ZebLR", "Zeb2.0"],
                                                  tooltip = "Select model for predicting behaviour")

        self.chkpt_dropdown = ComboBox(label='Model type', choices = [],
                                                  tooltip = "Select chkpt for predicting behaviour")
        self.model_dropdown.changed.connect(self.populate_chkpt_dropdown)

        self.analyse_button = PushButton(label = "Analyse")
        self.analyse_button.clicked.connect(self.analyse)

        self.train_button = PushButton(label = "Train")
        self.train_button.clicked.connect(self.train)

        self.finetune_button = PushButton(label = "Finetune")
        self.finetune_button.clicked.connect(self.finetune)

        self.test_button = PushButton(label = "Test")
        self.test_button.clicked.connect(self.test)

        

        self.extend([self.batch_size_spinbox,self.lr_spinbox,  self.chkpt_dropdown,
                    self.train_button, self.analyse_button, self.finetune_button, self.test_button ])#self.num_workers_spinbox,  self.dropout_spinbox,
                     #self.num_labels_spinbox, self.num_channels_spinbox, self.model_dropdown,, 
                     #])

        
    def decoder_dir_changed(self, value):
        # Look for and load yaml configuration file
        # load config
        self.decoder_data_dir = value
        print("Decoder Data Folder is {}".format(self.decoder_data_dir))
        self.config_file = os.path.join(self.decoder_data_dir, "decoder_config.yml")

        

        try:

            with open(self.config_file, 'r') as file:
                self.config_data = yaml.safe_load(file)


            n_nodes = self.config_data["data_cfg"]["V"]
            center_node = self.config_data["data_cfg"]["center"]
            self.set_center_node(center_node)
            self.set_n_nodes(n_nodes)
        except:
            print("No configuration yaml located in decoder data folder")

        self.populate_chkpt_dropdown() # load ckpt files if any

        print("decoder config is {}".format(self.config_data))
        self.view_data()


    
    def populate_chkpt_dropdown(self):
        # get all checkpoint files and allow user to select one

        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        if os.path.exists(log_folder):
            version_folders = [version_folder for version_folder in os.listdir(log_folder) if "version" in version_folder]
        
            self.ckpt_files = []
            for version_folder in version_folders:
                version_folder_files = os.listdir(os.path.join(log_folder, version_folder))

                for sub_file in version_folder_files:
                    if ".ckpt" in sub_file:
                        ckpt_file = os.path.join(version_folder, sub_file)
                        self.ckpt_files.append(ckpt_file)

        else:
            self.ckpt_files = []


        self.chkpt_dropdown.choices = self.ckpt_files


    def update_slider(self, event):
        print("updating slider")
        #if (event.axis == 0):
        print(event)
        self.frame = event.value[0]
        try:
            self.frame_line.data = np.c_[[self.frame, self.frame], [0, 10]]

        except:
            print("Failed to update frame line")
        #stokes_1d.plot(None, ind_lambda)
        print("updating slider frame {}".format(self.frame))

    

    def add_1d_widget(self):
        

        self.viewer1d = napari_plot.ViewerModel1D()
        widget = QtViewer(self.viewer1d)
        self.viewer.window.add_dock_widget(widget, area="bottom", name="Movement")
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.axis.y_label = "Movement"
        self.viewer1d.reset_view()
        self.frame = 0

        self.frame_line = self.viewer1d.add_line( np.c_[[self.frame, self.frame], [0, 10]], color = "gray")
        

        # Moving frames? - redundant 
        # Preprocess_txt_file? - maybe include later if a tx file is selected with a pop up?
        # Extend window - redundant

    def add_behaviour_from_selected_area(self, value):
        # get x range of viewer1d
        # subset data using those frame indices
        # append behaviour to self behaviours
        
        start, stop = self.viewer1d.camera.rect[:2]
        self.behaviours.append((int(start), int(stop)))
        self.behaviour_changed(len(self.behaviours))

    def plot_movement_1d(self):

        # plot colors mapped to confidence interval - can't do this yet even for scatter
        #ci = self.ci.iloc[:].std(axis = 0).to_numpy()
        #norm = plt.Normalize()
        #colors = plt.cm.jet(norm(ci))
        choices = self.label_menu.choices
        t = np.arange(self.gauss_filtered.shape[0])

        self.viewer1d.add_line( np.c_[t, self.gauss_filtered], color = "magenta", label = "Movement")
        self.viewer1d.add_line(np.c_[[0, self.gauss_filtered.shape[0]], [self.threshold, self.threshold]], color = "cyan", label = "Movement threshold")
        self.viewer1d.reset_view()
        self.label_menu.choices = choices
        
    def plot_behaving_region(self):
        choices = self.label_menu.choices
        regions = [
                ([self.start, self.stop], "vertical"),
            ]

        layer = self.viewer1d.add_region(
                regions,
                color=["green"],
                opacity = 0.4,
                name = "Behaviour",
            )
        self.label_menu.choices = choices

    def reset_layers(self):
            """Resest all napari layers. Called three times to ensure layers removed."""
            for layer in reversed(self.viewer.layers):
                #print(layer)
                self.viewer.layers.remove(layer)
            time.sleep(1)
            print("Layers remaining are {}".format(self.viewer.layers))
            try:
                for layer in self.viewer.layers:
                    #print(layer)
                    self.viewer.layers.remove(layer)
            except:
                pass
        
            try:
                for layer in self.viewer.layers:
                    #print(layer)
                    self.viewer.layers.remove(layer)
            except:
                pass
    
    def save_current_data(self):
        """Called when behaviour is changed."""
        self.classification_data[self.ind][self.last_behaviour] = {"classification" : self.label_menu.current_choice,
                                                            "coords" : self.point_subset,
                                                            "start" : self.start,
                                                            "stop": self.stop,
                                                             "ci" : self.ci_subset}
    
    def update_classification(self):
        """Updates classification label in GUI"""
        print("updated")
        #if self.labeled:
        #    try:
        #        self.label_menu.choices = tuple(self.txt_behaviours)
        #    except:
        #        pass
        #print(self.label_menu.choices)
        #print(self.classification_data[self.ind][self.behaviour_no]["classification"])
        try:
            print(self.label_menu.choices)
            print(self.classification_data[self.ind][self.behaviour_no]["classification"] in self.label_menu.choices)
            print(self.classification_data[self.ind][self.behaviour_no]["classification"])
            print(type(self.classification_data[self.ind][self.behaviour_no]["classification"]))
            self.label_menu.value = self.classification_data[self.ind][self.behaviour_no]["classification"]
        except:
            self.label_menu.value = str(self.classification_data[self.ind][self.behaviour_no]["classification"])

    def update_extract_method(self, value):

        self.extract_method = value
        print("Extract method is {}".format(self.extract_method))

    def get_points(self):
        """Converts coordinates into points format for napari points layer"""
        #print("Getting Individuals Points")
        x_flat = self.x.to_numpy().flatten()
        y_flat = self.y.to_numpy().flatten()
        z_flat = np.tile(self.x.columns, self.x.shape[0])

        zipped = zip(z_flat, y_flat, x_flat)
        points = [[z, y, x] for z, y, x in zipped]
        points = np.array(points)

        self.points = points

    def get_tracks(self):
        """Converts coordinates into tracks format for napari tracks layer"""
        #print("Getting Individuals Tracks")
        x_nose = self.x.to_numpy()[-1]
        y_nose = self.y.to_numpy()[-1]
        z_nose = np.arange(self.x.shape[1])
        nose_zipped = zip(z_nose, y_nose, x_nose)
        tracks = np.array([[0, z, y, x] for z, y, x in nose_zipped])

        self.tracks = tracks  
    
    def egocentric_variance(self):
        """Estimates locomotion based on peaks of egocentric movement ."""
        reshap = self.points.reshape(self.n_nodes, -1, 3)
        center = reshap[self.center_node, :, 1:] #selects x,y center nodes
        self.egocentric = reshap.copy()
        self.egocentric[:,:, 1:] = reshap[:, :, 1:]-center.reshape((-1, *center.shape)) # subtract center nodes
        absol_traj = (self.egocentric[:, 1:, 1:] - self.egocentric[:, :-1, 1:]) #trajectory
        self.euclidean = np.sqrt(np.abs((absol_traj[:, :, 0]**2) + (absol_traj[:, :, 1]**2))) # euclidean trajectory
        var = np.median(self.euclidean, axis=0) #median movement
        self.gauss_filtered = gaussian_filter1d(var, int(self.fps/10)) #smoothed movement
        amd = np.median(self.gauss_filtered-self.gauss_filtered[0])/0.6745
        peaks = find_peaks(self.gauss_filtered, prominence=amd*7, distance = int(self.fps/2), width = 5, rel_height = 0.6) #zeb
        
        #check stop does not come before start
        self.behaviours = [(int(start)-20, int(end)+20) for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"]) if end>start] # zeb
        
        # check behaviour has high confidence score
        self.behaviours = [(start, end) for start, end in self.behaviours if self.check_behaviour_confidence(start, end)]
        
        # check no overlap
        b_arr = np.array(self.behaviours)
        #b_arr = (b_arr/30) #convert to seconds
        overlap = b_arr[1:, 0]-b_arr[:-1, 1]
        overlap = np.where(overlap<=0)[0]+1

        
        b_arr[overlap, 0] = b_arr[overlap,0] + 10
        b_arr[overlap-1, 1] = b_arr[overlap,0] - 10
        self.behaviours = b_arr.tolist()
        
        #self.moving_fig, self.moving_ax = plt.subplots()
        #self.moving_ax.plot(self.gauss_filtered)
        #self.moving_ax.scatter(peaks[0], self.gauss_filtered[peaks[0]])
        #[self.moving_ax.axvspan(int(start), int(end), color=(0, 1, 0, 0.5)) for start, end in self.behaviours]


    def calulate_orthogonal_variance(self, amd_threshold = 2, confidence_threshold = 0.8):
        """Estimates locomotion based on orthogonal movement. Good for zebrafish."""
        #print("Calculating Orthogonal Variance")
        
        # Get euclidean trajectory - not necessary for orthogonal algorithm but batch requires it
        reshap = self.points.reshape(self.n_nodes, -1, 3)
        center = reshap[self.center_node, :, 1:] #selects x,y center nodes
        self.egocentric = reshap.copy()
        self.egocentric[:,:, 1:] = reshap[:, :, 1:]-center.reshape((-1, *center.shape)) # subtract center nodes
        absol_traj = (self.egocentric[:, 1:, 1:] - self.egocentric[:, :-1, 1:]) #trajectory
        self.euclidean = np.sqrt(np.abs((absol_traj[:, :, 0]**2) + (absol_traj[:, :, 1]**2))) # euclidean trajectory
        
        # use egocentric instead to eliminate crop jitter 
        #subsize = int(self.points.shape[0]/self.n_nodes)
        projections = []
        #maybe check % is 0
        for n in range(self.n_nodes):
            #subset = self.points[n*subsize: (n+1)*subsize]
            #trajectory_matrix = subset[1:, 1:] - subset[:-1, 1:]
            trajectory_matrix = absol_traj[n]
            orth_matrix = np.flip(trajectory_matrix, axis=1)
            orth_matrix[:,0] = -orth_matrix[:,0] # flip elements in trajectory matrix so x is y and y is x and reverse sign of first element. Only works for 2D vectors
            future_trajectory = trajectory_matrix[1:, ] #shift trajectory by looking forward 
            present_orth = orth_matrix[:-1, ] # subset all orth but last one
            projection = np.abs((np.sum(future_trajectory*present_orth, axis=1))/np.linalg.norm(present_orth, axis=1)) #project the dot product of each trajectory vector onto its orth vector
            projection[np.isnan(projection)] = 0
            projections.append(projection)

        proj = np.array(projections)
        var = np.median(proj, axis=0)
        self.gauss_filtered = gaussian_filter1d(var, int(self.fps/10)) #smoothed movement
        amd = st.median_abs_deviation(self.gauss_filtered)#    np.median(self.gauss_filtered)/0.6745
        median = np.median(self.gauss_filtered)
        self.threshold = amd * amd_threshold
        peaks = find_peaks(self.gauss_filtered, prominence=self.threshold, distance = int(self.fps/2), width = 5, rel_height = 0.6) #zeb
       
        #check stop does not come before start
        self.behaviours = [(int(start)-20, int(end)+20) for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"]) if end>start] # zeb
        
        # check behaviour has high confidence score
        self.behaviours = [(start, end) for start, end in self.behaviours if self.check_behaviour_confidence(start, end, confidence_threshold)]
        self.bad_behaviours = [(start, end) for start, end in self.behaviours if not self.check_behaviour_confidence(start, end, confidence_threshold)]
        # check no overlap
        b_arr = np.array(self.behaviours)
        
        if b_arr.ndim == 2:
            #b_arr = (b_arr/30) #convert to seconds
            overlap = b_arr[1:, 0]-b_arr[:-1, 1]
            overlap = np.where(overlap<=0)[0]+1


            b_arr[overlap, 0] = b_arr[overlap,0] + 10
            b_arr[overlap-1, 1] = b_arr[overlap,0] - 10
        self.behaviours = b_arr.tolist()
        
        #self.moving_fig, self.moving_ax = plt.subplots()
        #self.moving_ax.plot(self.gauss_filtered)
        #self.moving_ax.scatter(peaks[0], self.gauss_filtered[peaks[0]])
        #[self.moving_ax.axvspan(int(start), int(end), color=(0, 1, 0, 0.5)) for start, end in self.behaviours] # maybe utilise napari plot here?


    def check_behaviour_confidence(self, start, stop, confidence_threshold = 0.8):
        
        #subset confidence interval data for behaviour
        subset = self.ci.iloc[:, start:stop]
        
        #count number of values below threshold
        low_ci_counts = subset[(subset< confidence_threshold)].count()
        
        # average counts
        mean_low_ci_count = low_ci_counts.mean()
        
        #return boolean, True if ci counts are low (< 1) or high if ci_counts >1
        return mean_low_ci_count <= 1

    
        
    
    def plot_movement(self):
        """ Plot movement as track in shape I, Z, Y, X.
        X is range 0 - 1000
        y is range 1250 - 1200
        
        """
        z_no = len(self.gauss_filtered)
        x = np.arange(z_no)
        ratio = 1000/z_no
        x = (x * ratio).astype("int64")
        y = self.gauss_filtered # scale y to within 50
        y_ratio = y.max()/ 400
        y= -(y/y_ratio) + 200
        

        z = np.arange(0, z_no)
        i = np.zeros(z_no)
        self.movement  = np.stack([i, z, y, x]).transpose()

        
        self.movement_layer = self.viewer.add_tracks(self.movement, tail_length = 1000, tail_width = 3, opacity =1, colormap="twilight")
        self.label_menu.choices = self.choices
        
        
     
    def movement_labels(self):
        # get all moving frames
        moving_frames_idx = np.array([], dtype = "int64")

        for start, stop in  np.array(self.behaviours).tolist():#[random_integers].tolist():
            arr = np.arange(start, stop, dtype = "int64")
            moving_frames_idx = np.append(moving_frames_idx, arr)

        #get centre node
        centre = self.points.reshape(self.n_nodes, -1, 3)[self.center_node]

        #tile and reshape centre location
        centre_rs =  np.tile(centre[moving_frames_idx], 4).reshape(-1, 4, 3)

        #create array to add to centre node to create bounding box
        add_array = np.array([[0, -100, -100],
                              [0, -100, 100],
                              [0, 100, 100],
                              [0, 100, -100]])

        #define boxes by adding to centre_rs
        boxes = centre_rs + add_array.reshape(-1, *add_array.shape)

        #specify label params
        nframes = 300 # at the moment more than 300 is really slow
        labels = ["movement"] * nframes
        properties = {
            'label': labels,

        }

        # specify the display parameters for the text
        text_params = {
            'text': 'label: {label}',
            'size': 12,
            'color': 'green',
            'anchor': 'upper_left',
            'translation': [-3, 0]}

        # add shapes layer
        self.shapes_layer = self.viewer.add_shapes(boxes[:nframes], shape_type='rectangle', edge_width=5,
                              edge_color='#55ff00', face_color = "transparent", visible =True, properties=properties, text = text_params)
        self.label_menu.choices = self.choices


    def h5_picker_changed(self, event):
            """This function is called when a new h5/csv from DLC is selected.
        
            Parameters:
        
            event: widget event"""
            try:
                self.h5_file = event.value.value
            except:
                try:
                    self.h5_file = event.value
                except:
                    self.h5_file = str(event)
            #self.full_reset()
            self.read_coords(self.h5_file)
            
            
    
    def vid_picker_changed(self, event):
            """This function is called when a new video is selected.
        
            Parameters:
        
            event: widget event"""
            try:
                self.video_file = event.value.value
            except:
                try:
                    self.video_file = event.value
                except:
                    self.video_file = str(event)

            vid = pims.open(str(self.video_file))
            self.fps = vid.frame_rate
            
            self.im = VideoReaderNP(str(self.video_file))
            
            # add a video layer if none
            if self.im_subset is None:
                
                self.im_subset = self.viewer.add_image(self.im, name = "Video Recording")
                self.label_menu.choices = self.choices
            else:

                self.im_subset.data = self.im

        
    def convert_h5_todict(self, event):
        """reads pytables and converts to dict. If new dict saved overwrites existing pytables"""
        try:
            self.labeled_h5 = event.value.value
        except:
            try:
                self.labeled_h5 = event.value
            except:
                self.labeled_h5 = str(event)


        self.labeled_h5_file = tb.open_file(self.labeled_h5, mode = "a")
        self.classification_data = {}

        for group in self.labeled_h5_file.root.__getattr__("_v_groups"):
            ind = self.labeled_h5_file.root[group]
            behaviour_dict = {}
            arrays = {}
            
            for array in self.labeled_h5_file.list_nodes(ind, classname='Array'):
                arrays[int(array.name)] = array
            tables = []
            
            for table in self.labeled_h5_file.list_nodes(ind, classname="Table"):
                tables.append(table)

            behaviours = []
            classifications = []
            starts = []
            stops = []
            cis = []
            for row in tables[0].iterrows():
                behaviours.append(row["number"])
                classifications.append(row["classification"])
                starts.append(row["start"])
                stops.append(row["stop"])
                

            for behaviour, (classification, start, stop) in enumerate(zip(classifications, starts, stops)):
                class_dict = {"classification": classification.decode('utf-8'),
                                "coords" : arrays[behaviour+1][:, :3], 
                                 "start" : start,
                                 "stop" : stop,
                                 "ci" : arrays[behaviour+1][:, 3]}
                behaviour_dict[behaviour+1] = class_dict
            self.classification_data[int(group)] = behaviour_dict
        
        self.labeled = True
        self.labeled_h5_file.close()
        self.ind_spinbox.max = max(self.classification_data.keys())
        self.ind_spinbox.value = 0
        self.spinbox.value = 0 
        self.tracks = None # set this to none as it's not saved
        self.ind = 0
        self.choices = pd.Series([label["classification"] for k,label in self.classification_data[1].items()]).unique().tolist()
        print(self.choices)
        self.label_menu.choices = tuple(self.choices)
        
    def convert_txt_todict(self, event):
        """Reads event text file and converts it to usable format to display behaviours in GUI."""
        #self.full_reset()
        try:
                self.labeled_txt = event.value.value
        except:
            try:
                self.labeled_txt = event.value
            except:
                self.labeled_txt = str(event)
        
            
        event_df = pd.read_csv(self.labeled_txt, ",", header=2)
        
        if self.preprocess_txt_file:
            event_df = self.preprocess_txt(event_df)
        
        if self.extend_window:
            event_df.iloc[:, 1] = event_df.iloc[:, 1] + 500 # added because maggot behaviour durations cut behaviour short

        self.txt_behaviours = event_df.iloc[:, 2].unique().astype("str").tolist()
        self.label_menu.choices = tuple(self.txt_behaviours)
        fps = self.fps
        event_df.iloc[:, :2] = ((event_df.iloc[:, :2]/1e3) * fps).astype("int64")

        
        self.labeled = True
        self.ind_spinbox.value = 1
        
        key = list(self.coords_data.keys())[self.ind-1]
        self.x = self.coords_data[key]["x"]
        self.y = self.coords_data[key]["y"]
        self.ci = self.coords_data[key]["ci"]
        self.get_points()
        
        ind_dict = {}
        for n, row in enumerate(event_df.itertuples()):
            self.start = int(row[1]) #Time
            self.stop = int(self.start + np.ceil(row[2])) #Duration
            classification = row[3] #TrackName
            
            self.point_subset = self.points.reshape((self.n_nodes, -1, 3))[:, int(self.start):int(self.stop)].reshape(-1, 3)
            self.point_subset = self.point_subset - np.array([self.start, 0, 0])
            self.ci_subset = self.ci.iloc[:, int(self.start):int(self.stop)].to_numpy().flatten()
            behav_dic = {"classification": classification,
                        "coords": self.point_subset,
                        "start": self.start,
                        "stop": self.stop,
                        "ci" : self.ci_subset}

            ind_dict[n+1] = behav_dic

        self.classification_data = {}
        self.classification_data[1] = ind_dict
        
        self.ind_spinbox.max = max(self.classification_data.keys())
        
        self.spinbox.value = 0
        self.behaviour_no = 0
        self.label_menu.reset_choices()
        self.txt_behaviours = event_df.iloc[:, 2].unique().astype("str").tolist()
        self.label_menu.reset_choices()
        self.label_menu.choices = tuple(self.txt_behaviours)
        print(self.label_menu.choices)


        
    def individual_changed(self, event):
        """Called when individual spin box is changed. Gets coordinates for new individual, adds a points and tracks layer
        to the GUI and also estimates periods of locomotion."""
        last_ind = self.ind
        self.ind = event
        print("New individual is individual {}".format(self.ind))
        
        #check ind in data
        if self.labeled == True:
            
            self.im_subset.data = self.im
            
        else:
            exists = len([n for n, v in enumerate(self.coords_data) if self.ind-1 == n])
            if exists > 0:

                key = list(self.coords_data.keys())[self.ind-1]

                self.x = self.coords_data[key]["x"]
                self.y = self.coords_data[key]["y"]
                self.ci = self.coords_data[key]["ci"]
                self.get_points()
                self.get_tracks()
                
                #self.reset_layers()

                #self.viewer.add_image(self.im)
                self.im_subset.data = self.im
                
                #create points layer
                self.points_layer = self.viewer.add_points(self.points, size=3, visible= False)
                #self.track_layer = self.viewer.add_tracks(self.tracks, tail_length = 100, tail_width = 3)
                self.label_menu.choices = self.choices
    
    def extract_behaviours(self, value=None):
        print("Extracting behaviours using {} method".format(self.extract_method))
        if self.extract_method == "orth":
            #if (self.points.shape[0] > 1e6) & (cp.cuda.runtime.getDeviceCount() >0):
            #    print("Large video - sing GPU accelerated movement extraction")
                #self.calculate_orthogonal_variance_cupy()
            #else:
            self.amd_threshold = self.amd_threshold_spinbox.value
            self.confidence_threshold = self.confidence_threshold_spinbox.value
            self.calulate_orthogonal_variance(self.amd_threshold, self.confidence_threshold)
            self.movement_labels()
            #self.plot_movement()
                    
        elif self.extract_method == "egocentric":
            self.egocentric_variance()
            self.movement_labels()
            #self.plot_movement()
        else:
            pass

        #check if ind exists in classification data

        #exists = len([k for k in self.classification_data.keys() if k == self.ind])
        if self.ind in self.classification_data:#exists > 0:
            pass
        else:
            self.classification_data[self.ind] = {}

        #else:
        #     self.ind_spinbox.value = last_ind
        self.spinbox.value = 0
        self.plot_movement_1d()

            
    def behaviour_changed(self, event):
        """Called when behaviour number is changed."""
        self.last_behaviour = self.behaviour_no
        
        try:
            #choices = self.label_menu.choices
            self.viewer.layers.remove(self.shapes_layer)
            del self.shapes_layer
            # reset_choices as they seem to be forgotten when layers added or deleted
            self.label_menu.choices = self.choices

        except:
            print("no shape layer")
        try:

            self.behaviour_no = event.value

        except:
            self.behaviour_no = event

        print("New behaviour is {}".format(self.behaviour_no))

        
            
        #if (self.labeled != True):
            
        #    self.spinbox.max = len(self.behaviours)
            
        if self.behaviour_no >0:

            if self.ind in self.classification_data.keys():

                if (self.last_behaviour != 0) &(self.behaviour_no != 0): #event.value > 1:
                    self.save_current_data()
            
            
                #exists = len([k for k in self.classification_data[self.ind].keys() if k == self.behaviour_no])
                if self.behaviour_no in self.classification_data[self.ind]: #exists > 0:
                    print("exists")
                    # use self.classification_data
                
                    #self.reset_layers()
                
                    # get points from here, too complicated to create tracks here i think
                    #print(self.label_menu.choices)
                    self.point_subset = self.classification_data[self.ind][self.behaviour_no]["coords"]
                    self.start = self.classification_data[self.ind][self.behaviour_no]["start"]
                    self.stop = self.classification_data[self.ind][self.behaviour_no]["stop"]
                    self.ci_subset = self.classification_data[self.ind][self.behaviour_no]["ci"]
                    #self.im_subset = self.viewer.add_image(self.im[self.start:self.stop])
                    #self.points_layer = self.viewer.add_points(self.point_subset, size=5)
                
                
                    self.im_subset.data = self.im[self.start:self.stop]
                
                    #self.im_subset = self.viewer.layers[0]
                    try:
                        self.points_layer.data = self.point_subset
                    except:
                        self.points_layer = self.viewer.add_points(self.point_subset, size=5)
                        self.label_menu.choices = self.choices

                    if self.tracks is not None:
                        self.track_subset = self.tracks[self.start:self.stop]
                        self.track_subset = self.track_subset - np.array([0, self.start, 0, 0]) # zero z because add_image has zeroed
                
                        try:
                            self.track_layer.data = self.track_subset
                        except:
                    
                            self.track_layer = self.viewer.add_tracks(self.track_subset, tail_length = 500, tail_width = 3)
                            self.label_menu.choices = self.choices
                    #self.points_layer.data = self.point_subset
                
                    if self.label_menu.choices == ():
                        try:
                            self.label_menu.choices = self.txt_behaviours
                        except:
                            pass
                    self.update_classification()
                    #print(self.label_menu.choices)
                
                
                
                
                elif (self.behaviour_no not in self.classification_data[self.ind]) & (len(self.behaviours)>0):
                    print("extracting behaviour")
                    self.start, self.stop = self.behaviours[self.behaviour_no-1] # -1 because behaviours is array indexed
                    #self.reset_layers()

                    #self.im_subset = self.viewer.add_image(self.im[self.start:self.stop])
                    self.im_subset.data = self.im[self.start:self.stop]
                
                    dur = self.stop-self.start
                    self.point_subset = self.points.reshape((self.n_nodes, -1, 3))[:, self.start:self.stop].reshape((int(self.n_nodes*dur), 3))
                    self.point_subset = self.point_subset - np.array([self.start, 0, 0]) # zero z because add_image has zeroed

                    self.track_subset = self.tracks[self.start:self.stop]
                    self.track_subset = self.track_subset - np.array([0, self.start, 0, 0]) # zero z because add_image has zeroed
                
                    self.ci_subset = self.ci.iloc[:, self.start:self.stop].to_numpy().flatten()

                
                    #self.im_subset = self.viewer.layers[0]
                    #self.im_subset.data = self.im[self.start:self.stop]
                
                    #self.im_subset = self.viewer.layers[0]
                    try:
                        self.points_layer.data = self.point_subset
                    except:
                        self.points_layer = self.viewer.add_points(self.point_subset, size=5)
                        self.label_menu.choices = self.choices
                
                    #self.points_layer.data = self.point_subset
                    #self.points_layer = self.viewer.add_points(self.point_subset, size=5)
                    try:
                        self.track_layer.data = self.track_subset
                    except:
                    
                        self.track_layer = self.viewer.add_tracks(self.track_subset, tail_length = 500, tail_width = 3)
                        self.label_menu.choices = self.choices
                
           
                    if self.label_menu.choices == ():
                        print(self.label_menu.choices)
                        try:
                            self.label_menu.choices = self.txt_behaviours
                        except:
                            pass
                        print(self.label_menu.choices)

                    if self.b_labels is not None:
                        self.label_menu.value = self.b_labels[self.behaviour_no-1]
                        print("Label score is {}".format(self.predictions.numpy()[self.behaviour_no-1]))

                    self.plot_behaving_region()
                    #self.update_classification()
            elif (len(self.behaviours) == 0):
                self.show_data(self.behaviour_no -1)

            
            
                # save value classification value to something

    def save_to_h5(self, event):
        """converts classification data to pytables format for efficient storage.
        Creates PyTables file, and groups for each individual. Classification is stored
        in a table and coordinates are stored in arrays"""
        filename = str(self.video_file) +"_classification.h5"
        if os.path.exists(filename):
            print("{} exists".format(filename))
            classification_file = tb.open_file(filename, mode = "a", title = "classfication")
        else:
            classification_file = tb.open_file(filename, mode = "w", title = "classfication")
        
        # loop through ind, then loop through behaviours
        for ind in self.classification_data.keys():
            
            #if os.path.exists(filename): #add this to be able to append to h5 file
            #    print("classification file already exists")
                #ind_group = 
                #ind_table = 
            #else:
            ind_group = classification_file.create_group("/", str(ind), "Individual" + str(ind))
            ind_table = classification_file.create_table(ind_group, "labels", "Behaviour", "Individual {} Behaviours".format(str(ind)))
            
            ind_subset = self.classification_data[ind]
            
            for behaviour in ind_subset:
                try:
                    arr = ind_subset[behaviour]["coords"]
                    ci = ind_subset[behaviour]["ci"]
                    array = np.concatenate((arr, ci.reshape(-1, ci.shape[0]).T), axis=1)
                    classification_file.create_array(ind_group, str(behaviour), array,  "Behaviour" + str(behaviour))



                    ind_table.row["number"] = behaviour
                    ind_table.row["classification"] = ind_subset[behaviour]["classification"]
                    ind_table.row["n_nodes"] = self.n_nodes
                    ind_table.row["start"] = ind_subset[behaviour]["start"]
                    ind_table.row["stop"] = ind_subset[behaviour]["stop"]
                    ind_table.row.append()
                    ind_table.flush()
                except:
                    print("no pose data")
                
        classification_file.close()

    def read_coords(self, h5_file):
        """Reads coordinates from DLC files (h5 and csv). Optional data cleaning."""
        
        if ".h5" in str(h5_file):
            
            self.dlc_data = pd.read_hdf(h5_file)
            data_t = self.dlc_data.transpose()
            
            try:
                data_t["individuals"]
                data_t = data_t.reset_index()
            except:
                data_t["individuals"] = ["individual1"]*data_t.shape[0]
                data_t = data_t.reset_index().set_index(["scorer", "individuals", "bodyparts", "coords"]).reset_index()
        
        if ".csv" in str(h5_file):
            
            self.dlc_data = pd.read_csv(h5_file, header = [0,1, 2], index_col =0)
            data_t = self.dlc_data.transpose()
            data_t["individuals"] = ["individual1"]*data_t.shape[0]
            data_t = data_t.reset_index().set_index(["scorer", "individuals", "bodyparts", "coords"]).reset_index()

        for individual in data_t.individuals.unique():
            indv1 = data_t[data_t.individuals == individual].copy()
            # calculate interframe variability
            if self.clean:
                indv1.loc[:, 0:] = indv1.loc[:,0:].interpolate(axis=1) # fillsna
            x = indv1.loc[indv1.coords == "x", 0:].reset_index(drop=True)
            y = indv1.loc[indv1.coords == "y", 0:].reset_index(drop=True)
            ci = indv1.loc[indv1.coords == "likelihood", 0:].reset_index(drop=True)
            
            # cleaning
            if self.clean:
                x[ci<0.8] = np.nan
                y[ci<0.8] = np.nan

                x = x.interpolate(axis=1)
                y = y.interpolate(axis=1)
                
            self.coords_data[individual] = {"x" : x,
                                            "y" : y,
                                            "ci": ci} # think i need ci for the model too
        self.ind_spinbox.max = int(data_t.individuals.unique().shape[0])

    def add_behaviour(self, value):

        behaviour_label = self.add_behaviour_text.value

        # assert value contains a word in string
        assert len(behaviour_label) > 0

        assert type(behaviour_label) == str
        choices = list(self.label_menu.choices)
        choices.append(behaviour_label)
        self.choices = choices
        self.label_menu.choices = tuple(choices)
        self.add_behaviour_text.value = ""
            
    def set_n_nodes(self, value):
        self.n_nodes = value
        print("Number of nodes is {}".format(self.n_nodes))

    def set_center_node(self, value):
        self.center_node = value
        print("Center node is {}".format(self.center_node))

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")

    def analyse(self, value):
        self.preprocess_bouts() ## assumes behaviours extracted
        self.predict_behaviours()
        self.update_classification_data_with_predictions()


    def preprocess_bouts(self):
        # arrange in N, C, T, V format
        self.zebdata = ZebData()
        points = self.egocentric[:, :, 1:]
        points = np.swapaxes(points, 0, 2)
        ci_array = self.ci.to_numpy()
        ci_array = ci_array.reshape((*ci_array.shape, 1))
        cis = np.swapaxes(ci_array, 0, 2)

        # N, C, T, V, M - don't ignore confidence interval but give option of removing 
        N = len(self.behaviours)
        C = self.config_data["train_cfg"]["num_channels"]
        T2 = self.config_data["data_cfg"]["T2"]
        denominator = self.config_data["data_cfg"]["denominator"]
        T_method = self.config_data["data_cfg"]["T"]
        fps = self.config_data["data_cfg"]["fps"]

        if T_method == "window":
            T = 2 * int(fps/denominator)

        elif type(T_method) == "int":
            T = T_method # these methods assume behaviours last the same amount of time -which is a big assumption

        V = points.shape[2]
        M = 1

        bouts = np.zeros((N, C, T, V, M))
        padded_bouts = np.zeros((N, C, T2, V, M))

        # loop through movement windows when behaviour occuring 
        for n,  (bhv_start, bhv_end) in enumerate(self.behaviours):

            # focus on window of size tsne window around peak of movement
            bhv_mid = bhv_start + ((bhv_end - bhv_start)/2)
            new_start = int(bhv_mid - int(T/2)) # might be worth refining self behaviours from here
            new_end = int(bhv_mid + int(T/2))
            new_end = (T- (new_end-new_start)) + new_end # this adds any difference if not exactly T in length

            #self.behaviours[n] = (new_start, new_end)

            bhv = points[:, new_start:new_end]
            ci = cis[:, new_start:new_end]
            ci = ci.reshape((*ci.shape, 1))
                        
            # switch to x, y from y, x
            bhv_rs = bhv.copy()
            bhv_rs[1] = bhv[0]
            bhv_rs[0] = bhv[1]
            bhv = bhv_rs
                        
            # reflect y to convert to cartesian friendly coordinate system - is this needed if coordinates are egocentric?
            bhv = bhv.reshape((*bhv.shape, 1)) # reshape to N, C, T, V, M
            y_mirror = np.array([[1, 0],
                                [0, -1]])
                               
            for frame in range(bhv.shape[1]):
                bhv[:2, frame] = (bhv[:2, frame].T @ y_mirror).T
                        

            # align function takes shape N, C, T, V, M
            bhv_align = self.zebdata.align(bhv)
            
                        
            bouts[n, :2] = bhv_align
            bouts[n,  2] = ci[0]

            padded_bouts[n] =  self.zebdata.pad(bouts[n], 300)

            
        self.zebdata.data = padded_bouts
        self.zebdata.labels = np.zeros(padded_bouts.shape[0])

    def predict_behaviours(self):
        data_cfg, graph_cfg, hparams = self.initialise_params()

        data_loader = DataLoader(self.zebdata, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory = False)
        
        # load model check point, 
        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt =  os.path.join(log_folder, self.chkpt_dropdown.value)  # spinbox

        model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(in_channels = self.numChannels, 
                                                   num_class = self.numlabels, 
                                                   graph_cfg = graph_cfg, 
                                                   data_cfg = data_cfg, 
                                                   hparams = hparams).load_from_checkpoint(self.chkpt, 
                                                                                           in_channels = self.numChannels, 
                                                                                           num_class = self.numlabels, 
                                                                                           graph_cfg = graph_cfg, 
                                                                                           data_cfg = data_cfg,
                                                                                           hparams = hparams)


        # create trainer object, 
        ## optimise trainer to just predict as currently its preparing data thinking its training
        # predict
        trainer = Trainer(devices=self.devices, accelerator = self.accelerator)
        predictions = trainer.predict(model, data_loader) # returns a list of the processed batches
        self.predictions = torch.concat(predictions, dim = 0)
        print(self.predictions)


    def update_classification_data_with_predictions(self):

        label_dict = self.config_data["data_cfg"]["classification_dict"]

        # add predictions to classification data
        #if self.model_dropdown.value == "ZebLR":
        #    label_dict = {0 : "forward",
        #                  1: "left",
        #                  2: "right"}
        #
        preds = torch.argmax(self.predictions, dim=1).numpy()

        print(type(preds))
        print("predictions is {}".format(preds))

        self.b_labels = pd.Series(preds).map(label_dict).to_numpy()
        print(self.b_labels)
            
        self.choices = tuple(label_dict.values())
        self.label_menu.choices = self.choices

        # maybe loop and create new classification_data

    def convert_classification_files(self, train_files):
        # use folder and convert classification files 
        train_bouts = []
        train_labels = []
        

        classification_files = [tb.open_file(file, mode = "r") for file in train_files]
        for file in classification_files:
            classification_data = self.read_classification_h5(file)
            C = self.config_data["train_cfg"]["num_channels"] #3 # publicise these
            V = self.config_data["data_cfg"]["V"] #    19
            M = 1
            fps = self.config_data["data_cfg"]["fps"] # 330.
            denominator = self.config_data["data_cfg"]["denominator"] #8

            T_method = self.config_data["data_cfg"]["T"]
            

            if T_method == "window":
                T = 2 * int(fps/denominator)

            elif type(T_method) == "int":
                T = T_method # these methods assume behaviours last the same amount of time -which is a big assumption

            elif T_method == "None":
                # ragged nest, T should be max length and everything padded to that
                T = None # T2 should be specified to ensure same length


            center = self.config_data["data_cfg"]["center"]
            T2 = self.config_data["data_cfg"]["T2"]
            align = self.config_data["data_cfg"]["align"]
            all_bouts, all_labels = self.classification_data_to_bouts(classification_data, C, T, V, M, center= center, T2= T2, align=align)
            train_bouts.append(all_bouts)
            train_labels.append(all_labels)

        train_bouts = np.concatenate(train_bouts)
        train_labels = np.array([item for sublist in train_labels for item in sublist])


        return train_bouts, train_labels

    def prepare_data(self):
        
            # Prepare and save data
            # take one datafolder and 
            all_files = [os.path.join(self.decoder_data_dir, file) for file in os.listdir(self.decoder_data_dir) if "classification.h5" in file]
            nfiles= len(all_files)

            train_proportion = int(0.85 * nfiles)

            self.train_files = all_files[: train_proportion]
            self.test_files = all_files[train_proportion:]


            self.train_data,self.train_labels = self.convert_classification_files(self.train_files)
            self.test_data, self.test_labels = self.convert_classification_files(self.test_files)
            
            # drop labels to ignore
            labels_to_ignore = self.config_data["data_cfg"]["labels_to_ignore"]
            print("Labels to ignore are {}".format(labels_to_ignore))
            good_train_idx = np.where(~np.isin(self.train_labels, labels_to_ignore))[0]
            good_test_idx = np.where(~np.isin(self.test_labels, labels_to_ignore))[0]
            print("Subset of train labels ignoring some labels {}".format(self.train_labels[good_train_idx[:10]]))
            # check if any labels to ignore are still in 
            print("Unique labels are {}".format(np.unique(self.train_labels)))


            self.train_data, self.train_labels = self.train_data[good_train_idx], self.train_labels[good_train_idx]
            self.test_data, self.test_labels = self.test_data[good_test_idx], self.test_labels[good_test_idx]

            self.class_dict = self.config_data["data_cfg"]["classification_dict"]
            self.label_dict = {v:k for k, v in self.class_dict.items()}
            #label_dict = {k:v for v, k in enumerate(np.unique(self.train_labels))}
            print("Label dict is {}".format(self.label_dict))

            self.train_labels = pd.Series(self.train_labels).map(self.label_dict).to_numpy()
            self.test_labels = pd.Series(self.test_labels).map(self.label_dict).to_numpy()

            # check if any labels to ignore are still in 
            print("Unique labels are {}".format(np.unique(self.train_labels)))
            print("Label_counts are {}".format(pd.Series(self.train_labels).value_counts()))

            print("Training Data Shape is {}".format(self.train_data.shape))
            print("Test Data Shape is {}".format(self.test_data.shape))

            #np.save(os.path.join(self.decoder_data_dir, "label_dict.npy"), self.label_dict)
            self.augmentation = self.config_data["data_cfg"]["augmentation"]
            if self.augmentation is not False:
                zebdata = ZebData()
                zebdata.data = self.train_data
                zebdata.labels = self.train_labels
                zebdata.ideal_sample_no = self.augmentation
                zebdata.dynamic_augmentation()

                self.train_data = zebdata.data
                self.train_labels = zebdata.labels

                print("Augmented Training Data Shape is {}".format(self.train_data.shape))



            np.save(os.path.join(self.decoder_data_dir, "Zebtrain.npy"), self.train_data)
            np.save(os.path.join(self.decoder_data_dir, "Zebtrain_labels.npy"), self.train_labels)
            np.save(os.path.join(self.decoder_data_dir, "Zebtest.npy"), self.test_data)
            np.save(os.path.join(self.decoder_data_dir, "Zebtest_labels.npy"), self.test_labels)

            print("Data Prepared and Save at {}".format(self.decoder_data_dir))

    def train(self):
        #self.decoder_data_dir = self.decoder_dir_picker.value
        # Load prepare data
        if os.path.exists(os.path.join(self.decoder_data_dir, "Zebtrain.npy")):
            print("Data Prepared")
           

        else:
            print("Preparing Data")
            self.prepare_data()

        # train 

        data_cfg, graph_cfg, hparams = self.initialise_params()

        # create trainer object, 
        ## optimise trainer to just predict as currently its preparing data thinking its training
        # predict
        for n in range(4): # does ths reuse model in current state?
            
            if self.backbone == "ST-GCN":

                model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(in_channels = self.numChannels, 
                                                       num_class = self.numlabels, 
                                                       graph_cfg = graph_cfg, 
                                                       data_cfg = data_cfg, 
                                                       hparams = hparams)
            elif self.backbone == "C3D":
                model = c3d.C3D(num_class =self.numlabels,
                                 num_channels = self.numChannels,
                                 data_cfg = data_cfg,
                                 hparams= hparams,
                                 num_workers = self.num_workers
                                 )

            for param in model.parameters():
                if param.requires_grad:
                    print("param {} requires grad".format(param))
        

            print("trial is {}".format(n))

            TTLogger = TensorBoardLogger(save_dir = self.decoder_data_dir)
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience = 5)
            swa = StochasticWeightAveraging(swa_lrs=1e-2)

            log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
            if os.path.exists(log_folder):
                if len(os.listdir(log_folder)) > 0:
                    version_folders = [version_folder for version_folder in os.listdir(log_folder) if "version" in version_folder]
                    latest_version_number = max([int(version_folder.split("_")[-1]) for version_folder in version_folders]) # this is not working quite right not selectin latest folder
                    print("latest version folder is {}".format(latest_version_number))
                    new_version_number = latest_version_number + 1
                    new_version_folder = os.path.join(log_folder, "version_{}".format(new_version_number))
                    print(new_version_folder)

                else:
                    new_version_folder = os.path.join(log_folder, "version_0")

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

            print("new version folder is {}".format(new_version_folder))


            checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath= new_version_folder,
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,        # save the best model
            mode="min",
            every_n_epochs=1
            )

            ## Run this 4 times and select best model - fine tune that

        

            trainer = Trainer(logger = TTLogger,
                devices = 1, accelerator = "gpu",
                max_epochs=100,
                callbacks = [early_stop, checkpoint_callback], auto_lr_find=True)#, stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

            trainer.tune(model)

            ### DEBUG - overfit
            #trainer = Trainer(devices =1, accelerator = "gpu", overfit_batches=0.01)


            trainer.fit(model)

            print("Finished Training - best model is {}".format(checkpoint_callback.best_model_path)) 
        # Add finetune - freeze model-replace last layer and train

    def initialise_params(self):
        self.numlabels = self.config_data["data_cfg"]["numLabels"]
        self.devices = self.config_data["train_cfg"]["devices"]
        self.auto_lr = self.config_data["train_cfg"]["auto_lr"]
        self.accelerator = self.config_data["train_cfg"]["accelerator"]
        self.graph_layout = self.config_data["train_cfg"]["graph_layout"]
        self.dropout = self.config_data["train_cfg"]["dropout"]
        self.numChannels = self.config_data["train_cfg"]["num_channels"]
        self.num_workers = self.config_data["train_cfg"]["num_workers"]

        
        # create dataloader from preprocess swims
        self.batch_size = self.batch_size_spinbox.value # spinbox
        #self.num_workers = self.num_workers_spinbox.value # spinbox
        self.lr = self.lr_spinbox.value # spinbox
        #self.dropout = self.dropout_spinbox.value # spinbox

        try:
            self.backbone = self.config_data["train_cfg"]["backbone"]

        except:
            "print no backbone- defaulting to STGCN"
            self.backbone = "ST-GCN"

        try:
            self.transform = self.config_data["train_cfg"]["transform"]
            print("transform is {}".format(self.transform))
            if self.transfrom == "None":
                self.transform = None
        except:
            self.transform = None

        try:
            self.labels_to_ignore =  self.config_data["data_cfg"]["labels_to_ignore"]
        except:
            self.labels_to_ignore = None
        
        # assign model parameters
        PATH_DATASETS = self.decoder_data_dir
        #self.numlabels = self.num_labels_spinbox.value # spinbox
        #self.numChannels = self.num_channels_spinbox.value # X, Y and CI - spinbox
        

        data_cfg = { 'data_dir': PATH_DATASETS,
                    'augment': False,
                    'ideal_sample_no' : None,
                    'shift' : False,
                     'transform':self.transform,
                     'labels_to_ignore': self.labels_to_ignore}

        graph_cfg = {"layout":self.graph_layout}

        hparams = HyperParams(self.batch_size, self.lr, self.dropout)

        return (data_cfg, graph_cfg, hparams)

    def finetune(self):
        
        # load best checkpoint

        # freeze model
        #model.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # create model
        # load checkpoint
        # set all layers to grad = False
        # change configure optimised in st -gcn backbone
        # train
        data_cfg, graph_cfg, hparams = self.initialise_params()
       

        
        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt =  os.path.join(log_folder, self.chkpt_dropdown.value)  # spinbox

        for n in range(4): # does ths reuse model in current state?
            model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(in_channels = self.numChannels, 
                                                   num_class = self.numlabels, 
                                                   graph_cfg = graph_cfg, 
                                                   data_cfg = data_cfg, 
                                                   hparams = hparams).load_from_checkpoint(self.chkpt, 
                                                                                           in_channels = self.numChannels, 
                                                                                           num_class = self.numlabels, 
                                                                                           graph_cfg = graph_cfg, 
                                                                                           data_cfg = data_cfg,
                                                                                           hparams = hparams)


        
            # freeze model layers
            for param in model.parameters():
                param.requires_grad = False

            print(model.parameters)

            # add new model.fcn 
            model.fcn = nn.Conv2d(256, self.numlabels, kernel_size=1)

            for param in model.parameters():
                if param.requires_grad:
                    print("param {} requires grad".format(param))


            print("trial is {}".format(n))

            TTLogger = TensorBoardLogger(save_dir = self.decoder_data_dir)
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience = 5)
            swa = StochasticWeightAveraging(swa_lrs=1e-2)

            if os.path.exists(log_folder):
                if len(os.listdir(log_folder)) > 0:
                    version_folders = [version_folder for version_folder in os.listdir(log_folder) if "version" in version_folder]
                    latest_version_number = max([int(version_folder.split("_")[-1]) for version_folder in version_folders]) # this is not working quite right not selectin latest folder
                    print("latest version folder is {}".format(latest_version_number))
                    new_version_number = latest_version_number + 1
                    new_version_folder = os.path.join(log_folder, "version_{}".format(new_version_number))
                    print(new_version_folder)

                else:
                    new_version_folder = os.path.join(log_folder, "version_0")

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

            print("new version folder is {}".format(new_version_folder))


            checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath= new_version_folder,
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,        # save the best model
            mode="min",
            every_n_epochs=1
            )

            ## Run this 4 times and select best model - fine tune that

        

            trainer = Trainer(logger = TTLogger,
                devices = 1, accelerator = "gpu",
                max_epochs=100,
                callbacks = [early_stop, checkpoint_callback], auto_lr_find=True)#, stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

            trainer.tune(model)

            trainer.fit(model)

            print("Finished Finetuning - best model is {}".format(checkpoint_callback.best_model_path)) 
        
        

    def test(self):
        data_cfg, graph_cfg, hparams = self.initialise_params()
       

        
        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt =  os.path.join(log_folder, self.chkpt_dropdown.value)  # spinbox

        
        model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(in_channels = self.numChannels, 
                                                num_class = self.numlabels, 
                                                graph_cfg = graph_cfg, 
                                                data_cfg = data_cfg, 
                                                hparams = hparams).load_from_checkpoint(self.chkpt, 
                                                                                        in_channels = self.numChannels, 
                                                                                        num_class = self.numlabels, 
                                                                                        graph_cfg = graph_cfg, 
                                                                                        data_cfg = data_cfg,
                                                                                        hparams = hparams)


        
        model.freeze()

        

        TTLogger = TensorBoardLogger(save_dir = self.decoder_data_dir)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience = 5)
        swa = StochasticWeightAveraging(swa_lrs=1e-2)

        if os.path.exists(log_folder):
            if len(os.listdir(log_folder)) > 0:
                version_folders = [version_folder for version_folder in os.listdir(log_folder) if "version" in version_folder]
                latest_version_number = max([int(version_folder.split("_")[-1]) for version_folder in version_folders]) # this is not working quite right not selectin latest folder
                print("latest version folder is {}".format(latest_version_number))
                new_version_number = latest_version_number + 1
                new_version_folder = os.path.join(log_folder, "version_{}".format(new_version_number))
                print(new_version_folder)

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

        else:
            new_version_folder = os.path.join(log_folder, "version_0")

        print("new version folder is {}".format(new_version_folder))

        

        

        trainer = Trainer(logger = TTLogger,
            devices = self.devices, accelerator = self.accelerator,
            max_epochs=100,
            callbacks = [early_stop], auto_lr_find=True)#, stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

            
        trainer.test(model)
        predictions = trainer.predict(model, model.test_dataloader())
       
        predictions = torch.concat(predictions, dim = 0)

        np.save(os.path.join(os.path.dirname(self.chkpt), "predictions.npy"), predictions.numpy())

        print("Finished Testing")


    
    def read_classification_h5(self, file):
        
        classification_data = {}
        for group in file.root.__getattr__("_v_groups"):
            ind = file.root[group]
            behaviour_dict = {}
            arrays = {}

            for array in file.list_nodes(ind, classname='Array'):
                arrays[int(array.name)] = array
            tables = []

            for table in file.list_nodes(ind, classname="Table"):
                tables.append(table)

            behaviours = []
            classifications = []
            starts = []
            stops = []
            cis = []
            for row in tables[0].iterrows():
                behaviours.append(row["number"])
                classifications.append(row["classification"])
                starts.append(row["start"])
                stops.append(row["stop"])





            for (behaviour, classification, start, stop) in (zip(behaviours, classifications, starts, stops)):
            
                    class_dict = {"classification": classification.decode('utf-8'),
                                    "coords" : arrays[behaviour][:, :3], 
                                     "start" : start,
                                     "stop" : stop,
                                     "ci" : arrays[behaviour][:, 3]}
                    behaviour_dict[behaviour+1] = class_dict
            
            classification_data[int(group)] = behaviour_dict
        return classification_data




    def classification_data_to_bouts(self, classification_data, C, T, V, M, center = None, T2 = None, align = None):
        print(type(classification_data))
        all_ind_bouts = [] 
        all_labels = []
        for ind in classification_data.keys():
            behaviour_dict = classification_data[ind] 
            N = len(behaviour_dict.keys())
            
            if (T2 == "None") | (T2 is None):
                bout_data = np.zeros((N, C, T, V, M))
            else:
                bout_data = np.zeros((N, C, T2, V, M))
            bout_labels = []
            for bout_idx, bout in enumerate(behaviour_dict.keys()):

                # get coords
                coords = behaviour_dict[bout]["coords"]

                # reshape coords to V, T, (frame, Y, X)
                coords_reshaped = coords.reshape(V, -1, 3)

                print(coords_reshaped.shape)

                # get ci
                ci =  behaviour_dict[bout]["ci"]
                ci_reshaped = ci.reshape(V, -1, 1)

                # subset behaviour from the middle out
                # focus on window of size tsne window around peak of movement
            
                if T == None:
                    # take behaviour as is 
                    # pad to T2
                    print("T  is none")
                    coords_subset = coords_reshaped
                    ci_subset = ci_reshaped

                else:


                    mid_idx = int(coords_reshaped.shape[1]/2)
                    new_start =int( mid_idx - int(T/2))
                    new_end = int(mid_idx + int(T/2))
                    new_end = (T- (new_end-new_start)) + new_end # this adds any difference if not exactly T in length

                    coords_subset = coords_reshaped[:, new_start:new_end]
                    print(new_start, new_end)
                    print(coords_subset.shape)
                    ci_subset = ci_reshaped[:, new_start:new_end]
            
                # reshape from V, T, C to C, T, V
                swapped_coords = np.swapaxes(coords_subset, 0, 2)
                new_bout = np.zeros(swapped_coords.shape)
                print(new_bout.shape)
                swapped_ci = np.swapaxes(ci_subset, 0, 2)

                new_bout[0] = swapped_coords[2] # x
                new_bout[1] = swapped_coords[1]# y
                new_bout[2] = swapped_ci[0] # ci
            
                # reflect y to convert to cartesian friendly coordinate system
                new_bout = new_bout.reshape((*new_bout.shape, M)) # reshape to N, C, T, V, M
                y_mirror = np.array([[1, 0],
                                    [0, -1]])

                for frame in range(new_bout.shape[1]):
                    new_bout[:2, frame] = (new_bout[:2, frame].T @ y_mirror).T
                
                # align, pad, 
            
                zebdata = ZebData()
            
                if center != "None":
                    print("centering bout on center {}".format(center))
                    new_bout = zebdata.center_all(new_bout, center)
                    #new_bout = centered_bout.copy()

                if align:
                    print("aligning bout")
                    new_bout = zebdata.align(new_bout)

                if T2 != "None": 
                    print("padding bout, original size was {}, new T is {}".format(new_bout.shape, T2))
                    new_bout = zebdata.pad(new_bout, T2)
                    
    

                bout_data[bout_idx] = new_bout
                label = behaviour_dict[bout]["classification"]
                bout_labels.append(label)
            
            all_ind_bouts.append(bout_data)
            all_labels.append(bout_labels)
    
        all_ind_bouts = np.concatenate(all_ind_bouts)
        all_labels = np.array(all_labels).flatten()
    
        return all_ind_bouts, all_labels

    def view_data(self):
        train_data = np.load(os.path.join(self.decoder_data_dir, "Zebtrain.npy"))
        train_labels = np.load(os.path.join(self.decoder_data_dir, "Zebtrain_labels.npy"))
        self.transform = self.config_data["train_cfg"]["transform"]
        self.batch_size = self.batch_size_spinbox.value
        self.zebdata = ZebData(transform = self.transform)
        self.zebdata.data = train_data
        self.zebdata.labels = train_labels

        self.spinbox.max = self.zebdata.labels.shape[0]

    def show_data(self, idx):
        print(self.zebdata[idx][0].shape)
        if self.transform == "heatmap":

            if self.im_subset != None:
            
                self.im_subset.data = np.max(self.zebdata[idx][0].numpy(), axis=0)
            else:
                self.im_subset = self.viewer.add_image(np.max(self.zebdata[idx][0].numpy(), axis=0), name = "Video Recording")
        
        






