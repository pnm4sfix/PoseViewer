"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox
from magicgui.widgets import create_widget, Widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from dask_image.imread import imread

from scipy.ndimage import gaussian_filter1d as gaussian_filter1d
from scipy.signal import find_peaks
import os
import pywt
import pims
import tables as tb
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation
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
        self.add_behaviour_text = TextEdit(label = "Enter new behavioural label")
        self.add_behaviour_button = PushButton(label = "Add new behaviour label")
        self.add_behaviour_button.clicked.connect(self.add_behaviour)

        self.label_menu = ComboBox(label='Behaviour labels', choices = [], tooltip = "Select behaviour label")
        push_button = PushButton(label = "Save Labels")
        self.extend([self.title_label, self.add_behaviour_text, self.add_behaviour_button, self.label_menu, push_button])

        # Number of nodes in pose estimation
        self.n_node_select = SpinBox(label = "Number of nodes")
        self.n_node_select.changed.connect(self.set_n_nodes)
        # Center node
        self.center_node_select = SpinBox(label = "Center node")
        self.center_node_select.changed.connect(self.set_center_node)

        # file pickers
        label_txt_picker = FileEdit(value='./Select a labeled txt file', tooltip = "Select labeled txt file")
        label_h5_picker = FileEdit(value='./Select a labeled h5 file', tooltip = "Select labeled h5 file")
        h5_picker = FileEdit(value='./Select a DLC h5 file', tooltip = "Select h5 file")
        vid_picker = FileEdit(value='./Select the corresponding raw video', tooltip = "Select corresponding raw video")
        self.extend([label_txt_picker, label_h5_picker, h5_picker, vid_picker])

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
        self.amd_threshold_spinbox = SpinBox(label = "Movement Threshold", tooltip = "Change movement threshold for pose estimation") 

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
        
        self.extend([self.n_node_select, self.center_node_select,self.confidence_threshold_spinbox, 
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
        

        self.add_1d_widget()
        self.viewer.dims.events.current_step.connect(self.update_slider)
        

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
            choices = self.label_menu.choices
            self.viewer.layers.remove(self.shapes_layer)
            del self.shapes_layer
            # reset_choices as they seem to be forgotten when layers added or deleted
            self.label_menu.choices = choices

        except:
            print("no shape layer")
        try:

            self.behaviour_no = event.value

        except:
            self.behaviour_no = event

        print("New behaviour is {}".format(self.behaviour_no))

        if (self.last_behaviour != 0) &(self.behaviour_no != 0): #event.value > 1:
            self.save_current_data()
            
        if (self.labeled != True):
            
            self.spinbox.max = len(self.behaviours)
            
        if self.behaviour_no >0:
            
            
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

                self.plot_behaving_region()
                #self.update_classification()
            
            
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

