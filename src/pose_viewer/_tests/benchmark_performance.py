from ._widget import ExampleQWidget
from ._loader import HyperParams
from ._loader import ZebData

import napari

def benchmark_model_performance():
        viewer = napari.Viewer()
        widget = ExampleQWidget(viewer)

        data_cfg, graph_cfg, hparams = self.initialise_params()
       

        
        model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(in_channels = self.numChannels, 
                                                num_class = self.numlabels, # self.numlabels, 
                                                graph_cfg = graph_cfg, 
                                                data_cfg = data_cfg, 
                                                hparams = hparams).load_from_checkpoint(self.chkpt, 
                                                                                        in_channels = self.numChannels, 
                                                                                        num_class = self.numlabels, #self.numlabels, 
                                                                                        graph_cfg = graph_cfg, 
                                                                                        data_cfg = data_cfg,
                                                                                        hparams = hparams)
        device = torch.device("cuda")
        
        model.to(device)

        N, C, T, V, M = self.batch.value, 3, 300, 19, 1

        dummy_input = torch.randn(N, C, T, V, M, dtype=torch.float).to(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
           _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
          for rep in range(repetitions):
             starter.record()
             _ = model(dummy_input)
             ender.record()
             # WAIT FOR GPU SYNC
             torch.cuda.synchronize()
             curr_time = starter.elapsed_time(ender)
             timings[rep] = curr_timemean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn)
        pass