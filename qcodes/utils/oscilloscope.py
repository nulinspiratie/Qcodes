from typing import Sequence
import traceback
import numpy as np
import multiprocessing as mp
import logging
import time
from collections import namedtuple

from qcodes.instrument.parameter_node import ParameterNode, parameter
from qcodes.instrument.parameter import Parameter

import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
from pyqtgraph.multiprocess.remoteproxy import ClosedError

TransformState = namedtuple('TransformState', 'translate scale revisit')

logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/17103698/plotting-large-arrays-in-pyqtgraph?rq=1


class Oscilloscope(ParameterNode):
    """Create an oscilloscope GUI that displays acquisition traces

    Example code:

    ```
    oscilloscope = Oscilloscope(channels=['chA', 'chB', 'chC', 'chD'])
    oscilloscope.start_process()

    # Define settings
    oscilloscope.ylim = (-0.8, 1)
    oscilloscope.channels_settings['chB']['scale'] = 10
    oscilloscope.channels_settings['chB']['name'] = 'ESR'
    oscilloscope.channels_settings['chC']['scale'] = 50
    oscilloscope.channels_settings['chD']['scale'] = 50
    oscilloscope.update_settings()

    # Register oscilloscope update event with acquisition controller
    triggered_controller.buffer_actions = [oscilloscope.update_array]
    """

    def __init__(
            self,
            channels: list,
            max_points=200000,
            max_samples=200,
            channels_settings=None,
            figsize=(1200, 350),
            sample_rate=200e3,
            ylim=(-2, 2),
            interval=0.1,
            show_legend=True,
            channel_plot_2D=None,
            show_1D_from_2D: bool = False
    ):
        super().__init__('oscilloscope', use_as_attributes=True)
        self.max_samples = max_samples
        self.max_points = max_points
        self.channel_plot_2D = channel_plot_2D
        self.show_1D_from_2D = show_1D_from_2D

        assert isinstance(channels, (list, tuple))
        self.channels = channels

        self.channels_settings = channels_settings or {}
        for channel in channels:
            self.channels_settings.setdefault(channel, {})

        self.shape_1D = (len(channels), max_points)
        self.shape_2D = (len(channels), max_samples, max_points)

        self.sample_rate = sample_rate
        self.ylim = ylim
        self.interval = interval
        self.show_legend = show_legend

        # Create multiprocessing array for 1D traces
        self.mp_array_1D = mp.RawArray("d", int(len(channels) * max_points))
        self.np_array_1D = np.frombuffer(self.mp_array_1D, dtype=np.float64).reshape(
            self.shape_1D
        )

        # Create multiprocessing array for 2D traces
        self.mp_array_2D = mp.RawArray(
            "d", int(len(channels) * max_samples * max_points)
        )
        self.np_array_2D = np.frombuffer(self.mp_array_2D, dtype=np.float64).reshape(
            self.shape_2D
        )

        self.figsize = figsize

        self.queue = mp.Queue()

        self.process = None

        self.clim = Parameter(initial_value=None)

    @parameter
    def clim_vals(self, parameter, value):
        if value is None:
            return True
        elif isinstance(value, tuple) and len(value) == 2:
            return True
        else:
            return False

    @parameter
    def clim_set(self, parameter, value):
        """Set 2D colorscale limits

        cmin and cmax must both be floats that set the limits, or they are both
        None, in which case the limits are automatically scaled
        """
        if isinstance(value, tuple):
            cmin, cmax = value
            self._run_code(f'self.img_2D.setLevels(({cmin}, {cmax}))')

    def snapshot_base(self, update: bool = False,
                      params_to_skip_update: Sequence[str] = None,
                      skip_parameters: Sequence[str] = (),
                      skip_parameter_nodes: Sequence[str] = ()):
        return {}

    def start_process(self):
        self.process = mp.Process(
            target=OscilloscopeProcess,
            kwargs=dict(
                mp_array_1D=self.mp_array_1D,
                mp_array_2D=self.mp_array_2D,
                shape_1D=self.shape_1D,
                shape_2D=self.shape_2D,
                queue=self.queue,
                channels=self.channels,
                channels_settings=self.channels_settings,
                figsize=self.figsize,
                sample_rate=self.sample_rate,
                ylim=self.ylim,
                interval=self.interval,
                show_legend=self.show_legend,
                channel_plot_2D=self.channel_plot_2D,
                show_1D_from_2D=self.show_1D_from_2D
            ),
            daemon=True
        )
        self.process.start()

    def update_settings(self):
        self.queue.put(
            {
                "message": "update_settings",
                "ylim": self.ylim,
                "channels_settings": self.channels_settings,
                "sample_rate": self.sample_rate,
                "interval": self.interval,
                "show_legend": self.show_legend,
            }
        )

    def update_array_1D(self, array):
        assert len(array) == len(self.channels)

        if isinstance(array, dict):
            # Convert dict with an array per channel into a single array
            array = np.array(list(array.values()))

        points = array.shape[1]
        if points > self.max_points:
            array = array[:, :self.max_points]
            points = self.max_points

        # Copy new array to shared array
        self.np_array_1D[:, :points] = array

        self.queue.put({"message": "new_trace_1D", "points": points})

    def update_array_2D(self, array):
        if isinstance(array, dict):
            # Convert dict with an array per channel into a single array
            array = np.array(list(array.values()))

        channels, samples, points = array.shape

        assert channels == len(self.channels)
        if samples > self.max_samples:
            samples = self.max_samples
            array = array[:, :samples, :]
        if points > self.max_points:
            points = self.max_points
            array = array[:, :, :points]

        # Copy new array to shared array
        self.np_array_2D[:, :samples, :points] = array

        info = {"message": "new_trace_2D", "samples": samples, "points": points}

        # Add clim if needed
        if self.clim is not None:
            info['levels'] = self.clim

        self.queue.put(info)

    def _run_code(self, code):
        self.queue.put({"message": "execute", "code": code})


class OscilloscopeProcess:
    process = None
    rpg = None

    def __init__(
            self,
            mp_array_1D,
            mp_array_2D,
            shape_1D,
            shape_2D,
            queue,
            channels,
            channels_settings,
            figsize,
            sample_rate,
            ylim,
            interval,
            show_legend,
            channel_plot_2D,
            show_1D_from_2D
    ):
        self.shape_1D = shape_1D
        self.shape_2D = shape_2D
        self.queue = queue
        self.channels = channels
        self.channels_settings = channels_settings
        self.sample_rate = sample_rate
        self.ylim = ylim
        self.interval = interval
        self.show_legend = show_legend
        self.channel_plot_2D = channel_plot_2D
        self.show_1D_from_2D = show_1D_from_2D

        self.samples = None
        self.points = None
        self.legend = None

        self.mp_array_1D = mp_array_1D
        self.np_array_1D = np.frombuffer(mp_array_1D, dtype=np.float64).reshape(self.shape_1D)
        self.mp_array_2D = mp_array_2D
        self.np_array_2D = np.frombuffer(mp_array_2D, dtype=np.float64).reshape(self.shape_2D)

        self.active = True

        self.win = self.initialize_plot(figsize)
        if channel_plot_2D is not None:
            self.ax_1D = self.win.addPlot(0, 0)
            self.ax_2D = self.win.addPlot(1, 0)
            self.img_2D = self.rpg.ImageItem()
            self.img_2D.translate(0, 0)
            self.img_2D.scale(1 / self.sample_rate * 1e3, 1)

            self.ax_2D.getAxis('bottom').setLabel('Time', 'ms')
            self.ax_2D.getAxis('left').setLabel('Repetition', '')

            self.ax_2D.addItem(self.img_2D)
        else:
            self.ax_1D = self.win.addPlot()
            self.ax_2D = None
            self.img_2D = None

        self.ax_1D.disableAutoRange()

        try:
            self.legend = self.ax_1D.addLegend()
            self.legend.setVisible(show_legend)
            self.curves = [
                self.ax_1D.plot(pen=(k, self.shape_1D[0]),
                                name=channels_settings[channels[k]].get("name", channels[k])
                                )
                for k in range(self.shape_1D[0])
            ]
        except:
            print(traceback.format_exc())

        self.process_loop()

    def log(self, message):
        # print(message)
        pass

    def process_loop(self):
        while self.active:
            t0 = time.perf_counter()
            if not self.queue.empty():
                while not self.queue.empty():
                    info = self.queue.get(block=False)

                message = info.pop("message")
                self.log(f'Received message: {message}')

                if message == "new_trace_1D":
                    # Show a single trace
                    self.update_plot_1D(**info)
                elif message == 'new_trace_2D':
                    # Show a 2D plot of traces
                    self.update_plot_2D(**info)
                    self.counter_1D_from_2D = 0
                elif message == "stop":
                    self.active = False
                elif message == "clear":
                    if hasattr(self.win, "clear"):
                        self.win.clear()
                elif message == "update_settings":
                    self.update_settings(**info)
                elif message == "execute":
                    try:
                        exec(info["code"])
                    except Exception:
                        print(traceback.format_exc())
                else:
                    raise RuntimeError()

            if self.show_1D_from_2D and self.samples is not None and self.counter_1D_from_2D < self.samples:
                self.update_plot_1D_from_2D(self.counter_1D_from_2D)
                self.counter_1D_from_2D += 1

            dt = time.perf_counter() - t0
            if self.interval - dt > 0:
                time.sleep(self.interval - dt)


    def initialize_plot(self, figsize):
        if not self.__class__.process:
            self._init_qt()
        try:
            win = self.rpg.GraphicsWindow(title="title")
        except ClosedError as err:
            logger.error("Closed error")
            # Remote process may have crashed. Trying to restart
            self._init_qt()
            win = self.rpg.GraphicsWindow(title="title")

        win.setWindowTitle("Oscilloscope")
        win.resize(*figsize)

        logger.info("Initialized plot")
        return win

    def update_settings(self, ylim, sample_rate, channels_settings, interval, show_legend):
        self.ylim = ylim
        self.sample_rate = sample_rate
        self.channels_settings = channels_settings
        self.interval = interval
        self.show_legend = show_legend

        # Update the legend now with current plots
        self.legend.setVisible(self.show_legend)
        if self.show_legend:
            # Check if any legend names have changed and update them if so.
            re_add = False
            for k, channel in enumerate(self.channels):
                curve = self.curves[k]
                channel_settings = self.channels_settings.get(channel, {})
                channel_name = channel_settings.get("name", channel)

                # Once we find one changed name, we want to re-add each successive trace
                # to retain the original order in the legend.
                if channel_name != curve.name():
                    re_add = True

                if re_add:
                    curve.setData(name=channel_name)
                    self.ax_1D.removeItem(curve)
                    self.ax_1D.addItem(curve)
            if re_add and self.points is not None:
                # If we had to re-add the plots to update the legend, then
                # re-draw the traces.
                self.update_plot_1D(self.points)

    @classmethod
    def _init_qt(cls):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        pg.mkQApp()
        cls.process = pgmp.QtProcess()  # pyqtgraph multiprocessing
        cls.rpg = cls.process._import("pyqtgraph")

    def format_axis(self, time_prefix=""):
        self.ax_1D.setLabels(left="Voltage (V)", bottom=f"Time ({time_prefix}s)")

    def update_plot_1D(self, points, **kwargs):
        self.points = points

        arr = self.np_array_1D[:, :points]

        t_list = np.arange(points) / self.sample_rate

        # Extract highest engineering exponent (-9, -6, -3) for rescaling
        max_exponent = np.log10(max(t_list))
        highest_engineering_exponent = int(max_exponent // 3 * 3)
        time_prefix = {-9: "n", -6: "u", -3: "m", 0: ""}[highest_engineering_exponent]
        t_list_scaled = t_list / 10 ** highest_engineering_exponent

        try:
            for k, channel in enumerate(self.channels):
                row = arr[k]
                curve = self.curves[k]
                channel_settings = self.channels_settings.get(channel, {})

                if channel_settings.get("scale") is not None:
                    row = row * channel_settings["scale"]
                if channel_settings.get("offset") is not None:
                    row = row + channel_settings["offset"]

                curve.setData(t_list_scaled, row)
                curve.setZValue(channel_settings.get("zorder", k))

            self.ax_1D.showGrid(x=True, y=True)
            self.ax_1D.disableAutoRange()
            self.ax_1D.setRange(xRange=(0, max(t_list_scaled)), yRange=self.ylim, padding=0)
            self.format_axis(time_prefix=time_prefix)

            self.log(f'updating 1D plot')

        except Exception:
            print(traceback.format_exc())

    def update_plot_2D(self, samples, points, **kwargs):
        self.samples = samples
        self.points = points

        channel_idx = self.channels.index(self.channel_plot_2D)
        arr = self.np_array_2D[channel_idx, :samples, :points]

        # PyQtGraph treats the first dimension as the x axis
        arr = arr.transpose()

        t_list = np.arange(points) / self.sample_rate * 1e3
        repetitions = np.arange(samples)

        # Extract highest engineering exponent (-9, -6, -3) for rescaling
        max_exponent = np.log10(max(t_list))
        highest_engineering_exponent = int(max_exponent // 3 * 3)
        time_prefix = {-9: "n", -6: "u", -3: "m", 0: ""}[highest_engineering_exponent]
        t_list_scaled = t_list / 10 ** highest_engineering_exponent

        try:
            self.img_2D.setImage(arr, **kwargs)

            self.ax_2D.setRange(xRange=(0, max(t_list)), yRange=(0, samples), padding=0)

            # self.ax_2D.disableAutoRange()
            # self.ax_2D.setRange(xRange=(0, max(t_list)), yRange=(0, samples))
            # self.ax_2D.getViewBox().setXRange(0, max(t_list))
            # self.ax_2D.vb.setLimits(xMin=0, xMax=max(t_list), yMin=0, yMax=samples)
            # self.ax_2D.setRange(xRange=[5,20])

            # curve = self.curves[k]
            # channel_settings = self.channels_settings.get(channel, {})
            #
            # if channel_settings.get("scale") is not None:
            #     row = row * channel_settings["scale"]
            # if channel_settings.get("offset") is not None:
            #     row = row + channel_settings["offset"]
            #
            # curve.setData(t_list_scaled, row)
            # curve.setZValue(channel_settings.get("zorder", k))

            # self.ax_2D.showGrid(x=True, y=True)
            # self.ax_2D.disableAutoRange()
            # self.ax_2D.setRange(xRange=(0, max(t_list_scaled)), yRange=self.ylim)
            # self.format_axis(time_prefix=time_prefix)

            self.log(f'updating 2D plot')

        except Exception:
            print(traceback.format_exc())

    def update_plot_1D_from_2D(self, sample_idx):
        sample_array = self.np_array_2D[:, sample_idx, :self.points]
        self.np_array_1D[:, :self.points] = sample_array
        self.update_plot_1D(points=self.points)
        self.log(f'updating 1D plot idx {sample_idx} from 2D trace')
