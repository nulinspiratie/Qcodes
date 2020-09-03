import traceback
import numpy as np
import multiprocessing as mp
import logging
import time
from collections import namedtuple, deque

import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
from pyqtgraph.multiprocess.remoteproxy import ClosedError

TransformState = namedtuple('TransformState', 'translate scale revisit')

logger = logging.getLogger(__name__)

# https://stackoverflow.com/questions/17103698/plotting-large-arrays-in-pyqtgraph?rq=1


class Oscilloscope:
    """Create an oscilloscope GUI that displays acquisition traces

    Example code:

    ```
    oscilloscope = Oscilloscope(channels=['chA', 'chB', 'chC', 'chD'])
    oscilloscope.start_process()

    # Define settings
    oscilloscope.ylim = (-0.8, 1)
    oscilloscope.channels_settings['chB']['scale'] = 10
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
        channel_plot_2D=None,
        show_1D_from_2D: bool = False
    ):
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
                channel_plot_2D=self.channel_plot_2D,
                show_1D_from_2D=self.show_1D_from_2D
            ),
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
            }
        )

    def update_array_1D(self, array):
        assert len(array) == len(self.channels)

        if isinstance(array, dict):
            # Convert dict with an array per channel into a single array
            array = np.array([array[ch] for ch in self.channels])

        points = array.shape[1]
        assert points <= self.max_points

        # Copy new array to shared array
        self.np_array_1D[:, :points] = array

        self.queue.put({"message": "new_trace_1D", "points": points})

    def update_array_2D(self, array):
        if isinstance(array, dict):
            # Convert dict with an array per channel into a single array
            array = np.array([array[ch] for ch in self.channels])

        channels, samples, points = array.shape

        assert channels == len(self.channels)
        assert samples <= self.max_samples
        assert points <= self.max_points

        # Copy new array to shared array
        self.np_array_2D[:, :samples, :points] = array

        self.queue.put(
            {"message": "new_trace_2D", "samples": samples, "points": points}
        )

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
        self.channel_plot_2D = channel_plot_2D
        self.show_1D_from_2D = show_1D_from_2D
        
        self.samples = None
        self.points = None

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
            self.img_2D.scale(1/self.sample_rate*1e3, 1)

            self.ax_2D.getAxis('bottom').setLabel('Time', 'ms')
            self.ax_2D.getAxis('left').setLabel('Repetition', '')

            self.ax_2D.addItem(self.img_2D)
        else:
            self.ax_1D = self.win.addPlot()
            self.ax_2D = None
            self.img_2D = None

        self.ax_1D.disableAutoRange()

        try:
            self.curves = [
                self.ax_1D.plot(pen=(k, self.shape_1D[0])) for k in range(self.shape_1D[0])
            ]
        except:
            print(traceback.format_exc())

        self.t_last_update = None
        self.process_loop()

    def process_loop(self):
        while self.active:
            if not self.queue.empty():
                info = self.queue.get()
                message = info.pop("message")

                if message == "new_trace_1D":
                    # Show a single trace
                    if (
                        self.t_last_update is None
                        or self.interval is None
                        or time.perf_counter() - self.t_last_update > self.interval
                    ):
                        self.update_plot_1D(**info)
                elif message == 'new_trace_2D':
                    # Show a 2D plot of traces
                    if (
                        self.t_last_update is None
                        or self.interval is None
                        or time.perf_counter() - self.t_last_update > self.interval
                    ):
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

            time.sleep(self.interval)

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

    def update_settings(self, ylim, sample_rate, channels_settings, interval):
        self.ylim = ylim
        self.sample_rate = sample_rate
        self.channels_settings = channels_settings
        self.interval = interval

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

            self.t_last_update = time.perf_counter()

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
            self.img_2D.setImage(arr)

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

            self.t_last_update = time.perf_counter()

        except Exception:
            print(traceback.format_exc())

    def update_plot_1D_from_2D(self, sample_idx):
        sample_array = self.np_array_2D[:, sample_idx, :self.points]
        self.np_array_1D[:, :self.points] = sample_array
        self.update_plot_1D(points=self.points)
