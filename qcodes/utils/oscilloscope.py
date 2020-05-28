import traceback
import numpy as np
import multiprocessing as mp
import logging
import time

import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
from pyqtgraph import QtGui
from pyqtgraph.multiprocess.remoteproxy import ClosedError


logger = logging.getLogger(__name__)

# https://stackoverflow.com/questions/17103698/plotting-large-arrays-in-pyqtgraph?rq=1


class Oscilloscope():
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
            channels_settings=None,
            figsize=(1200, 350),
            sample_rate=200e3,
            ylim=[-2, 2],
            interval=0.1
        ):
        self.max_points = max_points

        assert isinstance(channels, (list, tuple))
        self.channels = channels

        self.channels_settings = channels_settings or {}
        for channel in channels:
            self.channels_settings.setdefault(channel, {})

        self.shape = (len(channels), max_points)
        self.sample_rate = sample_rate
        self.ylim = ylim
        self.interval = interval

        self.mp_array = mp.RawArray('d', int(len(channels) * max_points))
        self.np_array = np.frombuffer(
            self.mp_array, dtype=np.float64
        ).reshape(self.shape)

        self.figsize = figsize

        self.queue = mp.Queue()

        self.process = None



    def start_process(self):
        self.process = mp.Process(
            target=OscilloscopeProcess,
            kwargs=dict(
                mp_array=self.mp_array,
                shape=self.shape,
                queue=self.queue,
                channels=self.channels,
                channels_settings=self.channels_settings,
                figsize=self.figsize,
                sample_rate=self.sample_rate,
                ylim=self.ylim,
                interval=self.interval
            )
        )
        self.process.start()

    def update_settings(self):
        self.queue.put({
            'message': 'update_settings',
            'ylim': self.ylim,
            'channels_settings': self.channels_settings,
            'sample_rate': self.sample_rate,
            'interval': self.interval
        })

    def update_array(self, array):
        assert len(array) == len(self.channels)

        if isinstance(array, dict):
            # Convert dict with an array per channel into a single array
            array = np.array(list(array.values()))

        points = array.shape[1]
        assert points <= self.max_points

        # Copy new array to shared array
        self.np_array[:,:points] = array

        self.queue.put({
            'message': 'new_trace',
            'points': points
        })

    def _run_code(self, code):
        self.queue.put({
            'message': 'execute',
            'code': code
        })


class OscilloscopeProcess():
    process = None
    rpg = None
    def __init__(self, mp_array, shape, queue, channels, channels_settings, figsize, sample_rate, ylim, interval):
        self.shape = shape
        self.queue = queue
        self.channels = channels
        self.channels_settings = channels_settings
        self.sample_rate = sample_rate
        self.ylim = ylim
        self.interval = interval
        self.mp_array = mp_array
        self.np_array = np.frombuffer(
            self.mp_array, dtype=np.float64
        ).reshape(self.shape)

        self.active = True

        self.win = self.initialize_plot()
        self.win.setWindowTitle('Oscilloscope')
        self.win.resize(*figsize)
        self.ax = self.win.addPlot()
        self.ax.disableAutoRange()
#         self.ax.showGrid(x=True, y=True, alpha=0.2)

        try:
            self.curves = [self.ax.plot(pen=(k, self.shape[0])) for k in range(self.shape[0])]
        except:
            print(traceback.format_exc())

        self.t_last_update = None
        self.process_loop()

#     def modify_channel(channel, offset=None, scale=None):

    def process_loop(self):
        while self.active:
            info = self.queue.get()
            message = info.pop('message')

            if message == 'new_trace':
                if (
                    self.t_last_update is None
                    or self.interval is None
                    or time.perf_counter() - self.t_last_update > self.interval
                ):
                    self.update_plot(**info)
            elif message == 'stop':
                self.active = False
            elif message == 'clear':
                if hasattr(self.win, 'clear'):
                    self.win.clear()
            elif message == 'update_settings':
                self.update_settings(**info)
            elif message == 'execute':
                try:
                    exec(info['code'])
                except Exception:
                    print(traceback.format_exc())
            else:
                raise RuntimeError()

    def initialize_plot(self):
        if not self.__class__.process:
            self._init_qt()
        try:
            win = self.rpg.GraphicsWindow(title='title')
        except ClosedError as err:
            logger.error('Closed error')
            # Remote process may have crashed. Trying to restart
            self._init_qt()
            win = self.rpg.GraphicsWindow(title='title')

        logger.info('Initialized plot')
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
        cls.rpg = cls.process._import('pyqtgraph')

    def format_axis(self, time_prefix=''):
        self.ax.setLabels(left='Voltage (V)', bottom=f'Time ({time_prefix}s)')

    def update_plot(self, points, **kwargs):
        arr = self.np_array[:,:points]

        t_list = np.arange(points) / self.sample_rate

        # Extract highest engineering exponent (-9, -6, -3) for rescaling
        max_exponent = np.log10(max(t_list))
        highest_engineering_exponent = int(max_exponent // 3 * 3)
        time_prefix = {-9: 'n', -6: 'u', -3: 'm', 0: ''}[highest_engineering_exponent]
        t_list_scaled = t_list / 10**highest_engineering_exponent

        try:
            for k, channel in enumerate(self.channels):
                row = arr[k]
                curve = self.curves[k]
                channel_settings = self.channels_settings.get(channel, {})

                if channel_settings.get('scale') is not None:
                    row = row * channel_settings['scale']
                if channel_settings.get('offset') is not None:
                    row = row + channel_settings['offset']

                curve.setData(t_list_scaled, row)
                curve.setZValue(channel_settings.get('zorder', k))

            self.ax.showGrid(x=True, y=True)
            self.ax.disableAutoRange()
            self.ax.setRange(xRange=(0, max(t_list_scaled)), yRange=self.ylim)
            self.format_axis(time_prefix=time_prefix)

            self.t_last_update = time.perf_counter()

        except Exception:
            print(traceback.format_exc())
