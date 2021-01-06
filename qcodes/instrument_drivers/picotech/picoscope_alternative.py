import numpy as np
import logging
import time
from typing import Callable, List, Union
from matplotlib import pyplot as plt
from functools import partial, wraps
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import MultiParameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import Parameter
from qcodes import MatPlot
from qcodes.utils.helpers import PerformanceTimer

from picoscope import *
from picoscope import ps3000a as ps


logger = logging.getLogger(__name__)

def error_check(value, method_name=None):
    """Check if returned value after a set is an error code or not.

    Args:
        value: value to test.
        method_name: Name of called picosdk method, used for error message

    Raises:
        AssertionError if returned value is an error code.
    """
    assert isinstance(value, (str, bool, np.ndarray)) or (
        int(value) >= 0
    ), f"Error in call to picoscope.ps3000a.{method_name}, error code {value}"


def with_error_check(fun):
    @wraps(fun)
    def error_check_wrapper(*args, **kwargs):
        value = fun(*args, **kwargs)
        error_check(
            value,
            f"Error calling {fun.__name__} with args {kwargs}. "
            f"Return value = {value}",
        )
        return value

    return error_check_wrapper


class PicoParameter(Parameter):
    """Picoscope parameter designed to send picosdk commands.

    This parameter can function as a standard parameter, but can also be
    associated with a specific picosdk function.

    Args:
        name: Parameter name
        parent: Picoscope Instrument or instrument channel
            In case of a channel, it should have an id between 1 and n_channels.
        get_cmd: Standard optional Parameter get function
        get_function: picosdk function to be called when getting the
            parameter value. If set, get_cmd must be None.
        set_cmd: Standard optional Parameter set function
        set_function: picosdk function to be called when setting the
            parameter value. If set, set_cmd must be None.
        set_args: Optional ancillary parameter names that are passed to
            set_function. Some picosdk functions need to pass
            multiple parameters simultaneously. If set, the name of this
            parameter must also be included in the appropriate index.
        initial_value: initial value for the parameter. This does not actually
            perform a set, so it does not call any picosdk function.
        **kwargs: Additional kwargs passed to Parameter
    """

    def __init__(
        self,
        name: str,
        parent: Union[Instrument, InstrumentChannel] = None,
        get_cmd: Callable = None,
        get_function: Callable = None,
        set_cmd: Callable = False,
        set_function: Callable = None,
        set_args: List[str] = None,
        initial_value=None,
        **kwargs,
    ):
        self.get_cmd = get_cmd
        self.get_function = get_function

        self.set_cmd = set_cmd
        self.set_function = set_function
        self.set_args = set_args

        super().__init__(name=name, parent=parent, **kwargs)

        if initial_value is not None:
            # We set the initial value here to ensure that it does not call
            # the set_raw method the first time
            if self.val_mapping is not None:
                initial_value = self.val_mapping[initial_value]
            self._save_val(initial_value)
            self.raw_value = initial_value

    def set_raw(self, val):
        if self.set_cmd is not False:
            if self.set_cmd is not None:
                return self.set_cmd(val)
            else:
                return
        elif self.set_function is not None:
            if self.set_args is None:
                set_vals = [val]
            else:
                # Convert set args, which are parameter names, to their
                # corresponding parameter values
                set_vals = []
                for set_arg in self.set_args:
                    if set_arg == self.name:
                        set_vals.append(val)
                    else:
                        # Get the current value of the parameter
                        set_vals.append(getattr(self.parent, set_arg).raw_value)

            # Evaluate the set function with the necessary set parameter values
            if isinstance(self.parent, InstrumentChannel):
                # Also pass the channel id
                return_val = self.set_function(
                    self.parent.id,
                    *set_vals,
                )
            else:
                return_val = self.set_function(self.parent._chandle, *set_vals)
            # Check if the returned value is an error
            method_name = self.set_function.__name__
            error_check(return_val, method_name=method_name)
        else:
            # Do nothing, value is saved
            pass

    def get_raw(self):
        if self.get_cmd is not None:
            return self.get_cmd()
        elif self.get_function is not None:
            if isinstance(self.parent, InstrumentChannel):
                return self.get_function(self.parent.id)
            else:
                return self.get_function()
        else:
            return self.get_latest(raw=True)

class PicoAcquisitionParameter(MultiParameter):
    def __init__(self, instrument, **kwargs):
        self.instrument = instrument
        super().__init__(snapshot_value=False,
                         names=[''], shapes=[()], **kwargs)

    @property
    def names(self):
        if self.instrument is None or \
                not hasattr(self.instrument, 'active_channels')\
                or self.instrument.active_channels is None:
            return ['']
        else:
            return tuple(['ch{}_signal'.format(ch.id) for ch in
                          self.instrument.active_channels])

    @names.setter
    def names(self, names):
        # Ignore setter since getter is extracted from instrument
        pass

    @property
    def labels(self):
        return self.names

    @labels.setter
    def labels(self, labels):
        # Ignore setter since getter is extracted from instrument
        pass

    @property
    def units(self):
        return ['V'] * len(self.names)

    @units.setter
    def units(self, units):
        # Ignore setter since getter is extracted from instrument
        pass

    @property
    def shapes(self):
        if hasattr(self.instrument, 'average_mode'):
            average_mode = self.instrument.average_mode()

            if average_mode == 'point':
                shape = ()
            elif average_mode == 'trace':
                shape = (self.instrument.samples(),)
            else:
                shape = (self.instrument.samples(),
                         self.instrument.points_per_trace())
            return tuple([shape] * len(self.instrument.active_channels))
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get_raw(self):
        raw_data = self.instrument.acquisition()
        if self.instrument.average_mode() == 'point':
            return raw_data.mean()
        if self.instrument.average_mode() == 'trace':
            return raw_data.mean(axis=0)
        else:
            return raw_data


class ScopeChannel(InstrumentChannel):
    """Picoscope channel

    Args:
        parent: Parent Picoscope instrument
        name: channel name (e.g. 'chA')
        id: channel id (e.g. 1)
        **kwargs: Additional kwargs passed to InstrumentChannel
    """

    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

        self.id = id
        self.enabled = PicoParameter(
            "enabled",
            initial_value=True,
            vals=vals.Bool(),
            set_function=parent.setChannel,
            set_args=["coupling", "range", "offset", "enabled"],
        )

        self.coupling = PicoParameter(
            "coupling",
            initial_value="DC",
            vals=vals.Enum(*parent.CHANNEL_COUPLINGS.keys()),
            set_function=parent.setChannel,
            set_args=["coupling", "range", "offset", "enabled"],
        )

        self.range = PicoParameter(
            "range",
            initial_value=2, # 2 V range
            vals=vals.Enum(*[option['rangeV'] for option in parent.CHANNEL_RANGE]),
            set_function=parent.setChannel,
            set_args=["coupling", "range", "offset", "enabled"],
        )

        self.offset = PicoParameter(
            "offset",
            initial_value=0,
            vals=vals.Numbers(),
            set_function=parent.setChannel,
            set_args=["coupling", "range", "offset", "enabled"],
        )

    def add_parameter(self, name: str, parameter_class: type = PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(
            name=name, parameter_class=parameter_class, parent=self, **kwargs
        )


class PicoScope(Instrument, ps.PS3000a):
    channel_names = ["A", "B", "C", "D"]

    def __init__(self, name='picoscope', serialNumber=None, connect=True, **kwargs):
        """

        Args:
            name:
                The name of the instrument.
            serialNumber:
                Passed to the picoscope API to locate the instrument
            connect: Bool
                When True, opens the instrument automatically. If False, you
                will need to manually connect by using self.open(serialNumber).
            **kwargs:
        """
        Instrument.__init__(self, name, **kwargs)
        ps.PS3000a.__init__(self, serialNumber=serialNumber, connect=connect, **kwargs)


        self._raw_buffers = {}
        self.buffers = {ch: None for ch in self.channel_names}

        # Define channels
        channels = ChannelList(self, name="channels", chan_type=ScopeChannel)
        for ch in self.channel_names:
            channel = ScopeChannel(self, name=f"ch{ch}", id=ch)
            setattr(self, f"ch{ch}", channel)
            channels.append(channel)

        self.add_submodule("channels", channels)

        self.initialize_channels()


        self.points_per_trace = Parameter(set_cmd=None, initial_value=1000, set_parser=lambda x: int(round(x)))
        self.samples = Parameter(set_cmd=None, initial_value=10, set_parser=lambda x: int(round(x)))
        self.sample_rate = Parameter(set_cmd=None, initial_value=500e3, set_parser=lambda x: int(round(x)))
        self.average_mode = Parameter(set_cmd=None, get_cmd=None, vals=vals.Enum('point', 'trace', None))

        # Trigger parameters
        self.use_trigger = Parameter(set_cmd=None, initial_value=False, vals=vals.Bool())
        self.trigger_channel = Parameter(
            set_cmd=None, initial_value='External',
            vals=vals.Enum(*[ch.id for ch in self.channels], 'External'),
            # val_mapping={
            #     **{ch:ch for ch in self.channels},
            #     'external': ps.PS3000A_CHANNEL[f'PS3000A_EXTERNAL']
            # }
        )
        self.trigger_threshold = Parameter(set_cmd=None, unit='V', initial_value=0.5, vals=vals.Numbers())
        self.trigger_direction = Parameter(
            set_cmd=None, initial_value='Rising', vals=vals.Enum(*self.THRESHOLD_TYPE)
        )
        self.trigger_delay = Parameter(set_cmd=None, initial_value=0, unit='s')
        self.autotrigger_ms = Parameter(set_cmd=None, initial_value=0, unit='ms',
                                        docstring='milliseconds to wait after trigger. 0 means wait indefinitely')
        self.buffer_actions = []
        self.timings = PerformanceTimer()

    @property
    def active_channels(self):
        return [ch for ch in self.channels if ch.enabled()]

    @property
    def active_channel_ids(self):
        return [ch.id for ch in self.active_channels]

    def close(self):
        self.stop()
        ps.PS3000a.close(self)
        Instrument.close(self)

    def stop(self):
        ps.PS3000a.stop(self)

    def add_parameter(self, name: str, parameter_class: type = PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(
            name=name, parameter_class=parameter_class, parent=self, **kwargs
        )

    def initialize_channels(self):
        """Initialize picoscope channels (enable, coupling, range, offset)"""
        for ch in self.channels:
            #  specifies whether an input channel is to be enabled, its input coupling type, voltage range and analog offset.
            self.setChannel(
                channel=ch.id,
                enabled=ch.enabled.raw_value,
                coupling=ch.coupling.raw_value,
                VRange=ch.range.raw_value,
                VOffset=ch.offset.raw_value,
            )

    def initialize_buffers(self):
        """Initializes buffers (number of buffers and buffer size)"""
        # Create buffers ready for assigning pointers for data collection
        self._raw_buffers = {
            ch.id: np.zeros(shape=(self.samples(), self.points_per_trace()), dtype=np.int16)
            for ch in self.active_channels
        }
        self.buffers = {ch.id: None for ch in self.active_channels} # empty for now

    def setup_block_capture(self):
        self.setSamplingFrequency(self.sample_rate(), self.points_per_trace())
        # self.setSimpleTrigger("A", threshold_V=0.5)

        samples_per_segment = self.memorySegments(self.samples())
        self.setNoOfCaptures(self.samples())
        self.runBlock()

    def get_data(self):
        with self.timings.record('wait_for_acquisition'):
            self.waitReady()

        for ch in self.active_channels:
            self.getDataRawBulk(ch.id, data=self._raw_buffers[ch.id])

    def setup_trigger(self):
        trigger_channel_id = self.trigger_channel()
        self.setSimpleTrigger(trigSrc=trigger_channel_id,
                              enabled=True,
                              threshold_V=self.trigger_threshold(),
                              direction=self.trigger_direction(),
                              delay=self.trigger_delay(),
                              timeout_ms=self.autotrigger_ms())

    def process_data(self):
        # Convert ADC counts data to V
        for ch in self.active_channels:
            if self.buffers[ch.id] is not None and self.buffers[ch.id].shape == self._raw_buffers[ch.id].shape:
                # Save memory by passing the already created buffers
                self.rawToV(ch.id, dataRaw=self._raw_buffers[ch.id], dataV=self.buffers[ch.id])
            else:
                self.buffers[ch.id] = np.reshape(self.rawToV(ch.id, self._raw_buffers[ch.id]),
                                                 newshape=(self.samples(), self.points_per_trace()))
        for buffer_action in self.buffer_actions:
            buffer_action(self.buffers)

        return self.buffers

    def plot_traces(self):
        plot = MatPlot(subplots=len(self.buffers))
        t_list = np.arange(self.points_per_trace(), 1/self.sample_rate())
        samples = np.arange(self.samples(), dtype=float)
        for k, (ch, buffer) in enumerate(self.buffers.items()):
            plot[k].add(buffer, x=t_list, y=samples)

            plot[k].set_title(f'Channel {ch}')

    def acquisition(self):
        with self.timings.record('setup_trigger'):
            self.setup_trigger()
        with self.timings.record('setup_block_capture'):
            self.setup_block_capture()
        with self.timings.record('get_data'):
            self.get_data()
        with self.timings.record('process_data'):
            self.process_data()
        return self.buffers
