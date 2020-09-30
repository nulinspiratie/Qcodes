from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import *
from qcodes.instrument.parameter import Parameter

import ctypes
import numpy as np
from picosdk.ps3000a import ps3000a as picosdk
from picosdk.functions import adc2mV, assert_pico_ok
import time

def error_check(value, method_name=None):
    """Check if returned value after a set is an error code or not.

    Args:
        value: value to test.
        method_name: Name of called picosdk method, used for error message

    Raises:
        AssertionError if returned value is an error code.
    """
    assert isinstance(value, (str, bool, np.ndarray)) or (int(value) >= 0), \
        f'Error in call to picosdk.{method_name}, error code {value}'


def with_error_check(fun):
    @wraps(fun)
    def error_check_wrapper(*args, **kwargs):
        value = fun(*args, **kwargs)
        error_check(value, f'Error calling {fun.__name__} with args {kwargs}. '
                           f'Return value = {value}')
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
    def __init__(self,
                 name: str,
                 parent: Union[Instrument, InstrumentChannel] = None,
                 get_cmd: Callable = None,
                 get_function: Callable = None,
                 set_cmd: Callable = False,
                 set_function: Callable = None,
                 set_args: List[str] = None,
                 initial_value=None,
                 **kwargs):
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
                return_val = self.set_function(self.parent.id, *set_vals)
            else:
                return_val = self.set_function(*set_vals)
            # Check if the returned value is an error
            method_name = self.set_function.__func__.__name__
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
        self.enabled = PicoParameter('enabled',
                                     initial_value=True,
                                     vals=Bool(),
                                     set_function=picosdk.ps300aSetChannel,
                                     set_args=['type', 'range', 'analogue_offset'])

        self.type = PicoParameter('type',
                                     initial_value='DC',
                                     vals=Enum(**picosdk.PICO_COUPLING),
                                     set_function=picosdk.ps300aSetChannel,
                                     set_args=['enabled', 'range', 'analogue_offset'])

        self.range = PicoParameter('range',
                                  initial_value=True,
                                  vals=Enum(**picosdk.PICO_VOLTAGE_RANGE),
                                  set_function=picosdk.ps300aSetChannel,
                                  set_args=['enabled', 'type', 'analogue_offset'])

        self.analogue_offset = PicoParameter('analogue_offset',
                                   initial_value=True,
                                   vals=Numbers(),
                                   # get_function=picosdk.ps3000aGetAnalogueOffset,
                                   set_function=picosdk.ps300aSetChannel,
                                   set_args=['enabled', 'type', 'range'])

    def add_parameter(self, name: str, parameter_class: type=PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class, parent=self, **kwargs)



class PicoScope(Instrument):
    def add_parameter(self, name: str, parameter_class: type=PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class, parent=self, **kwargs)

    def __init__(self, name, **kwargs):
        self.channels = ['A', 'B', 'C', 'D']
        self.buffers = {ch : None for ch in self.channels}
        # Create chandle and status ready for use
        self._chandle = ctypes.c_int16()
        status = {}


        # Open PicoScope 3000 Series device
        status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(self._chandle), None)
        try:
            assert_pico_ok(status["openunit"])
        except:  # PicoNotOkError:

            powerStatus = status["openunit"]

            if powerStatus == 286:
                status["changePowerSource"] = ps.ps3000aChangePowerSource(self._chandle, powerStatus)
            elif powerStatus == 282:
                status["changePowerSource"] = ps.ps3000aChangePowerSource(self._chandle, powerStatus)
            else:
                raise

            assert_pico_ok(status["changePowerSource"])



        self.initialize_driver()
        super().__init__(name, **kwargs)

    def initialize_driver(self):
        analogue_offset = 0.0
        channel_range = ps.PS3000A_RANGE['PS3000A_2V']
        status = {}
        for ch in self.channels:
            status[f"setCh{ch}"] = ps.ps3000aSetChannel(self._chandle,
                                                    ps.PS3000A_CHANNEL[f'PS3000A_CHANNEL_{ch}'],
                                                    True, #enable
                                                    ps.PS3000A_COUPLING['PS3000A_DC'],
                                                    channel_range,
                                                    analogue_offset)
            assert_pico_ok(status[f"setCh{ch}"])
        return status

    def initialize_buffers(self, buffer_size, num_buffers):
        status = {}
        # Create buffers ready for assigning pointers for data collection
        self._raw_buffers = {ch: np.zeros(shape=(num_buffers, buffer_size), dtype=np.int16) for ch in self.channels}

        memory_segment = 0

        # Set data buffer location for data collection from channel A
        # handle = chandle
        # source = PS3000A_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segment index = 0
        # ratio mode = PS3000A_RATIO_MODE_NONE = 0
        for ch in self.channels:
            status[f"setDataBuffers{ch}"] = ps.ps3000aSetDataBuffers(self._chandle,
                                                                     ps.PS3000A_CHANNEL[f'PS3000A_CHANNEL_{ch}'],
                                                                     self._raw_buffers[ch].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                                     None,
                                                                     buffer_size,
                                                                     memory_segment,
                                                                     ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'])
            assert_pico_ok(status[f"setDataBuffers{ch}"])
        return status

    def begin_streaming(self):
        status = {}

        sample_interval = ctypes.c_int32(250)
        sample_units = ps.PS3000A_TIME_UNITS['PS3000A_US']
        # We are not triggering:
        max_pre_trigger_samples = 0
        auto_stop_on = True
        # No downsampling:
        downsample_ratio = 1
        total_samples = max([np.product(buf.shape) for buf in self._raw_buffers.values()])
        buffer_size = max([buf.shape[-1] for buf in self._raw_buffers.values()])
        print(f"Total samples = {total_samples}, buffer_size = {buffer_size}")
        status["runStreaming"] = ps.ps3000aRunStreaming(self._chandle,
                                                        ctypes.byref(sample_interval),
                                                        sample_units,
                                                        max_pre_trigger_samples,
                                                        total_samples,
                                                        auto_stop_on,
                                                        downsample_ratio,
                                                        ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'],
                                                        buffer_size)
        assert_pico_ok(status["runStreaming"])

        actual_sample_interval = sample_interval.value
        actual_sample_interval_ns = actual_sample_interval * 1000

        print("Capturing at sample interval %s ns" % actual_sample_interval_ns)

        # We need a big buffer, not registered with the driver, to keep our complete capture in.
        self.buffers = {ch : np.zeros(shape=total_samples, dtype=np.int16) for ch in self.channels}
        next_sample = 0
        auto_stop_outer = False
        was_called_back = False


        def streaming_callback(mydickt, handle, num_samples, startIndex, overflow, triggerAt, triggered, auto_stop, param):
            next_sample = mydickt['next_sample']
            auto_stop_outer = mydickt['auto_stop_outer']
            was_called_back = mydickt['was_called_back']
            # global next_sample, auto_stop_outer, was_called_back
            was_called_back = True
            destEnd = next_sample + num_samples
            sourceEnd = startIndex + num_samples
            for ch in self.channels:
                self.buffers[ch][next_sample:destEnd] = self._raw_buffers[ch].flatten()[startIndex:sourceEnd]

            next_sample += num_samples
            if auto_stop:
                auto_stop_outer = True

            mydickt['next_sample'] = next_sample
            mydickt['auto_stop_outer'] = auto_stop_outer
            mydickt['was_called_back'] = was_called_back

        from functools import partial
        mydickt = dict(next_sample=next_sample, auto_stop_outer=auto_stop_outer, was_called_back=was_called_back)
        # Convert the python function into a C function pointer.
        c_callback = ps.StreamingReadyType(partial(streaming_callback, mydickt))


        mydickt['next_sample'] = next_sample
        mydickt['auto_stop_outer'] = auto_stop_outer
        # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
        while mydickt['next_sample'] < total_samples and not auto_stop_outer:
            mydickt['was_called_back'] = False
            status["getStreamingLastestValues"] = ps.ps3000aGetStreamingLatestValues(self._chandle, c_callback, None)
            if not was_called_back:
                # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                # again.
                time.sleep(0.01)

        print("Done grabbing values.")
        return status

    def process_data(self):
        status = {}
        total_samples = max([np.product(buf.shape) for buf in self._raw_buffers])
        channel_range = ps.PS3000A_RANGE['PS3000A_2V']
        sample_interval = ctypes.c_int32(250)
        actual_sample_interval = sample_interval.value
        actual_sample_interval_ns = actual_sample_interval * 1000

        # Find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        max_adc = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(self._chandle, ctypes.byref(max_adc ))
        assert_pico_ok(status["maximumValue"])

        # Convert ADC counts data to mV
        scaled_data = {}
        for ch in self.channels:
            scaled_data[ch] = adc2mV(self.buffers[ch], channel_range, max_adc)

        # Create time data
        time = np.linspace(0, (total_samples) * actual_sample_interval_ns, total_samples)
        return time, scaled_data
