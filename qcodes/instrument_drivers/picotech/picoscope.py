import numpy as np
import logging
import time
from typing import Callable, List, Union
from matplotlib import pyplot as plt
from functools import partial, wraps
import ctypes

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import Parameter
from qcodes import MatPlot
from qcodes.utils.helpers import PerformanceTimer

from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.constants import PICO_STATUS_LOOKUP


logger = logging.getLogger(__name__)


# REMOVEME
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok



def adc2mV(bufferADC, range, maxADC, inverse=False):
    """Takes a buffer of raw adc count values and converts it into millivolts"""
    if isinstance(bufferADC, (list, tuple)):
        bufferADC = np.array(bufferADC)

    channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    vRange = channelInputRanges[range]
    if inverse:
        value = int(bufferADC / (vRange / maxADC.value))
    else:
        value = bufferADC * (vRange / maxADC.value)
    # bufferV = [(x * vRange) / maxADC.value for x in bufferADC]
    return value



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
    ), f"Error in call to picosdk.{method_name}, error code {value}"


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
                    self.parent.parent._chandle,
                    ps.PS3000A_CHANNEL[f"PS3000A_CHANNEL_{self.parent.id}"],
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
            set_function=ps.ps3000aSetChannel,
            set_args=["enabled", "coupling", "range", "offset"],
        )

        self.coupling = PicoParameter(
            "coupling",
            initial_value="DC",
            val_mapping=ps.PICO_COUPLING,
            set_function=ps.ps3000aSetChannel,
            set_args=["enabled", "coupling", "range", "offset"],
        )

        self.range = PicoParameter(
            "range",
            initial_value="PS3000A_2V",
            val_mapping=ps.PS3000A_RANGE,
            set_function=ps.ps3000aSetChannel,
            set_args=["enabled", "coupling", "range", "offset"],
        )

        self.offset = PicoParameter(
            "offset",
            initial_value=0,
            vals=vals.Numbers(),
            # get_function=picosdk.ps3000aGetAnalogueOffset,
            set_function=ps.ps3000aSetChannel,
            set_args=["enabled", "coupling", "range", "offset"],
        )

    def add_parameter(self, name: str, parameter_class: type = PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(
            name=name, parameter_class=parameter_class, parent=self, **kwargs
        )


class PicoScope(Instrument):
    # Create chandle and status ready for use
    _chandle = ctypes.c_int16()
    channel_names = ["A", "B", "C", "D"]

    def __init__(self, name='picoscope', **kwargs):
        super().__init__(name, **kwargs)

        self.buffers = {ch: None for ch in self.channel_names}

        # Open PicoScope 3000 Series device
        self.initialize_driver()

        # Define channels
        channels = ChannelList(self, name="channels", chan_type=ScopeChannel)
        for ch in self.channel_names:
            channel = ScopeChannel(self, name=f"ch{ch}", id=ch)
            setattr(self, f"ch{ch}", channel)
            channels.append(channel)

        self.add_submodule("channels", channels)

        self.initialize_channels()

        self._raw_single_buffers = {}
        self._raw_buffers = {}

        self.points_per_trace = Parameter(set_cmd=None, initial_value=1000)
        self.samples = Parameter(set_cmd=None, initial_value=10)
        self.sample_rate = Parameter(set_cmd=None, initial_value=500e3)

        # Trigger parameters
        self.use_trigger = Parameter(set_cmd=None, initial_value=False, vals=vals.Bool())
        self.trigger_channel = Parameter(
            set_cmd=None, initial_value='external',
            val_mapping={
                **{ch.id: ps.PS3000A_CHANNEL[f'PS3000A_CHANNEL_{ch.id}'] for ch in self.channels},
                'external': ps.PS3000A_CHANNEL[f'PS3000A_EXTERNAL']
            }
        )
        self.trigger_threshold = Parameter(set_cmd=None, unit='V', initial_value=0.5, vals=vals.Numbers())
        self.trigger_direction = Parameter(
            set_cmd=None, initial_value='rising', val_mapping={
                key: ps.PS3000A_THRESHOLD_DIRECTION[f'PS3000A_{key.upper()}']
                for key in ['above', 'below', 'rising', 'falling', 'rising_or_falling']
            }
        )
        self.trigger_delay = Parameter(set_cmd=None, initial_value=0, unit='s')
        self.autotrigger_ms = Parameter(set_cmd=None, initial_value=0, unit='ms',
                                        docstring='milliseconds to wait after trigger. 0 means wait indefinitely')


        # Delay between successive fetches
        self.stream_fetch_pause = Parameter(set_cmd=None, initial_value=0.002)

        self.timings = PerformanceTimer()

    @property
    def active_channels(self):
        return [ch for ch in self.channels if ch.enabled()]

    @property
    def active_channel_ids(self):
        return [ch.id for ch in self.active_channels]

    def close(self):
        status = {}
        self.stop()

        # Closes the unit
        # Handle = chandle
        status["close"] = ps.ps3000aCloseUnit(self._chandle)
        assert_pico_ok(status["close"])
        super().close()

    def stop(self):
        status = {}
        status["stop"] = ps.ps3000aStop(self._chandle)
        assert_pico_ok(status["stop"])

    def add_parameter(self, name: str, parameter_class: type = PicoParameter, **kwargs):
        """Use PicoParameter by default"""
        super().add_parameter(
            name=name, parameter_class=parameter_class, parent=self, **kwargs
        )

    def initialize_driver(self):
        """Open PicoScope 3000 Series device"""
        status = {}
        status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(self._chandle), None)
        try:
            assert_pico_ok(status["openunit"])
        except:  # PicoNotOkError:

            powerStatus = status["openunit"]

            if powerStatus == 286:
                status["changePowerSource"] = ps.ps3000aChangePowerSource(
                    self._chandle, powerStatus
                )
            elif powerStatus == 282:
                status["changePowerSource"] = ps.ps3000aChangePowerSource(
                    self._chandle, powerStatus
                )
            else:
                raise
        assert_pico_ok(status["changePowerSource"])
        return status

    def initialize_channels(self):
        """Initialize picoscope channels (enable, coupling, range, offset)"""
        status = {}
        for ch in self.channels:
            #  specifies whether an input channel is to be enabled, its input coupling type, voltage range and analog offset.
            channel_status = ps.ps3000aSetChannel(
                self._chandle,
                ps.PS3000A_CHANNEL[f"PS3000A_CHANNEL_{ch.id}"],
                ch.enabled.raw_value,  # enable channel
                ch.coupling.raw_value,  # DC coupling
                ch.range.raw_value,  # Set channel range
                ch.offset.raw_value,  # DC offset
            )

            assert_pico_ok(channel_status)
            status[ch.name] = PICO_STATUS_LOOKUP[channel_status]
        return status

    def initialize_buffers(self):
        """Initializes buffers (number of buffers and buffer size)"""
        # Create buffers ready for assigning pointers for data collection
        self._raw_buffers = {
            ch.id: np.zeros(shape=(self.samples(), self.points_per_trace()), dtype=np.int16)
            for ch in self.active_channels
        }

    def post_buffers(self):
        """Send buffer locations to the picoscope ready for data transfer"""
        status = {}
        for ch in self.channels:
            status[ch.name] = {}
            for i in range(self.samples()):
            # for single_buffer in self._raw_buffers[ch.id]:
                buffer_status = ps.ps3000aSetDataBuffers(
                    self._chandle,
                    ps.PS3000A_CHANNEL[f"PS3000A_CHANNEL_{ch.id}"],
                    self._raw_buffers[ch.id][i].ctypes.data,
                    None,
                    self.points_per_trace(),
                    i,
                    ps.PS3000A_RATIO_MODE["PS3000A_RATIO_MODE_NONE"],
                )
                assert_pico_ok(buffer_status)
                status[ch.name][f'buffer[{i}]'] = PICO_STATUS_LOOKUP[buffer_status], self._raw_buffers[ch.id][i].ctypes.data
            # status[ch.name] = PICO_STATUS_LOOKUP[channel_status]
        return status

    def setup_block_capture(self):
        status = {}
        preTriggerSamples = self.points_per_trace()//2
        postTriggerSamples = self.points_per_trace() - preTriggerSamples

        timebase = 2 # 4 ns?

        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32() # in the example given this is a short, but it should be a long
        status["GetTimebase"] = ps.ps3000aGetTimebase2(self._chandle, timebase, self.points_per_trace(), ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(status["GetTimebase"])
        status["timeIntervalns"] = timeIntervalns
        status["returnedMaxSamples"] = returnedMaxSamples

        cmaxSamples = ctypes.c_int32(self.points_per_trace())
        status["MemorySegments"] = ps.ps3000aMemorySegments(self._chandle, self.samples(), ctypes.byref(cmaxSamples))
        assert_pico_ok(status["MemorySegments"])
        status["cmaxSamples"] = cmaxSamples

        # sets number of captures
        status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(self._chandle, self.samples())
        assert_pico_ok(status["SetNoOfCaptures"])

        status["runblock"] = ps.ps3000aRunBlock(self._chandle, preTriggerSamples, postTriggerSamples, timebase,
                                                1, None, 0, None, None)
        assert_pico_ok(status["runblock"])
        return status

    def get_blocks(self):
        status = {}

        ready = ctypes.c_int16(0)
        print("Waiting for acquisition to finish...")
        while not ready:
            status["isReady"] = ps.ps3000aIsReady(self._chandle, ctypes.byref(ready))

        print("Acqusition complete.")
        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * self.samples())()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(self.points_per_trace())
        print("Getting bulk values.")
        status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self._chandle, ctypes.byref(cmaxSamples), 0, self.samples()-1, 1, 0, ctypes.byref(overflow))
        assert_pico_ok(status["GetValuesBulk"])
        print(status["GetValuesBulk"])


        Times = (ctypes.c_int16*self.samples())()
        TimeUnits = ctypes.c_char()
        status["GetValuesTriggerTimeOffsetBulk"] = ps.ps3000aGetValuesTriggerTimeOffsetBulk64(self._chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0, self.samples()-1)
        assert_pico_ok(status["GetValuesTriggerTimeOffsetBulk"])
        return status


    def setup_buffers(self, num_buffers, buffer_size, memory_segment=0):
        """Initializes buffers (number of buffers and buffer size)"""
        status = {}

        totalSamples = buffer_size * num_buffers

        # Set data buffer location for data collection
        # handle = chandle
        # source = PS3000A_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segment index = 0
        # ratio mode = PS3000A_RATIO_MODE_NONE = 0
        for ch in self.channels:
            channel_status = ps.ps3000aSetDataBuffers(
                self._chandle,
                ps.PS3000A_CHANNEL[f"PS3000A_CHANNEL_{ch.id}"],
                self._raw_single_buffers[ch.id].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                buffer_size,
                memory_segment,
                ps.PS3000A_RATIO_MODE["PS3000A_RATIO_MODE_NONE"],
            )
            assert_pico_ok(channel_status)
            status[ch.name] = PICO_STATUS_LOOKUP[channel_status]
        return status

    def setup_trigger(self):
        status = {}

        # Determine threshold ADC value
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(self._chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])
        trigger_channel_id = self.trigger_channel()
        if len(trigger_channel_id) == 1:
            trigger_channel = self.channels[f'ch{trigger_channel_id}']

            threshold_adc = adc2mV(self.trigger_threshold(), trigger_channel.range.raw_value, maxADC, inverse=True)
            logger.debug(f'Trigger threshold adc: {threshold_adc}')
        else: # external trigger
            # The external trigger range is always fixed to +- 5 V
            # https://www.picotech.com/support/viewtopic.php?t=36471
            threshold_adc = adc2mV(self.trigger_threshold(), ps.PS3000A_RANGE["PS3000A_5V"], maxADC, inverse=True)
            logger.debug(f'Trigger threshold adc: {threshold_adc}')

        # Sets up simple trigger
        # Handle = Chandle
        # Enable = 0
        # Source = ps3000A_channel_B = 0
        # Threshold = 1024 ADC counts
        # Direction = ps3000A_Falling = 3
        # Delay = 0
        # autoTrigger_ms = 1000
        trigger_status = ps.ps3000aSetSimpleTrigger(
            self._chandle,
            self.use_trigger.raw_value,
            self.trigger_channel.raw_value,  # convert to channel idx
            threshold_adc, # convert to int
            self.trigger_direction.raw_value,  # rising, falling, rising_or_falling, above, below,
            int(self.trigger_delay.raw_value * self.sample_rate()),  # number of sample periods
            self.autotrigger_ms.raw_value
        )
        assert_pico_ok(trigger_status)

    # def begin_block_capture(self):
    #     status = {}
    #     totalSamples = self.samples() * self.points_per_trace()
    #     preTriggerSamples = self.points_per_trace()//2
    #     postTriggerSamples = self.points_per_trace() - preTriggerSamples
    #     timebase = 2 # 4 ns?
    #
    #     with self.timings.record('start_block_capture'):
    #         status["runblock"] = ps.ps3000aRunBlock(self._chandle, preTriggerSamples, postTriggerSamples, timebase,
    #                                                 1, None, 0, None, None)
    #     assert_pico_ok(status["runblock"])

    streaming_info = {}
    def begin_streaming(self):
        status = {}

        totalSamples = self.samples() * self.points_per_trace()

        with self.timings.record('setup_buffers'):
            self.setup_buffers(num_buffers=self.samples(), buffer_size=self.points_per_trace(), memory_segment=0)

        # Begin streaming mode:
        with self.timings.record('begin_streaming'):
            interval = 1 / self.sample_rate()
            sampleInterval = ctypes.c_int32(int(interval * 1e9))
            sampleUnits = ps.PS3000A_TIME_UNITS['PS3000A_NS']
            # We are not triggering:
            maxPreTriggerSamples = 0
            autoStopOn = 1
            # No downsampling:
            downsampleRatio = 1
            status["runStreaming"] = ps.ps3000aRunStreaming(self._chandle,
                                                            ctypes.byref(sampleInterval),
                                                            sampleUnits,
                                                            maxPreTriggerSamples,
                                                            totalSamples,
                                                            autoStopOn,
                                                            downsampleRatio,
                                                            ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'],
                                                            self.points_per_trace())
            assert_pico_ok(status["runStreaming"])

            actualSampleInterval = sampleInterval.value
            self.streaming_info['actualSampleIntervalNs'] = actualSampleInterval

            logger.debug("Capturing at sample interval %s ns" % self.streaming_info['actualSampleIntervalNs'])

            self.streaming_info['nextSample'] = 0
            self.streaming_info['autoStopOuter'] = False
            self.streaming_info['wasCalledBack'] = False

            # Convert the python function into a C function pointer.
            cFuncPtr = ps.StreamingReadyType(self.streaming_callback)


        with self.timings.record('fetch_data'):
            # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
            while self.streaming_info['nextSample'] < totalSamples and not self.streaming_info['autoStopOuter']:
                wasCalledBack = False
                status["getStreamingLastestValues"] = ps.ps3000aGetStreamingLatestValues(self._chandle, cFuncPtr, None)
                if not wasCalledBack:
                    # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                    # again.
                    time.sleep(self.stream_fetch_pause())

        logger.debug("Done grabbing values.")

        # Stop the scope
        with self.timings.record('stop_scope'):
            # handle = self._chandle
            status["stop"] = ps.ps3000aStop(self._chandle)
            assert_pico_ok(status["stop"])

        logger.debug('Processing data')
        with self.timings.record('process_data'):
            buffers = self.process_data()
        return buffers

    def streaming_callback(
            self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param
    ):
        self.streaming_info['wasCalledBack'] = True
        self.streaming_info['points_per_callback'] = noOfSamples

        nextSample = self.streaming_info['nextSample']
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples

        for ch in self.active_channel_ids:
            self._raw_buffers[ch][nextSample:destEnd] = self._raw_single_buffers[ch][startIndex:sourceEnd]
        self.streaming_info['nextSample'] += noOfSamples

        # print('Triggered:', triggered, 'triggerAt', triggerAt)

        if autoStop:
            self.streaming_info['autoStopOuter'] = True

    def process_data(self):
        status = {}
        # Find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(self._chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])

        # Convert ADC counts data to mV
        for ch in self.active_channels:
            buffer_list = np.array(adc2mV(self._raw_buffers[ch.id], ch.range.raw_value, maxADC))
            buffer_list /= 1e3  # mV to V
            self.buffers[ch.id] = np.reshape(buffer_list, (self.samples(), self.points_per_trace()))

        return self.buffers

    def plot_traces(self):
        plot = MatPlot(subplots=len(self.buffers))
        t_list = np.arange(self.points_per_trace()) * (self.streaming_info['actualSampleIntervalNs'] / 1e9)
        samples = np.arange(self.samples(), dtype=float)
        for k, (ch, buffer) in enumerate(self.buffers.items()):
            plot[k].add(buffer, x=t_list, y=samples)

            plot[k].set_title(f'Channel {ch}')

    def acquisition(self):
        import ctypes
        from picosdk.ps3000a import ps3000a as ps
        import numpy as np
        import matplotlib.pyplot as plt
        from picosdk.functions import adc2mV, assert_pico_ok

        # Create chandle and status ready for use
        status = {}

        # Set up channel A
        # handle = self._chandle
        # channel = PS3000A_CHANNEL_A = 0
        # enabled = 1
        # coupling type = PS3000A_DC = 1
        # range = PS3000A_10V = 8
        # analogue offset = 0 V
        chARange = 8
        status["setChA"] = ps.ps3000aSetChannel(self._chandle, 0, 1, 1, chARange, 0)
        assert_pico_ok(status["setChA"])

        # Sets up single trigger
        # Handle = Chandle
        # Enable = 1
        # Source = ps3000A_channel_A = 0
        # Threshold = 1024 ADC counts
        # Direction = ps3000A_Falling = 3
        # Delay = 0
        # autoTrigger_ms = 1000
        status["trigger"] = ps.ps3000aSetSimpleTrigger(self._chandle, 1, 0, 1024, 3, 0, 1000)
        assert_pico_ok(status["trigger"])

        # Setting the number of sample to be collected
        preTriggerSamples = 40000
        postTriggerSamples = 40000
        maxsamples = preTriggerSamples + postTriggerSamples

        # Gets timebase innfomation
        # Handle = self._chandle
        # Timebase = 2 = timebase
        # Nosample = maxsamples
        # TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
        # MaxSamples = ctypes.byref(returnedMaxSamples)
        # Segement index = 0
        timebase = 2
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        status["GetTimebase"] = ps.ps3000aGetTimebase2(self._chandle, timebase, maxsamples, ctypes.byref(timeIntervalns), 1,
                                                       ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(status["GetTimebase"])

        # Creates a overlow location for data
        overflow = ctypes.c_int16()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(maxsamples)

        # Handle = Chandle
        # nSegments = 10
        # nMaxSamples = ctypes.byref(cmaxSamples)

        status["MemorySegments"] = ps.ps3000aMemorySegments(self._chandle, 10, ctypes.byref(cmaxSamples))
        assert_pico_ok(status["MemorySegments"])

        # sets number of captures
        status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(self._chandle, 10)
        assert_pico_ok(status["SetNoOfCaptures"])

        # Starts the block capture
        # Handle = self._chandle
        # Number of prTriggerSamples
        # Number of postTriggerSamples
        # Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
        # time indisposed ms = None (This is not needed within the example)
        # Segment index = 0
        # LpRead = None
        # pParameter = None
        status["runblock"] = ps.ps3000aRunBlock(self._chandle, preTriggerSamples, postTriggerSamples, timebase, 1, None, 0,
                                                None, None)
        assert_pico_ok(status["runblock"])

        # Create buffers ready for assigning pointers for data collection
        # bufferAMax = (ctypes.c_int16 * maxsamples)()
        # bufferAMin = (ctypes.c_int16 * maxsamples)() # used for downsampling which isn't in the scope of this example

        bufferAMax = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin = np.empty(maxsamples,
                              dtype=np.dtype('int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 0
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax.ctypes.data, bufferAMin.ctypes.data,
                                                            maxsamples, 0, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax1 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin1 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax1)
        # Buffer min = ctypes.byref(bufferAMin1)
        # Buffer length = maxsamples
        # Segment index = 1
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax1.ctypes.data,
                                                            bufferAMin1.ctypes.data, maxsamples, 1, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax2 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin2 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 2
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax2.ctypes.data,
                                                            bufferAMin2.ctypes.data, maxsamples, 2, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax3 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin3 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 3
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax3.ctypes.data,
                                                            bufferAMin3.ctypes.data, maxsamples, 3, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax4 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin4 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 4
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax4.ctypes.data,
                                                            bufferAMin4.ctypes.data, maxsamples, 4, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax5 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin5 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 5
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax5.ctypes.data,
                                                            bufferAMin5.ctypes.data, maxsamples, 5, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax6 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin6 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 6
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax6.ctypes.data,
                                                            bufferAMin6.ctypes.data, maxsamples, 6, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax7 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin7 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 7
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax7.ctypes.data,
                                                            bufferAMin7.ctypes.data, maxsamples, 7, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax8 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin8 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 8
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax8.ctypes.data,
                                                            bufferAMin8.ctypes.data, maxsamples, 8, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Create buffers ready for assigning pointers for data collection
        bufferAMax9 = np.empty(maxsamples, dtype=np.dtype('int16'))
        bufferAMin9 = np.empty(maxsamples, dtype=np.dtype(
            'int16'))  # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 9
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax9.ctypes.data,
                                                            bufferAMin9.ctypes.data, maxsamples, 9, 0)
        assert_pico_ok(status["SetDataBuffers"])

        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * 10)()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(maxsamples)

        # Checks data collection to finish the capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps3000aIsReady(self._chandle, ctypes.byref(ready))

        # Handle = self._chandle
        # noOfSamples = ctypes.byref(cmaxSamples)
        # fromSegmentIndex = 0
        # ToSegmentIndex = 9
        # DownSampleRatio = 0
        # DownSampleRatioMode = 0
        # Overflow = ctypes.byref(overflow)

        status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self._chandle, ctypes.byref(cmaxSamples), 0, 9, 1, 0,
                                                          ctypes.byref(overflow))
        assert_pico_ok(status["GetValuesBulk"])

        # Handle = self._chandle
        # Times = Times = (ctypes.c_int16*10)() = ctypes.byref(Times)
        # Timeunits = TimeUnits = ctypes.c_char() = ctypes.byref(TimeUnits)
        # Fromsegmentindex = 0
        # Tosegementindex = 9
        Times = (ctypes.c_int16 * 10)()
        TimeUnits = ctypes.c_char()
        status["GetValuesTriggerTimeOffsetBulk"] = ps.ps3000aGetValuesTriggerTimeOffsetBulk64(self._chandle,
                                                                                              ctypes.byref(Times),
                                                                                              ctypes.byref(TimeUnits),
                                                                                              0, 9)
        assert_pico_ok(status["GetValuesTriggerTimeOffsetBulk"])

        # Finds the max ADC count
        # Handle = self._chandle
        # Value = ctype.byref(maxADC)
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(self._chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])

        # Converts ADC from channel A to mV
        adc2mVChAMax = adc2mV(bufferAMax, chARange, maxADC)
        adc2mVChAMax1 = adc2mV(bufferAMax1, chARange, maxADC)
        adc2mVChAMax2 = adc2mV(bufferAMax2, chARange, maxADC)
        adc2mVChAMax3 = adc2mV(bufferAMax3, chARange, maxADC)
        adc2mVChAMax4 = adc2mV(bufferAMax4, chARange, maxADC)
        adc2mVChAMax5 = adc2mV(bufferAMax5, chARange, maxADC)
        adc2mVChAMax6 = adc2mV(bufferAMax6, chARange, maxADC)
        adc2mVChAMax7 = adc2mV(bufferAMax7, chARange, maxADC)
        adc2mVChAMax8 = adc2mV(bufferAMax8, chARange, maxADC)
        adc2mVChAMax9 = adc2mV(bufferAMax9, chARange, maxADC)

        # Creates the time data
        time = np.linspace(0, (cmaxSamples.value) * timeIntervalns.value, cmaxSamples.value)

        # Plots the data from channel A onto a graph
        plt.plot(time, adc2mVChAMax[:])
        plt.plot(time, adc2mVChAMax1[:])
        plt.plot(time, adc2mVChAMax2[:])
        plt.plot(time, adc2mVChAMax3[:])
        plt.plot(time, adc2mVChAMax4[:])
        plt.plot(time, adc2mVChAMax5[:])
        plt.plot(time, adc2mVChAMax6[:])
        plt.plot(time, adc2mVChAMax7[:])
        plt.plot(time, adc2mVChAMax8[:])
        plt.plot(time, adc2mVChAMax9[:])
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.show()

        # Stops the scope
        # Handle = self._chandle
        status["stop"] = ps.ps3000aStop(self._chandle)
        assert_pico_ok(status["stop"])

        # Displays the staus returns
        print(status)

    def acquisition_old(self):

        samples = 10
        pre_trigger_points = 40000
        points_per_trace = 40000

        status = {}

        # Setting the number of sample to be collected
        total_points_per_trace = pre_trigger_points + points_per_trace
        ctotal_points_per_trace = ctypes.c_int32(total_points_per_trace)  # C type

        # self.setup_trigger()

        # Gets timebase information
        # Handle = chandle
        # Timebase = 2 = timebase
        # Nosample = maxsamples
        # TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
        # MaxSamples = ctypes.byref(returnedMaxSamples)
        # Segement index = 0
        timebase = 2
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        status["GetTimebase"] = ps.ps3000aGetTimebase2(self._chandle, timebase, total_points_per_trace, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(status["GetTimebase"])

        # Set number of memory segments, should be at least equal to number of samples (i.e. traces).
        # Handle = Chandle
        # nSegments = 10
        # nMaxSamples = ctypes.byref(cmaxSamples)
        status["MemorySegments"] = ps.ps3000aMemorySegments(self._chandle, samples, ctypes.byref(ctotal_points_per_trace))
        assert_pico_ok(status["MemorySegments"])

        # sets number of samples (i.e. traces)
        status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(self._chandle, samples)
        assert_pico_ok(status["SetNoOfCaptures"])

        # Starts the block capture
        # Handle = self._chandle
        # Number of prTriggerSamples
        # Number of postTriggerSamples
        # Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
        # time indisposed ms = None (This is not needed within the example)
        # Segment index = 0
        # LpRead = None
        # pParameter = None
        status["runblock"] = ps.ps3000aRunBlock(self._chandle, pre_trigger_points, points_per_trace, timebase, 1, None, 0, None, None)
        assert_pico_ok(status["runblock"])

        # Collect data into arrays
        min_buffers = []
        max_buffers = []
        for k in range(samples):
            # Create buffers ready for assigning pointers for data collection
            bufferAMax = np.empty(total_points_per_trace, dtype=np.dtype('int16'))
            bufferAMin = np.empty(total_points_per_trace, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example
            min_buffers.append(bufferAMin)
            max_buffers.append(bufferAMax)

            # Setting the data buffer location for data collection from channel A
            # Handle = Chandle
            # source = ps3000A_channel_A = 0
            # Buffer max = ctypes.byref(bufferAMax)
            # Buffer min = ctypes.byref(bufferAMin)
            # Buffer length = maxsamples
            # Segment index = 0
            # Ratio mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self._chandle, 0, bufferAMax.ctypes.data, bufferAMin.ctypes.data, total_points_per_trace, 0, 0)
            assert_pico_ok(status["SetDataBuffers"])

        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * 10)()
        # Creates converted types maxsamples
        ctotal_points_per_trace = ctypes.c_int32(total_points_per_trace)

        # Checks data collection to finish the capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps3000aIsReady(self._chandle, ctypes.byref(ready))

        # Handle = self._chandle
        # noOfSamples = ctypes.byref(cmaxSamples)
        # fromSegmentIndex = 0
        # ToSegmentIndex = 9
        # DownSampleRatio = 0
        # DownSampleRatioMode = 0
        # Overflow = ctypes.byref(overflow)
        status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self._chandle, ctypes.byref(ctotal_points_per_trace), 0, 9, 1, 0, ctypes.byref(overflow))
        assert_pico_ok(status["GetValuesBulk"])

        # Handle = self._chandle
        # Times = Times = (ctypes.c_int16*10)() = ctypes.byref(Times)
        # Timeunits = TimeUnits = ctypes.c_char() = ctypes.byref(TimeUnits)
        # Fromsegmentindex = 0
        # Tosegementindex = 9
        Times = (ctypes.c_int16*10)()
        TimeUnits = ctypes.c_char()
        status["GetValuesTriggerTimeOffsetBulk"] = ps.ps3000aGetValuesTriggerTimeOffsetBulk64(self._chandle, ctypes.byref(Times), ctypes.byref(TimeUnits), 0, 9)
        assert_pico_ok(status["GetValuesTriggerTimeOffsetBulk"])

        # Finds the max ADC count
        # Handle = self._chandle
        # Value = ctype.byref(maxADC)
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(self._chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])

        # Converts ADC from channel A to mV
        channel = self.chA
        buffers = [adc2mV(buffer_ADC, channel.range.raw_value, maxADC) for buffer_ADC in max_buffers]

        # Stops the scope
        # Handle = chandle
        status["stop"] = ps.ps3000aStop(self._chandle)
        assert_pico_ok(status["stop"])

        # Creates the time data
        time = np.linspace(0, (ctotal_points_per_trace.value) * timeIntervalns.value, ctotal_points_per_trace.value)

        # Plots the data from channel A onto a graph

        fig, ax = plt.subplots()
        for buffer in buffers:
            plt.plot(time, buffer)
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.show()

        # Displays the staus returns
        return buffers
