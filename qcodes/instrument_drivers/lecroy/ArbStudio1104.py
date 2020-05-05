import clr  # Import pythonnet to talk to dll
from System import Array
import numpy as np
import numbers
from time import sleep
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter_node import parameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils import validators as vals
import logging


logger = logging.getLogger(__name__)


class AWGChannel(InstrumentChannel):
    def __init__(self, parent, name, id, channel_api, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.id = id
        self.channel_api = channel_api
        self._api = self.parent._api

        self.trigger_mode = Parameter(
            label=f"Channel {id} trigger mode",
            vals=vals.Enum("single", "continuous", "stepped", "burst"),
        )

        self.trigger_source = Parameter(
            label=f"Channel {id} trigger source",
            vals=vals.Enum(
                "stop", "start", "event_marker", "dc_trigger_in", "fp_trigger_in"
            ),
        )

        self.sampling_rate_prescaler = Parameter(
            label=f"Channel {id} sampling rate prescaler",
            vals=vals.MultiType(vals.Multiples(2), vals.Enum(1)),
        )

        self.max_voltage = Parameter(
            label="Maximum waveform voltage",
            unit="V",
            set_cmd=None,
            initial_value=6,
            vals=vals.Numbers(),
            docstring="Maximum waveform voltage. Any waveform added cannot have"
            "a voltage higher than this.",
        )

        self.sequence = Parameter(
            label=f"Channel {id} Sequence",
            set_cmd=None,
            initial_value=[],
            vals=vals.Anything(),
            log_changes=False,
            snapshot_value=False,
        )  # Can we test for an (int, int) tuple list?

        # Keep actual waveforms hidden so they cannot be directly set
        # i.e. arbstudio.waveforms = waveform_list  # Raises an error
        # waveforms should be added via arbstudio.add_waveform
        self._waveforms = []

    @property
    def waveforms(self):
        return self._waveforms

    @parameter
    def trigger_mode_set(self, parameter, trigger_mode):
        # Create dictionary with possible TriggerMode objects
        trigger_modes = {
            "single": self._api.TriggerMode.Single,
            "continuous": self._api.TriggerMode.Continuous,
            "stepped": self._api.TriggerMode.Stepped,
            "burst": self._api.TriggerMode.Burst,
        }

        self.parent.call_dll(
            self.channel_api.SetTriggerMode,
            trigger_modes[trigger_mode],
            msg=f"setting trigger_mode of ch{self.id} to {trigger_mode}",
        )

    @parameter
    def trigger_source_set(self, parameter, trigger_source):
        if trigger_source == "internal":
            self.parent.call_dll(
                self.channel_api.SetInternalTrigger,
                f"setting channel {self.id} trigger source to {trigger_source}",
            )
        else:
            # Collect external trigger arguments
            trigger_source = self.parent._trigger_sources[trigger_source]
            trigger_sensitivity_edge = self.parent._trigger_sensitivity_edges[
                self.parent.trigger_sensitivity_edge()
            ]
            trigger_action = self.parent._trigger_actions[self.parent.trigger_action()]

            self.parent.call_dll(
                self.channel_api.SetExternalTrigger,
                trigger_source,
                trigger_sensitivity_edge,
                trigger_action,
                msg=f"setting channel {self.id} trigger source to {trigger_source}",
            )

    @parameter
    def sampling_rate_prescaler_get(self, parameter):
        return self.channel_api.SampligRatePrescaler  # Typo is intentional

    @parameter
    def sampling_rate_prescaler_set(self, parameter, prescaler):
        self.channel_api.SampligRatePrescaler = prescaler  # Typo is intentional

    def add_waveform(self, waveform):
        assert len(waveform) % 2 == 0, "Waveform must have an even number of points"
        assert len(waveform) > 2, "Waveform must have at least four points"
        assert (
            max(abs(waveform)) <= self.max_voltage()
        ), f"Waveform may not exceed {self.max_voltage()} V"
        self.waveforms.append(waveform)

    def load_waveforms(self):
        if not self.waveforms:
            # Must have at least one waveform, add one with 0V
            self.add_waveform(np.array([0, 0, 0, 0]))

        # Initialize array of waves
        waveforms = Array.CreateInstance(self._api.WaveformStruct, len(self.waveforms))

        # We have to create separate wave instances and load them into
        # the waves array one by one
        for k, waveform_array in enumerate(self.waveforms):
            wave = self._api.WaveformStruct()
            wave.Sample = waveform_array
            waveforms[k] = wave

        self.parent.call_dll(
            self.channel_api.LoadWaveforms, waveforms, msg="loading waveforms"
        )

    def load_sequence(self):
        # Check if sequence consists of repetitions of a subsequence

        sequence = self.sequence()
        if "divisors" in self.parent.optimize:
            try:
                N = len(sequence)
                divisors = [n for n in reversed(np.arange(2, N + 1)) if N % n == 0]
                for divisor in divisors:
                    sequence_arr = np.array(sequence)
                    reshaped_arr = sequence_arr.reshape(divisor, int(N / divisor))
                    if (reshaped_arr == reshaped_arr[0]).all():
                        sequence = reshaped_arr[0]
                        break
            except Exception:
                pass

        self.sequence = sequence

        sequence_obj = Array.CreateInstance(
            self._api.GenerationSequenceStruct, len(sequence)
        )
        for k, subsequence_info in enumerate(sequence):
            subsequence = self._api.GenerationSequenceStruct()
            # Must compare with Integral since np.int32 is not an int
            if isinstance(subsequence_info, numbers.Integral):
                subsequence.WaveformIndex = subsequence_info
                # Set repetitions to 1 (default) if subsequence info is an int
                subsequence.Repetitions = 1
            elif isinstance(subsequence_info, tuple):
                assert (
                    len(subsequence_info) == 2
                ), "A subsequence tuple must be of the form (WaveformIndex, Repetitions)"
                subsequence.WaveformIndex = subsequence_info[0]
                subsequence.Repetitions = subsequence_info[1]
            else:
                raise TypeError(
                    "A subsequence must be either an int or (int, int) tuple"
                )
            sequence_obj[k] = subsequence

        # Set transfermode to USB (seems to be a fixed function)
        transfer_mode = Array.CreateInstance(self._api.TransferMode, 1)
        self.parent.call_dll(
            self.channel_api.LoadGenerationSequence,
            sequence_obj,
            transfer_mode[0],
            True,
            msg="loading sequence",
        )

    def clear_waveforms(self):
        self.waveforms.clear()


class ArbStudio1104(Instrument):
    # Allowed optimizations: "divisors"
    optimize = []

    # Number of times to try and call a dll function beforegiving up
    dll_call_attempts = 5

    # Delay between successive DLL calls
    dll_call_delay = 0.5  # seconds

    # Sampling rate itself is fixed, but can be modified with
    # sampling_rate_prescaler
    sampling_rate = 250e6  # Hz

    def __init__(self, name, dll_path, **kwargs):
        super().__init__(name, **kwargs)

        # Add .NET assembly to pythonnet
        # This allows importing its functions
        clr.System.Reflection.Assembly.LoadFile(dll_path)
        from clr import ActiveTechnologies

        self._api = ActiveTechnologies.Instruments.AWG4000.Control

        ### Instrument constants (set here since api is only defined above)
        self._trigger_sources = {
            "stop": self._api.TriggerSource.Stop,
            "start": self._api.TriggerSource.Start,
            "event_marker": self._api.TriggerSource.Event_Marker,
            "dc_trigger_in": self._api.TriggerSource.DCTriggerIN,
            "fp_trigger_in": self._api.TriggerSource.FPTriggerIN,
        }
        self._trigger_sensitivity_edges = {
            "rising": self._api.SensitivityEdge.RisingEdge,
            "falling": self._api.SensitivityEdge.RisingEdge,
        }
        self._trigger_actions = {
            "start": self._api.TriggerAction.TriggerStart,
            "stop": self._api.TriggerAction.TriggerStop,
            "ignore": self._api.TriggerAction.TriggerIgnore,
        }

        ### Get device object
        self._device = self._api.DeviceSet().DeviceList[0]

        ### Initialize device and channels
        self.initialize()

        ### Add channels containing their own parameters/functions
        for channel_id in range(1, 5):
            channel = AWGChannel(
                self,
                name=f"ch{channel_id}",
                id=channel_id,
                channel_api=self._device.GetChannel(channel_id - 1),
            )
            setattr(self, f"ch{channel_id}", channel)

        self.channels = ChannelList(
            parent=self,
            name="channels",
            chan_type=AWGChannel,
            chan_list=[self.ch1, self.ch2, self.ch3, self.ch4],
        )
        self.add_submodule("channels", self.channels)

        ### Add channel pair settings
        self.left_frequency_interpolation = Parameter(
            set_cmd=self._device.PairLeft.SetFrequencyInterpolation,
            vals=vals.Enum(1, 2, 4),
        )
        self.right_frequency_interpolation = Parameter(
            set_cmd=self._device.PairRight.SetFrequencyInterpolation,
            vals=vals.Enum(1, 2, 4),
        )

        # TODO Need to implement frequency interpolation for channel pairs
        # self.add_parameter('frequency_interpolation',
        #                    label='DAC frequency interpolation factor',
        #                    vals=vals.Enum(1, 2, 4))

        self.trigger_sensitivity_edge = Parameter(
            initial_value="rising",
            label="Trigger sensitivity edge for in/out",
            set_cmd=None,
            vals=vals.Enum("rising", "falling"),
        )

        self.trigger_action = Parameter(
            initial_value="start",
            label="Trigger action",
            set_cmd=None,
            vals=vals.Enum("start", "stop", "ignore"),
        )

    def initialize(self):
        # Create empty array of four channels.
        # These are only necessary for initialization
        channels = Array.CreateInstance(self._api.Functionality, 4)
        # Initialize each of the channels
        channels[0] = self._api.Functionality.ARB
        channels[1] = self._api.Functionality.ARB
        channels[2] = self._api.Functionality.ARB
        channels[3] = self._api.Functionality.ARB

        # Initialise ArbStudio
        self.call_dll(self._device.Initialize, channels, msg="initializing")

    def call_dll(self, command, *args, msg="", attempts=None, **kwargs):
        if attempts is None:
            attempts = self.dll_call_attempts
        for k in range(attempts):
            return_msg = command(*args, **kwargs)

            if return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS:
                break
            else:
                logger.warning(f"Arbstudio unsuccessful attempt {k} at {msg} ")
                sleep(self.dll_call_delay)
        else:
            raise RuntimeError(f"Arbstudio error {msg}: {return_msg.ErrorDescription}")

        return return_msg

    def load_waveforms(self, channels=(1, 2, 3, 4)):
        for channel in self.channels:
            if channel.id not in channels:
                continue

            channel.load_waveforms()

    def load_sequence(self, channels=(1, 2, 3, 4)):
        for channel in self.channels:
            if channel.id not in channels:
                continue

            channel.load_sequence()

    def run(self, channels=[1, 2, 3, 4]):
        """
        Run sequences on given channels
        Args:
            channels: List of channels to run, starting at 1 (default all)

        Returns:
            None
        """
        self.call_dll(self._device.RUN, channels, msg="running")

    def stop(self):
        """
        Stop sequence on given channels
        Args:
            channels: List of channels to stop, starting at 1 (default all)

        Returns:
            None
        """
        self.call_dll(self._device.STOP, msg="stopping")

        # A stop command seems to reset trigger sources. For the channels that had a trigger source,
        # this will reset it to its previoius value
        for channel in self.channels:
            trigger_source = channel.trigger_source.get_latest()
            if trigger_source:
                channel.trigger_source(trigger_source)

