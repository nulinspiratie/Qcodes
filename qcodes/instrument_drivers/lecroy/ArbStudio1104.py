import os
import clr  # Import pythonnet to talk to dll
from System import Array
import numpy as np
import numbers
from time import sleep
from functools import partial
from qcodes import Instrument
from qcodes.utils import validators as vals
import logging


logger = logging.getLogger(__name__)

class ArbStudio1104(Instrument):
    optimize = []
    def __init__(self, name, dll_path, **kwargs):
        super().__init__(name, **kwargs)

        # Add .NET assembly to pythonnet
        # This allows importing its functions
        clr.System.Reflection.Assembly.LoadFile(dll_path)
        from clr import ActiveTechnologies
        self._api = ActiveTechnologies.Instruments.AWG4000.Control

        # Instrument constants (set here since api is only defined above)
        self._trigger_sources = {
            'stop': self._api.TriggerSource.Stop,
            'start': self._api.TriggerSource.Start,
            'event_marker': self._api.TriggerSource.Event_Marker,
            'dc_trigger_in': self._api.TriggerSource.DCTriggerIN,
            'fp_trigger_in': self._api.TriggerSource.FPTriggerIN
        }
        self._trigger_sensitivity_edges = {
            'rising': self._api.SensitivityEdge.RisingEdge,
            'falling': self._api.SensitivityEdge.RisingEdge
        }
        self._trigger_actions= {
            'start': self._api.TriggerAction.TriggerStart,
            'stop': self._api.TriggerAction.TriggerStop,
            'ignore': self._api.TriggerAction.TriggerIgnore
        }

        # Get device object
        self._device = self._api.DeviceSet().DeviceList[0]

        self.initialize()
        self._channels = [self._device.GetChannel(k) for k in range(4)]
        self._channel_pairs = {'left': self._device.PairLeft,
                               'right': self._device.PairRight}

        # Initialize waveforms and sequences
        self._waveforms = [[] for k in range(4)]

        # Sampling rate itself is fixed, but can be modified with
        # sampling_rate_prescaler
        self.sampling_rate = 250e6 #Hz

        self.add_parameter('waveforms',
                           get_cmd=lambda: self._waveforms)

        for ch in range(1,5):
            self.add_parameter('ch{}_trigger_mode'.format(ch),
                               label='Channel {} trigger mode'.format(ch),
                               set_cmd=partial(self._set_trigger_mode, ch),
                               vals=vals.Strings())

            self.add_parameter('ch{}_trigger_source'.format(ch),
                               label='Channel {} trigger source'.format(ch),
                               set_cmd=partial(self._set_trigger_source, ch),
                               vals=vals.Enum('stop', 'start', 'event_marker',
                                              'dc_trigger_in', 'fp_trigger_in'))

            self.add_parameter('ch{}_sampling_rate_prescaler'.format(ch),
                               label='Channel {} sampling rate prescaler'.format(ch),
                               get_cmd=partial(self._get_sampling_rate_prescaler, ch),
                               set_cmd=partial(self._set_sampling_rate_prescaler, ch),
                               vals=vals.MultiType(Multiples(2), vals.Enum(1)))

            self.add_parameter('ch{}_sequence'.format(ch),
                               label='Channel {} Sequence'.format(ch),
                               set_cmd=None,
                               initial_value=[],
                               vals=vals.Anything()) # Can we test for an (int, int) tuple list?

            self.add_function('ch{}_add_waveform'.format(ch),
                              call_cmd=partial(self._add_waveform, ch),
                              args=[vals.Anything()]) # Can we test for a float list/array?

            self.add_function('ch{}_clear_waveforms'.format(ch),
                              call_cmd=self._waveforms[ch-1].clear)

        for ch_pair_name in ['left', 'right']:
            ch_pair = self._channel_pairs[ch_pair_name]

            self.add_parameter('{}_frequency_interpolation'.format(ch_pair_name),
                               set_cmd=ch_pair.SetFrequencyInterpolation,
                               vals=vals.Enum(1,2,4))

        self.add_parameter('max_voltage',
                           label='Maximum waveform voltage',
                           unit='V',
                           set_cmd=None,
                           initial_value=6,
                           vals=vals.Numbers())  # Can we test

        # TODO Need to implement frequency interpolation for channel pairs
        # self.add_parameter('frequency_interpolation',
        #                    label='DAC frequency interpolation factor',
        #                    vals=vals.Enum(1, 2, 4))

        self.add_parameter('trigger_sensitivity_edge',
                           initial_value='rising',
                           label='Trigger sensitivity edge for in/out',
                           set_cmd=None,
                           vals=vals.Enum('rising', 'falling'))

        self.add_parameter('trigger_action',
                           initial_value='start',
                           label='Trigger action',
                           set_cmd=None,
                           vals=vals.Enum('start', 'stop', 'ignore'))

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
        self._call_dll(self._device.Initialize, channels, msg="initializing")

    def _call_dll(self, command, *args, msg="", attempts=5, **kwargs):
        for k in range(attempts):
            return_msg = command(*args, **kwargs)

            if return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS:
                break
            else:
                logger.warning(f'Arbstudio unsuccessful attempt {k} at {msg} ')
                sleep(0.5)
        else:
            raise RuntimeError(f"Arbstudio error {msg}: {return_msg.ErrorDescription}")

        return return_msg


    def _get_sampling_rate_prescaler(self, ch):
        return self._channels[ch - 1].SampligRatePrescaler #Typo is intentional

    def _set_sampling_rate_prescaler(self, ch, prescaler):
        self._channels[ch - 1].SampligRatePrescaler = prescaler
        #Typo is intentional

    def _set_trigger_mode(self, ch, trigger_mode_string):
        #Create dictionary with possible TriggerMode objects
        trigger_modes = {'single': self._api.TriggerMode.Single,
                         'continuous': self._api.TriggerMode.Continuous,
                         'stepped': self._api.TriggerMode.Stepped,
                         'burst': self._api.TriggerMode.Burst}

        #Transform trigger mode to lowercase, such that letter case does not matter
        trigger_mode_string = trigger_mode_string.lower()
        trigger_mode = trigger_modes[trigger_mode_string]
        self._call_dll(
            self._channels[ch-1].SetTriggerMode,
            trigger_mode,
            msg=f"setting trigger_mode of ch{ch} to {trigger_mode}"
        )

    def _set_trigger_source(self, ch, trigger_source_str):
        if trigger_source_str == 'internal':
            self._call_dll(
                self._channels[ch-1].SetInternalTrigger,
                f"setting channel {ch} trigger source to {trigger_source_str}"
            )
        else:
            # Collect external trigger arguments
            trigger_source = self._trigger_sources[trigger_source_str]
            trigger_sensitivity_edge = self._trigger_sensitivity_edges[self.trigger_sensitivity_edge()]
            trigger_action = self._trigger_actions[self.trigger_action()]

            self._call_dll(
                self._channels[ch-1].SetExternalTrigger,
                trigger_source,
                trigger_sensitivity_edge,
                trigger_action,
                msg=f"setting channel {ch} trigger source to {trigger_source_str}"
            )

    def _add_waveform(self, channel, waveform):
        assert len(waveform)%2 == 0, 'Waveform must have an even number of points'
        assert len(waveform)> 2, 'Waveform must have at least four points'
        assert max(abs(waveform)) <= self.max_voltage(), \
            f'Waveform may not exceed {self.max_voltage()} V'
        self._waveforms[channel - 1].append(waveform)

    def load_waveforms(self, channels=[1, 2, 3, 4]):
        channels.sort()
        waveforms_list = []
        for ch, channel in enumerate(self._channels):
            waveforms_array = self._waveforms[ch]
            if not waveforms_array:
                # Must have at least one waveform, add one with 0V
                waveforms_array = [np.array([0, 0, 0, 0])]
            # Initialize array of waves
            waveforms = Array.CreateInstance(self._api.WaveformStruct,len(waveforms_array))
            # We have to create separate wave instances and load them into
            # the waves array one by one
            for k, waveform_array in enumerate(waveforms_array):
                wave = self._api.WaveformStruct()
                wave.Sample = waveform_array
                waveforms[k] = wave

            self._call_dll(channel.LoadWaveforms, waveforms, msg="loading waveforms")
            waveforms_list.append(waveforms)

    def load_sequence(self, channels=[1, 2, 3, 4]):
        channels.sort()
        sequence_list = []
        for ch in channels:
            channel = self._channels[ch-1]
            channel_sequence = eval(f"self.ch{ch}_sequence()")

            # Check if sequence consists of repetitions of a subsequence
            if 'divisors' in self.optimize:
                try:
                    N = len(channel_sequence)
                    divisors = [n for n in reversed(np.arange(2, N+1))
                                                    if N % n == 0]
                    for divisor in divisors:
                        sequence_arr = np.array(channel_sequence)
                        reshaped_arr = sequence_arr.reshape(divisor, int(N/divisor))
                        if (reshaped_arr == reshaped_arr[0]).all():
                            channel_sequence = reshaped_arr[0]
                            break
                except:
                    pass

            exec(f"self.ch{ch}_sequence(channel_sequence)")

            sequence = Array.CreateInstance(self._api.GenerationSequenceStruct,len(channel_sequence))
            for k, subsequence_info in enumerate(channel_sequence):
                subsequence = self._api.GenerationSequenceStruct()
                # Must compare with Integral since np.int32 is not an int
                if isinstance(subsequence_info, numbers.Integral):
                    subsequence.WaveformIndex = subsequence_info
                    # Set repetitions to 1 (default) if subsequence info is an int
                    subsequence.Repetitions = 1
                elif isinstance(subsequence_info, tuple):
                    assert len(subsequence_info) == 2, \
                        'A subsequence tuple must be of the form (WaveformIndex, Repetitions)'
                    subsequence.WaveformIndex = subsequence_info[0]
                    subsequence.Repetitions = subsequence_info[1]
                else:
                    raise TypeError("A subsequence must be either an int or (int, int) tuple")
                sequence[k] = subsequence

            sequence_list.append(sequence)

            # Set transfermode to USB (seems to be a fixed function)
            trans = Array.CreateInstance(self._api.TransferMode, 1)
            self._call_dll(
                channel.LoadGenerationSequence,
                sequence,
                trans[0],
                True,
                msg="loading sequence"
            )

    def run(self, channels=[1, 2, 3, 4]):
        """
        Run sequences on given channels
        Args:
            channels: List of channels to run, starting at 1 (default all)

        Returns:
            None
        """
        self._call_dll(self._device.RUN, channels, msg="running")

    def stop(self, channels=[1, 2, 3, 4]):
        """
        Stop sequence on given channels
        Args:
            channels: List of channels to stop, starting at 1 (default all)

        Returns:
            None
        """
        self._call_dll(self._device.STOP, msg="stopping")

        # A stop command seems to reset trigger sources. For the channels that had a trigger source,
        # this will reset it to its previoius value
        for ch in range(1,5):
            trigger_source = eval('self.ch{}_trigger_source.get_latest()'.format(ch))
            if trigger_source:
                eval("self.ch{}_trigger_source(trigger_source)".format(ch))


class Multiples(vals.Ints):
    '''
    requires an integer
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    divisor enforces that value % divisor == 0
    '''

    def __init__(self, divisor=1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(divisor, int):
            raise TypeError('divisor must be an integer')
        self._divisor = divisor

    def validate(self, value, context=''):
        super().validate(value=value, context=context)
        if not value % self._divisor == 0:
            raise TypeError('{} is not a multiple of {}; {}'.format(
                repr(value), repr(self._divisor), context))

    def __repr__(self):
        return super().__repr__()[:-1] + ', Multiples of {}>'.format(self._divisor)
