from .ATS import AcquisitionController
import math
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class Basic_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """
    def __init__(self, name, alazar_name, **kwargs):
        super().__init__(name, alazar_name, **kwargs)
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None
        self.buffer_idx = 0

        self._acquisition_settings = {}

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))
        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisition_settings. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_acquisition,
                           shapes=((),),
                           snapshot_value=False)

    def get_acquisition_setting(self, setting):
        """
        Obtain an acquisition setting for the ATS.
        It checks if the setting is in ATS_controller._acquisition_settings
        If not, it will retrieve the ATS latest parameter value

        Args:
            setting: acquisition setting to look for

        Returns:
            Value of the acquisition setting
        """
        if setting in self._acquisition_settings.keys():
            return self._acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self._alazar.parameters[setting].get_latest()

    def update_acquisition_settings(self, **kwargs):
        self._acquisition_settings.update(**kwargs)

    def setup(self):
        """
        Setup the ATS controller by updating most current ATS values and setting
        the acquisition parameter metadata

        Returns:
            None
        """
        # Update acquisition parameter values. These depend on the average mode
        for attr in ['channel_selection', 'samples_per_record',
                     'records_per_buffer', 'buffers_per_acquisition']:
            setattr(self, attr, self.get_acquisition_setting(attr))
        self.number_of_channels = len(self.channel_selection)
        self.records_per_acquisition = self.buffers_per_acquisition * \
                                       self.records_per_buffer

        # Set acquisition parameter metadata
        self.acquisition.names = tuple(['ch{}_signal'.format(ch) for ch in
                                        self.channel_selection])
        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'] * self.number_of_channels

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()] * self.number_of_channels)
        elif self.average_mode() == 'trace':
            shape = (self.samples_per_record,)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)
        else:
            shape = (self.records_per_buffer * self.buffers_per_acquisition,
                     self.samples_per_record)
            self.acquisition.shapes = tuple([shape] * self.number_of_channels)

    def pre_start_capture(self):
        """
        Initializes buffers before capturing
        """
        self.buffer_idx = 0
        if self.average_mode() in ['point', 'trace']:
            self.buffer = np.zeros(self.samples_per_record *
                                   self.records_per_buffer *
                                   self.number_of_channels)
        else:
            self.buffer = np.zeros((self.buffers_per_acquisition,
                                    self.samples_per_record *
                                    self.records_per_buffer *
                                    self.number_of_channels))

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def do_acquisition(self):
        records = self._alazar.acquire(acquisition_controller=self,
                                       **self._acquisition_settings)
        return records

    def handle_buffer(self, data):
        if self.buffer_idx < self.buffers_per_acquisition:
            if self.average_mode() in ['point', 'trace']:
                self.buffer += data
            else:
                self.buffer[self.buffer_idx] = data
        else:
            print('*'*20+'\nIgnoring extra ATS buffer')
            pass
        self.buffer_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.

        ch_offset = lambda ch: ch * self.samples_per_record * \
                               self.records_per_buffer

        if self.average_mode() == 'none':
            records = [self.buffer[:, ch_offset(ch):ch_offset(ch+1)].reshape(
                (self.records_per_acquisition, self.samples_per_record))
                       for ch in range(self.number_of_channels)]

        elif self.average_mode() == 'trace':
            records = [np.zeros(self.samples_per_record)
                       for _ in range(self.number_of_channels)]

            for ch in range(self.number_of_chs):
                for i in range(self.records_per_buffer):
                    i0 = ch_offset(ch) + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[ch] += self.buffer[i0:i1]
                records[ch] /= self.records_per_acquisition

        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length])
                       / self.records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Convert data points from an uint8 to volts
        for ch, record in enumerate(records):
            ch_idx = self.channel_selection[ch]
            ch_range = self._alazar.parameters['channel_range'+ch_idx]()
            records[ch] = (record - 127.5) / 127.5 * ch_range
        return records


# DFT AcquisitionController
class Demodulation_AcquisitionController(AcquisitionController):
    """
    This class represents an example acquisition controller. End users will
    probably want to use something more sophisticated. It will average all
    buffers and then perform a fourier transform on the resulting average trace
    for one frequency component. The amplitude of the result of channel_a will
    be returned.

    args:
    name: name for this acquisition_conroller as an instrument
    alazar_name: the name of the alazar instrument such that this controller
        can communicate with the Alazar
    demodulation_frequency: the selected component for the fourier transform
    **kwargs: kwargs are forwarded to the Instrument base class
    """
    def __init__(self, name, alazar_name, demodulation_frequency, **kwargs):
        self.demodulation_frequency = demodulation_frequency
        self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        sample_speed = alazar.get_sample_rate()
        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.demodulation_frequency / sample_speed *
                      integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(self, data):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        # average all records in a buffer
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record
            i1 = i0 + self.samples_per_record
            recordA += self.buffer[i0:i1] / records_per_acquisition

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record + len(self.buffer) // 2
            i1 = i0 + self.samples_per_record
            recordB += self.buffer[i0:i1] / records_per_acquisition

        if self.number_of_channels == 2:
            # fit channel A and channel B
            res1 = self.fit(recordA)
            res2 = self.fit(recordB)
            #return [alazar.signal_to_volt(1, res1[0] + 127.5),
            #        alazar.signal_to_volt(2, res2[0] + 127.5),
            #        res1[1], res2[1],
            #        (res1[1] - res2[1]) % 360]
            return alazar.signal_to_volt(1, res1[0] + 127.5)
        else:
            raise Exception("Could not find CHANNEL_B during data extraction")
        return None

    def fit(self, buf):
        """
        the DFT is implemented in this method
        :param buf: buffer to perform the transform on
        :return: return amplitude and phase of the resulted transform
        """
        # Discrete Fourier Transform
        RePart = np.dot(buf - 127.5, self.cos_list) / self.samples_per_record
        ImPart = np.dot(buf - 127.5, self.sin_list) / self.samples_per_record

        # the factor of 2 in the amplitude is due to the fact that there is
        # a negative frequency as well
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)

        # see manual page 52!!! (using unsigned data)
        return [ampl, math.atan2(ImPart, RePart) * 360 / (2 * math.pi)]
