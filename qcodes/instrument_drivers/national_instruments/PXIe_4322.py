from qcodes.instrument.base import Instrument
from qcodes import validators as validator

from functools import partial
import warnings
import json
import math
from timeit import default_timer as timer
from time import sleep

try:
    import nidaqmx
except ImportError:
    raise ImportError('to use the National Instrument PXIe-4322 driver, please install the nidaqmx package'
                      '(https://github.com/ni/nidaqmx-python)')


class PXIe_4322(Instrument):
    """
    This is the QCoDeS driver for the National Instrument PXIe-4322 Analog Output Module.

    The current version of this driver only allows using the PXIe-4322 as a DC Voltage Output.

    This driver makes use of the Python API for interacting with the NI-DAQmx driver. Both the NI-DAQmx driver and
    the nidaqmx package need to be installed in order to use this QCoDeS driver.
    """

    def __init__(self, name, device_name, step_size=0.01, step_rate=10, **kwargs):
        super().__init__(name, **kwargs)

        self.device_name = device_name

        self.channels = 8

        self.step_size = step_size
        self.step_rate = step_rate
        self.step_delay = 1/step_rate

        try:
            with open('NI_voltages_{}.json'.format(device_name)) as data_file:
                self.__voltage = json.load(data_file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            self.__voltage = [0] * self.channels

        print('Please read the following warning message:')

        warnings.warn('The last known output values are: {} Please check these values and make sure they correspond '
                      'to the actual output of the PXIe-4322 module. Any difference between stored value and actual '
                      'value WILL cause sudden jumps in output.'.format(self.__voltage), UserWarning)

        for i in range(self.channels):
            self.add_parameter('voltage_channel_{}'.format(i),
                               label='voltage channel {}'.format(i),
                               unit='V',
                               set_cmd=partial(self.set_voltage, channel=i),
                               get_cmd=partial(self.get_voltage, channel=i),
                               docstring='The DC output voltage of channel {}'.format(i),
                               vals=validator.Numbers(-16, 16))

    def set_voltage(self, voltage, channel, save_to_file=True):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(self.device_name, channel),
                                                 min_val=-16.0, max_val=16.0)

            if abs(voltage - self.__voltage[channel]) > self.step_size:
                if (voltage - self.__voltage[channel]) < 0:
                    step = -self.step_size
                else:
                    step = self.step_size
                for voltage_step in frange(self.__voltage[channel], voltage, step):
                    t_start = timer()
                    task.write(voltage_step)
                    t_stop = timer()
                    sleep(max(self.step_delay-(t_stop-t_start), 0.0))

            task.write(voltage)
            self.__voltage[channel] = voltage
            if save_to_file:
                with open('NI_voltages_{}.json'.format(self.device_name), 'w') as output_file:
                    json.dump(self.__voltage, output_file, ensure_ascii=False)

    def get_voltage(self, channel):
        return self.__voltage[channel]

    def set_gates_simultaneously(self, gate_values):
        assert len(gate_values) == self.channels, 'number of values in gate_values list ({}) must be same as number ' \
                                                  'of channels: {}'.format(len(gate_values), self.channels)

        diff = [gate_values[i] - self.__voltage[i] for i in range(len(self.__voltage))]
        step = [self.step_size if diff_i >= 0 else -self.step_size for diff_i in diff]
        volt_steps = [frange(self.__voltage[i], gate_values[i], step[i]) for i in range(len(self.__voltage))]
        number_of_steps = [len(volt_steps[i]) for i in range(len(volt_steps))]

        channel_mask = [True] * self.channels

        for i in range(max(number_of_steps)):
            for chan in range(self.channels):
                if channel_mask[chan]:
                    try:
                        voltage = volt_steps[chan][i]
                        self.set_voltage(voltage, chan)
                    except IndexError:
                        channel_mask[chan] = False
                        pass

        for i in range(self.channels):
            self.set_voltage(gate_values[i], i)


def frange(start, stop, step):
    if stop is None:
        stop, start = start, 0.
    else:
        start = float(start)

    count = int(math.ceil((stop - start) / step))
    return [start + n * step for n in range(count)]

