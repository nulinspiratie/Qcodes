
from functools import partial
import logging
import numpy as np

from qcodes import VisaInstrument
from qcodes.instrument.parameter import StandardParameter, ManualParameter
from qcodes.utils import validators as vals
from time import sleep


cmdbase = "TERM LF\nFLSH\nFLOQ\n"

class SIM928(StandardParameter):
    """
    This is the parameter class for the SIM928 rechargeable isolated voltage source module

    Args:
        channel (int): SIM900 channel for the SIM928 module

        name (Optional[str]): Module name (default 'channel_{channel}')

        max_voltage (Optional[float]): Maximum voltage (default 20)
    """
    def __init__(self, channel, name=None, max_voltage=20, **kwargs):
        if not name:
            name = 'channel_{}'.format(channel)

        self.send_cmd = cmdbase + "SNDT {:d} ,".format(channel)

        super().__init__(name=name,
                         unit='V',
                         get_cmd=self.get_voltage,
                         set_cmd=self.send_cmd + '"VOLT {:.4f}"',
                         step=0.01,
                         delay=0.035,
                         vals=vals.Numbers(-max_voltage, max_voltage),
                         **kwargs)
        self.channel = channel

        self._meta_attrs.extend(['reset', 'charge_cycles'])

    @property
    def charge_cycles(self):
        self._instrument.write(self.send_cmd + '"BIDN? CYCLES"')
        sleep(0.08)
        return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
        try:
            return int(return_str.rstrip()[5:])
        except:
            logging.warning('Return string not understood: ' + return_str)
            return -1

    def get_voltage(self):
        """
        Retrieves the DAC voltage.
        Note that there is a small delay, since two commands must be sent.

        Returns:
            Channel voltage
        """
        # Two commands must be sent to the instrument to retrieve the channel voltage
        self._instrument.write(self.send_cmd + '"VOLT?"')
        # A small wait is needed before the actual voltage can be retrieved
        sleep(0.05)
        return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
        if return_str == '#3000\n':
            self._instrument.reset_slot(self.channel)
            sleep(1)
            return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
        return float(return_str[5:-3])


class SIM900(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research SIM900.
    It is currently only programmed for DAC voltage sources.

    Args:
        name (str): name of the instrument.
        address (str): The GPIB address of the instrument.
    """

    # Dictionary containing current module classes
    modules = {'SIM928': SIM928}

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        # The SIM900 has eight channels
        self.number_of_channels = 8

        # Dictionary with (channel, module) elements
        self._modules = {}

        # Start with empty list of channels. These are
        self.add_parameter('channels',
                           parameter_class=ManualParameter,
                           initial_value={},
                           vals=vals.Anything(),
                           snapshot_value=False)

    def define_slot(self, channel, name=None, module='SIM928', **kwargs):
        """
        Define a module for a SIM900 slot.
        Args:
            channel (int): The SIM900 slot channel for the module
            name (Optional[str]): Module name (default 'channel_{channel}')
            module (Optional[str]): Module type (default 'SIM928)
            **kwargs: Module-specific kwargs, and StandardParameter kwargs

        Returns:
            None
        """
        assert isinstance(channel, int), "Channel {} must be an integer".format(channel)
        assert channel not in self.channels().keys(), "Channel {} already exists".format(channel)
        assert module in self.modules.keys(), "Module {} is not programmed".format(module)

        self.add_parameter(name=name,
                           channel=channel,
                           parameter_class=self.modules[module],
                           **kwargs)

        # Add
        channels = self.channels()
        channels[channel] = name
        self.channels(channels)

    def reset_slot(self, channel):
        self.write(cmdbase + 'SRST {}'.format(channel))


def get_voltages():
    """ Get scaled parameter voltages as dict """
    # TODO find way to not have to use variable SIM900_scaled_parameters
    global SIM900_scaled_parameters
    return {param.name: param() for param in SIM900_scaled_parameters}


def ramp_voltages(target_voltage=None, channels=None, use_scaled=True,
                  **kwargs):
    """
    Ramp multiple gates in multiple steps.
    
    Note that SIM900_scaled_parameters must be defined in your global 
    namespace as a list containing scaled SIM928 parameters
    
    Usage:
        ramp_voltages(target_voltage)
            Ramp voltages of all gates to target_voltage
        ramp_voltages(target_voltage, channels)
            Ramp voltages of gates with names in channels to target_voltage
        ramp_voltages(gate1=val1, gate2=val2, ...)
            Ramp voltage of gate1 to val1, gate2 to val2, etc.
            
    Args:
        target_voltage (int): target voltage (can be omitted)
        channels (str list): Names of gates to be ramped (can be omitted)
        use_scaled: Use scaled SIM parameter (SIM900_scaled_parameters)
        **kwargs: 

    Returns:
        None
    """
    if use_scaled:
        global SIM900_scaled_parameters
        parameters = {param.name: param for param in SIM900_scaled_parameters}
    else:
        global SIM900
        parameters = {parameter_name: parameter
                      for [parameter_name, parameter] in SIM900.parameters.items()
                      if hasattr(parameter, 'channel')}

    if target_voltage is not None:
        if channels is None:
            channels = kwargs.keys()
        target_voltages = {parameters[channel]: target_voltage
                           for channel in channels}
    elif kwargs:
        target_voltages = {parameters[key]: val for key, val in kwargs.items()}

    initial_voltages = {channel: parameters[channel]() for channel in channels}

    for ratio in np.linspace(0, 1, 11):
        for channel in target_voltages:
            voltage = (1 - ratio) * initial_voltages[channel] + \
                      ratio * target_voltages[channel]
            parameters[channel](voltage)