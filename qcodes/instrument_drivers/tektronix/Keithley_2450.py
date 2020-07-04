from qcodes import VisaInstrument
from qcodes.utils.validators import Bool, Enum, Ints, MultiType, Numbers


class Keithley_2450(VisaInstrument):
    """
    QCoDeS driver for the Keithley 2450 SMU.

    Written/modified by R.Savytskyy and M.Johnson (23/09/2019)

    NOTE: Not full list of parameters, however basic functions are implemented.
          Needs further testing, but is ready for usage.
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        ### Sense parameters ###
        self.add_parameter('sense_mode',
                           vals=Enum('VOLT', 'CURR', 'RES'),
                           get_cmd=':SENS:FUNC?',
                           get_parser=lambda s: s[:4], # Truncate to 4 chars
                           set_cmd=self._set_sense_mode,
                           label='Sense mode',
                           docstring='Sets the sensing to a voltage, current '
                                     'or resistance.')

        self.add_parameter('sense_value',
                           get_cmd=':READ?',
                           get_parser=float,
                           label='Sense value',
                           docstring='Reading the sensing value of the active sense mode.')

        self.add_parameter('count',
                           vals=Numbers(min_value=1, max_value=300000),
                           get_cmd=':SENS:COUN?',
                           set_cmd=':SENS:COUN {:d}',
                           get_parser=int,
                           set_parser=int,
                           label='Count',
                           docstring='The number of measurements to perform upon request.')

        self.add_parameter('average_count',
                           vals=MultiType(Ints(min_value=1, max_value=100),
                                          Enum('MIN', 'DEF', 'MAX')),
                           get_cmd=self._get_average_count,
                           set_cmd=self._set_average_count,
                           get_parser=int,
                           label='Average count',
                           docstring='The number of measurements to average over.')

        self.add_parameter('average_mode',
                           vals=Enum('MOV', 'REP'),
                           get_cmd=self._get_average_mode,
                           set_cmd=self._set_average_mode,
                           label='Average mode',
                           docstring='A moving filter will average data from sample to sample, \
                           but a true average will not be generated until the chosen count is reached. \
                           A repeating filter will only output an average once all measurement counts \
                           are collected and is hence slower.')

        self.add_parameter('average_enabled',
                           vals=Bool(),
                           get_cmd=self._get_average_state,
                           set_cmd=self._set_average_state,
                           get_parser=bool,
                           label='Average enabled',
                           docstring='If averaging is enabled, each read will '
                                     'be averaged using either a moving or '
                                     'repeated average (see average_mode) for a '
                                     'number of counts (see average_count)')

        self.add_parameter('sense_range_auto',
                           vals=Bool(),
                           get_cmd=self._get_sense_range_auto,
                           set_cmd=self._set_sense_range_auto,
                           get_parser=lambda x: bool(int(x)),
                           label='Sense range auto mode',
                           docstring='This determines if the range for '
                                     'measurements is selected automatically '
                                     '(True) or manually (False).')

        self.add_parameter('sense_range_auto_lower_limit',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_auto_lower_limit,
                           set_cmd=self._set_sense_range_auto_lower_limit,
                           get_parser=float,
                           set_parser=float,
                           label='Auto range lower limit',
                           docstring='This sets the lower limit used when in auto-ranging mode. \
                           The lower this limit requires a longer settling time, and so you can \
                           speed up measurements by choosing a suitably high lower limit.')

        self.add_parameter('sense_range_auto_upper_limit',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_auto_upper_limit,
                           set_cmd=self._set_sense_range_auto_upper_limit,
                           get_parser=float,
                           set_parser=float,
                           label='Auto range upper limit',
                           docstring='This sets the upper limit used when in auto-ranging mode. \
                           This is only used when measuring a resistance.')

        # TODO: needs connection with source range setting
        self.add_parameter('sense_range_manual',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_manual,
                           set_cmd=self._set_sense_range_manual,
                           get_parser=float,
                           set_parser=float,
                           label='Manual range upper limit',
                           docstring='The upper limit of what is being measured when in manual mode')

        self.add_parameter('nplc',
                           vals=Numbers(min_value=0.01, max_value=10),
                           get_cmd=self._get_nplc,
                           set_cmd=self._set_nplc,
                           get_parser=float,
                           set_parser=float,
                           label='Sensed input integration time',
                           docstring='This command sets the amount of time that the input signal is measured. \
                                      The amount of time is specified in parameters that are based on the \
                                      number of power line cycles (NPLCs). Each PLC for 60 Hz is 16.67 ms \
                                      (1/60) and each PLC for 50 Hz is 20 ms (1/50).')

        self.add_parameter('relative_offset',
                           vals=Numbers(),
                           get_cmd=self._get_relative_offset,
                           set_cmd=self._set_relative_offset,
                           get_parser=float,
                           set_parser=float,
                           label='Relative offset value for a measurement.',
                           docstring='This specifies an internal offset that can be applied to measured data')

        self.add_parameter('relative_offset_enabled',
                           vals=Bool(),
                           get_cmd=self._get_relative_offset_state,
                           set_cmd=self._set_relative_offset_state,
                           label='Relative offset enabled',
                           docstring='This determines if the relative offset is to be applied to measurements.')

        self.add_parameter('four_W_mode',
                           vals=Bool(),
                           get_cmd=self._get_four_wire_mode,
                           set_cmd=self._set_four_wire_mode,
                           get_parser = lambda x : bool(int(x)),
                           label='Four-wire sensing state',
                           docstring='This determines whether you sense in '
                                     'four-wire (True) or two-wire (False) mode')

        ### Source parameters ###
        self.add_parameter('source_mode',
                           vals=Enum('VOLT', 'CURR'),
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=self._set_source_mode,
                           label='Source mode',
                           docstring='This determines whether a voltage or current is being sourced.')

        self.add_parameter('source_level',
                           vals=Numbers(),
                           get_cmd=self._get_source_level,
                           set_cmd=self._set_source_level,
                           get_parser=float,
                           set_parser=float,
                           label='Source level',
                           docstring='This sets/reads the output voltage or current level of the source.')

        self.add_parameter('output_on',
                           vals=Bool(),
                           set_cmd=':OUTP:STAT {:d}',
                           get_cmd=':OUTP:STAT?',
                           get_parser=lambda x: bool(int(x)),
                           label='Output on',
                           docstring='Determines whether output is on (True) '
                                     'or off (False)')

        self.add_parameter('source_limit',
                           vals=Numbers(),
                           get_cmd=self._get_source_limit,
                           set_cmd=self._set_source_limit,
                           get_parser=float,
                           set_parser=float,
                           label='Source limit',
                           docstring='The current (voltage) limit when sourcing voltage (current).')

        self.add_parameter('source_limit_tripped',
                           get_cmd=self._get_source_limit_tripped,
                           get_parser = lambda x : bool(int(x)),
                           label='Source limit reached boolean',
                           docstring='Returns True if the source limit has '
                                     'been reached and False otherwise.')

        self.add_parameter('source_range',
                           vals=Numbers(),
                           get_cmd=self._get_source_range,
                           set_cmd=self._set_source_range,
                           label='Source range',
                           docstring='The voltage (current) output range when sourcing a voltage (current).')

        self.add_parameter('source_range_auto',
                           vals=Bool(),
                           get_cmd=self._get_source_range_auto,
                           set_cmd=self._set_source_range_auto,
                           get_parser=lambda x: bool(int(x)),
                           label='Source range auto mode',
                           docstring='Determines if the range for sourcing is selected automatically (True) or manually (False)')

        self.add_parameter('source_read_back',
                           vals=Bool(),
                           get_cmd=self._get_source_read_back,
                           set_cmd=self._set_source_read_back,
                           get_parser=lambda x: bool(int(x)),
                           label='Source read-back',
                           docstring='Determines whether the recorded output '
                                     'is the measured source value \
                           or the configured source value. The former increases the precision, \
                           but slows down the measurements.')

        # Note: delay value for 'MAX' is 10 000 instead of 4.
        self.add_parameter('source_delay',
                           vals=MultiType(Numbers(min_value=0.0, max_value=4.0),
                                          Enum('MIN', 'DEF', 'MAX')),
                           get_cmd=self._get_source_delay,
                           set_cmd=self._set_source_delay,
                           unit='s',
                           label='Source measurement delay',
                           docstring='This determines the delay between the source changing and a measurement \
                           being recorded.')

        # TODO: Is this even needed?
        self.add_parameter('source_delay_auto',
                           get_cmd=self._get_source_delay_auto_state,
                           set_cmd=self._set_source_delay_auto_state,
                           label='Source measurement delay auto state',
                           docstring='This determines the autodelay between '
                                     'the source changing and a measurement '
                                     'being recorded set to state ON/OFF.')

        self.add_parameter('source_overvoltage_protection',
                           vals=Enum('PROT2', 'PROT5', 'PROT10', 'PROT20', 'PROT40', 'PROT60', 'PROT80', 'PROT100',
                                     'PROT120', 'PROT140', 'PROT160', 'PROT180', 'NONE'),
                           get_cmd='SOUR:VOLT:PROT?',
                           set_cmd='SOUR:VOLT:PROT {:s}',
                           label='Source overvoltage protection',
                           docstring='This sets the overvoltage protection setting of the source output. \
                           Overvoltage protection restricts the maximum voltage level that the instrument can source. \
                           It is in effect when either current or voltage is sourced.')

        self.add_parameter('source_overvoltage_protection_tripped',
                           get_cmd='SOUR:VOLT:PROT:TRIP?',
                           get_parser=lambda x: bool(int(x)),
                           label='Source overvoltage protection tripped status',
                           docstring='True if the voltage source exceeded '
                                     'the protection limits, False otherwise.')

        self.add_parameter('voltage',
                           get_cmd=self.get_voltage,
                           set_cmd=self.set_voltage,
                           unit='V',
                           label='Voltage',
                           docstring='A parameter to get and set a voltage. '
                                 'Equivalent to sense_value (source_level) if the '
                                 'sense_mode (source_mode) is set to "VOLT"'
                           )

        self.add_parameter('current',
                           get_cmd=self.get_current,
                           set_cmd=self.set_current,
                           unit='A',
                           label='Current',
                           docstring='A parameter to get and set a current. '
                                 'Equivalent to sense_value (source_level) if the '
                                 'sense_mode (source_mode) is set to "CURR"'
                           )

        self.add_parameter('resistance',
                           get_cmd=self.get_resistance,
                           unit='Ohm',
                           label='Sensed resistance',
                           docstring='A parameter to return a sensed '
                                     'resistance. '
                                     'Equivalent to sense_value if the '
                                     'sense_mode is set to "RES"'
                           )

    ### Functions ###
    def reset(self):
        """
        Resets the instrument. During reset, it cancels all pending commands
        and all previously sent `*OPC` and `*OPC?`
        """
        self.write(':*RST')

    def log_message_count(self, eventType="ALL"):
        return self.ask(f":SYSTem:EVENtlog:COUNt? {eventType}")

    def next_log_message(self, eventType="ALL"):
        return self.ask(f":SYSTem:EVENtlog:NEXT? {eventType}")

    def get_voltage(self):
        """A handy function to return the voltage if in the correct mode
            :return:
                The sensed voltage

            :raise:
                RunTimeError
        """
        if self.sense_mode() == 'VOLT':
            return self.sense_value()
        else:
            raise RuntimeError(f"{self.name} is not configured to sense a "
                               f"voltage.")

    def get_current(self):
        """A handy function to return the current if in the correct mode
            :return:
                The sensed current

            :raise:
                RunTimeError
        """
        if self.sense_mode() == 'CURR':
            return self.sense_value()
        else:
            raise RuntimeError(f"{self.name} is not configured to sense a "
                               f"current.")

    def get_resistance(self):
        """A handy function to return the resistance if in the correct mode
            :return:
                The sensed resistance

            :raise:
                RunTimeError
        """
        if self.sense_mode() == 'RES':
            return self.sense_value()
        else:
            raise RuntimeError(f"{self.name} is not configured to sense a "
                               f"resistance.")

    def set_voltage(self, value):
        """A handy function to set the voltage if in the correct mode
            :raise:
                RunTimeError
        """
        if self.source_mode() == 'VOLT':
            return self.source_level(value)
        else:
            raise RuntimeError(f"{self.name} is not configured to source a "
                               f"voltage.")

    def set_current(self, value):
        """A handy function to set the current if in the correct mode
            :raise:
                RunTimeError
        """
        if self.source_mode() == 'CURR':
            return self.source_level(value)
        else:
            raise RuntimeError(f"{self.name} is not configured to source a "
                               f"current.")

    def _set_source_mode(self, mode):
        # Set the appropriate unit for the source parameter
        if 'VOLT' in mode:
            self.source_level.unit = 'V'
            self.source_range.unit = 'V'
            self.source_limit.unit = 'A'
        else:
            self.source_level.unit = 'A'
            self.source_range.unit = 'A'
            self.source_limit.unit = 'V'
        self.write(f':SOUR:FUNC {mode}')

    def _get_source_level(self):
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}?')

    def _set_source_level(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode} {value:f}')

    def _get_source_limit(self):
        mode = self.source_mode()
        IV = 'I' if mode == 'VOLT' else 'V'
        return self.ask(f':SOUR:VOLT:{IV}LIM?')

    def _set_source_limit(self, value):
        mode = self.source_mode()
        IV = 'I' if mode == 'VOLT' else 'V'
        self.write(f':SOUR:VOLT:{IV}LIM {value}')

    def _get_source_limit_tripped(self):
        mode = self.source_mode()
        IV = 'I' if mode == 'VOLT' else 'V'
        return self.ask(f':SOUR:VOLT:{IV}LIM:TRIP?')

    def _get_source_range(self):
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}:RANG?')

    def _set_source_range(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode}:RANG {value:f}')

    def _get_source_range_auto(self) -> object:
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}:RANG:AUTO?')

    def _set_source_range_auto(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode}:RANG:AUTO {value:d}')

    def _get_source_read_back(self):
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}:READ:BACK?')

    def _set_source_read_back(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode}:READ:BACK {value:d}')

    def _get_source_delay(self):
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}:DEL?')

    def _set_source_delay(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode}:DEL {value}')

    def _get_source_delay_auto_state(self):
        mode = self.source_mode()
        return self.ask(f':SOUR:{mode}:DEL:AUTO?')

    def _set_source_delay_auto_state(self, value):
        mode = self.source_mode()
        self.write(f':SOUR:{mode}:DEL:AUTO {value:d}')

    def _set_sense_mode(self, mode):
        if 'VOLT' in mode:
            self.sense_value.unit = 'V'
            self.sense_range_manual.unit = 'V'
        elif 'CURR' in mode:
            self.sense_value.unit = 'A'
            self.sense_range_manual.unit = 'A'
        elif 'RES' in mode:
            self.sense_value.unit = 'Ohm' # unicode upper-case omega is \u03A9
            self.sense_range_manual.unit = 'Ohm'
        else:
            raise UserWarning('Unknown sense mode')
        self.write(f':SENS:FUNC "{mode:s}"')

    def _get_average_count(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:AVER:COUNT?')

    def _set_average_count(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:AVER:COUNT {value}')

    def _get_average_mode(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:AVER:TCON?')

    def _set_average_mode(self, filter_type):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:AVER:TCON {filter_type}')

    def _get_average_state(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:AVER?')

    def _set_average_state(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:AVER {value:d}')

    def _get_sense_range_auto(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:RANG:AUTO?')

    def _set_sense_range_auto(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:RANG:AUTO {value:d}')

    def _get_sense_range_manual(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:RANG?')

    def _set_sense_range_manual(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:RANG {value:f}')

    def _get_sense_range_auto_lower_limit(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:RANG:AUTO:LLIM?')

    def _set_sense_range_auto_lower_limit(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:RANG:AUTO:LLIM {value:f}')

    def _get_sense_range_auto_upper_limit(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:RANG:AUTO:ULIM?')

    def _set_sense_range_auto_upper_limit(self, value):
        mode = self.sense_mode()
        # TODO: check if auto range upper limit only works for resistance.
        self.write(f':SENS:{mode}:RANG:AUTO:ULIM {value:f}')

    def _get_nplc(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:NPLC?')

    def _set_nplc(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:NPLC {value:f}')

    def _get_relative_offset(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:REL?')

    def _set_relative_offset(self, value):
        mode = self.sense_mode()
        self.write(':SENS:{mode}:REL {value:f}')

    def _get_relative_offset_state(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:REL:STAT?')

    def _set_relative_offset_state(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:REL:STAT {value:d}')

    def _get_four_wire_mode(self):
        mode = self.sense_mode()
        return self.ask(f':SENS:{mode}:RSEN?')

    def _set_four_wire_mode(self, value):
        mode = self.sense_mode()
        self.write(f':SENS:{mode}:RSEN {value:d}')

    ### Other deprecated functions ###
    # deprecated
    def make_buffer(self, buffer_name, buffer_size):
        self.write('TRACe:MAKE {:s}, {:d}'.format(buffer_name, buffer_size))

    # deprecated
    def clear_buffer(self, buffer_name):
        self.write(':TRACe:CLEar {:s}'.format(buffer_name))