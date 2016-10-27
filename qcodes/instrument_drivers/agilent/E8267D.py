from numpy import pi

from qcodes import VisaInstrument, validators as vals


class Keysight_E8267D(VisaInstrument):
    '''
    This is the qcodes driver for the Agilent_E8267D PSG vector signal generator

    This driver does not contain all commands available for the E8267D but
    only the ones most commonly used.
    '''
    def __init__(self, name, address, step_attenuator=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='SOURce:FREQuency:CW?',
                           set_cmd='SOURce:FREQuency:CW' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1e5, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='SOURce:PHASe?',
                           set_cmd='SOURce:PHASe' + ' {:.8f}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=vals.Numbers(-180, 180))
        self.add_parameter(name='power',
                           label='Power',
                           units='dBm',
                           get_cmd='POW:AMPL?',
                           set_cmd='POW:AMPL' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-150, 10))
        self.add_parameter('status',
                           get_cmd=':OUTP?',
                           set_cmd='OUTP {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('phase_modulation',
                           get_cmd=':PM:STAT?',
                           set_cmd='PM:STAT {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('output_modulation',
                           get_cmd='OUTPut:MOD?',
                           set_cmd='OUTPut:MOD {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('pulse_modulation',
                           get_cmd='PULM:STAT?',
                           set_cmd='PULM:STAT {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('pulse_modulation_source',
                           get_cmd='PULM:SOURce?',
                           set_cmd='PULM:SOURce {}',
                           vals=vals.Enum('INT', 'internal',
                                          'EXT', 'external',
                                          'scalar', 'SCAL',))
        self.add_parameter('automatic_leveling_control',
                           get_cmd='POW:ALC?',
                           set_cmd='POW:ALC {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('frequency_modulation',
                           get_cmd='FM:STAT?',
                           set_cmd='FM:STAT {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('amplitude_modulation',
                           get_cmd='AM:STAT?',
                           set_cmd='AM:STAT {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('internal_IQ_modulation',
                           get_cmd='SOURce:DM:STATe?',
                           set_cmd='SOURce:DM:STATe {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))
        self.add_parameter('internal_arb_system',
                           get_cmd='SOURce:RADio:ARB:STATe?',
                           set_cmd='SOURce:RADio:ARB:STATe {}',
                           get_parser=self.parse_on_off,
                           # Only listed most common spellings idealy want a
                           # .upper val for Enum or string
                           vals=vals.Enum('on', 'On', 'ON',
                                          'off', 'Off', 'OFF'))




        self.connect_message()

    # Note it would be useful to have functions like this in some module instead
    # of repeated in every instrument driver
    def rad_to_deg(self, angle_rad):
        angle_deg = float(angle_rad)/(2*pi)*360
        return angle_deg

    def deg_to_rad(self, angle_deg):
        angle_rad = float(angle_deg)/360 * 2 * pi
        return angle_rad

    def parse_on_off(self, stat):
        if stat.startswith('0'):
            stat = 'Off'
        elif stat.startswith('1'):
            stat = 'On'
        return stat

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')