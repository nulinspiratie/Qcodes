from qcodes import VisaInstrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers, Bool, Enum

class DSG815(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name=name, address=address, terminator='\n', **kwargs)
        self.reset()

        self.unit = Parameter('unit',
                              parent=self,
                              vals=Enum('DBM', 'DBMV', 'DBUV', 'V', 'W'),
                              get_cmd=':UNIT:POWer?',
                              set_cmd=':UNIT:POWer {}',
                              label='RF output signal unit',
                              docstring='Unit for the RF output level command to use')

        self.power = Parameter('power',
                               parent=self,
                               vals=Numbers(-110, 20),
                               get_cmd=':LEVel?',
                               set_cmd=':LEVel {}',
                               unit='dBm',
                               label='RF output power')

        self.amplitude = Parameter('amplitude',
                               parent=self,
                               vals=Numbers(1e-6, 3.162),
                               get_cmd=':LEVel?',
                               set_cmd=':LEVel {}V',
                               unit='V',
                               label='RF output Vp')

        self.frequency = Parameter('frequency',
                                   parent=self,
                                   vals=Numbers(9e3, 3e9),
                                   get_cmd=':FREQ?',
                                   set_cmd=':FREQ {}',
                                   unit='Hz',
                                   label='RF output frequency.')

        self.output_on = Parameter('output_on',
                                   parent=self,
                                   vals=Bool(),
                                   get_cmd=':OUTPut?',
                                   get_parser=bool,
                                   set_cmd=':OUTPut {:d}',
                                   label='RF output state')

    def reset(self):
        self.write(':SYSTem:CLEar')
