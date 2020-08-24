from qcodes import VisaInstrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers, Bool, EnumVisa
import numpy as np

class DSG815(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name=name, address=address, terminator='\n', **kwargs)
        self.reset()

        self.power = Parameter('power',
                               parent=self,
                               vals=Numbers(-110, 20),
                               get_cmd=':LEVel?',
                               set_cmd=':LEVel {}',
                               get_parser=float,
                               unit='dBm',
                               label='RF output power')

        self.amplitude_rms = Parameter('amplitude_rms',
                               parent=self,
                               vals=Numbers(1e-6/np.sqrt(2), np.sqrt(10/2)),
                               get_cmd=':LEVel?',
                               set_cmd=':LEVel {}V',
                               get_parser=lambda x: np.sqrt(
                                   10 ** (float(x) / 10) * 1e-3 * 50),
                               unit='V',
                               label='RF output rms amplitude.')

        self.amplitude = Parameter('amplitude',
                               parent=self,
                               vals=Numbers(1e-6, 3.162),
                               get_cmd=self.amplitude_rms,
                               set_cmd=self.amplitude_rms,
                               get_parser=lambda x: x*np.sqrt(2),
                               set_parser=lambda x: x/np.sqrt(2),
                               unit='V',
                               label='RF output peak amplitude.')

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

        self.display_unit = Parameter('display_unit',
                                      parent=self,
                                      vals=EnumVisa('DBM', 'DBMV', 'DBUV', 'V', 'W'),
                                      get_cmd=':UNIT:POWer?',
                                      set_cmd=':UNIT:POWer {}',
                                      label='RF output signal unit',
                                      docstring='Unit for the instrument display.')

    def off(self):
        self.output_on(False)

    def on(self):
        self.output_on(True)

    def reset(self):
        self.write(':SYSTem:CLEar')
