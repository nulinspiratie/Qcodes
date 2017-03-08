from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from functools import partial
try:
    import Signadyne.signadyne.SD_AIN as SD_AIN
    import Signadyne.signadyne.SD_AIN_TriggerMode as SD_TriggerMode
    # TODO: Import all Signadyne classes as themselves
except ImportError:
    raise ImportError('To use a Signadyne Digitizer, install the Signadyne module')

class SD_DIG(Instrument):
    """
    This is the qcodes driver for a generic Signadyne Digitizer of the M32/33XX series.

    Status: pre-alpha

    This driver is written with the M3300A in mind.

    Args:
        name (str)      : the name of the digitizer card
        n_channels (int): the number of digitizer channels for the card 

    """
    def __init__(self, **kwargs):
        """ Initialises a generic Signadyne digitizer and its parameters

            Args:
                name (str)          : the name of the digitizer card
                [n_channels] (int)  : the number of input channels the specified card has
        """
        super().__init__(name, **kwargs)
        self.SD_AIN = SD_AIN()
        self.n_channels = kwargs['n_channels']

        ########################################################################
        ### Create a set of internal variables to aid set/get cmds in params ###
        ########################################################################

        # for triggerIOconfig
        self.__direction                  =  0
        # for clockSetFrequency
        self.__frequency                  =  100e6
        # for clockResetPhase
        self.__trigger_behaviour          =  0 
        self.__PXItrigger                 =  0
        self.__skew                       =  0
        
        # for triggerIOconfig
        self.add_parameter(
            'trigger_direction',
            label='Trigger direction for trigger port',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The trigger direction for digitizer trigger port'
        )

        # for clockSetFrequency
        self.add_parameter(
            'frequency',
            label='CLKsys frequency',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The frequency of internal CLKsys in Hz'
        )

        # for clockResetPhase
        self.add_parameter(
            'trigger_behaviour',
            label='Trigger behaviour for resetting CLKsys phase',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The trigger behaviour for resetting CLKsys phase'
        )

        self.add_parameter(
            'PXI_trigger',
            label='PXI trigger for clockResetPhase',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The PXI trigger which resets CLKsys'
        )

        self.add_parameter(
            'skew',
            label='Skew between PXI_CLK10 and CLKsync',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The skew between PXI_CLK10 and CLKsync in multiples of 10 ns'
        )

        # Create distinct parameters for each of the digitizer channels
        for n in range(n_channels):

            # For channelInputConfig
            self.__full_scale[self, n]               =  1 # By default, full scale = 1V
            self.__impedance[self, n]                =  0 # By default, Hi-z
            self.__coupling[self, n]                 =  0 # By default, DC coupling
            # For channelPrescalerConfig 
            self.__prescaler[self, n]                =  0 # By default, no prescaling
            # For channelTriggerConfig
            self.__trigger_mode[self, n]             =  SD_TriggerMode.RISING_EDGE
            self.__trigger_threshold[self, n]        =  0 # By default, threshold at 0V
            # For DAQconfig
            self.__points_per_cycle[self, n]         =  0
            self.__n_cycles[self, n]                 =  0
            self.__trigger_delay[self, n]            =  0
            self.__trigger_mode[self, n]             =  SD_TriggerMode.RISING_EDGE
            # For DAQtriggerExternalConfig
            self.__digital_trigger_mode[self, n]     =  0 
            self.__digital_trigger_source[self, n]   =  0
            self.__analog_trigger_mask[self, n]      =  0
            # For DAQread
            self.__n_points[self, n]                 =  0
            self.__timeout[self, n]                  = -1

            # For channelInputConfig
            self.add_parameter(
                'full_scale_{}'.format(n),
                label='Full scale range for channel {}'.format(n),
                initial_value=0,
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(SD_AIN.channelInputConfig, channel=n, impedance=self.impedance_),
                get_cmd=partial(channelFullScale, channel=n),
                docstring='The full scale voltage for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'impedance_{}'.format(n),
                label='Impedance for channel {}'.format(n),
                initial_value=0,
                vals=[0,1],
                set_cmd=None,
                get_cmd=None,
                docstring='The input impedance of channel {}'.format(n)
            )

            self.add_parameter(
                'coupling_{}'.format(n),
                label='Coupling for channel {}'.format(n),
                initial_value=0,
                vals=[0,1],
                set_cmd=None,
                get_cmd=None,
                docstring='The coupling of channel {}'.format(n)
            )

            # For channelPrescalerConfig 
            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                initial_value=0,
                vals=range(0,4096),
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(SD_AIN.channelPrescalerConfig,  nChannel=n),
                get_cmd=None,
                docstring='The sampling frequency prescaler for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'trigger_mode_{}'.format(n), label='Trigger mode for channel {}'.format(n), initial_value=0,
                vals=[0,1],
                set_cmd=None,
                get_cmd=None,
                docstring='The trigger mode for channel {}'.format(n)
            )

            self.add_parameter(
                'trigger_threshold_{}'.format(n),
                label='Trigger threshold for channel {}'.format(n),
                initial_value=0,
                vals=Numbers(-3,3),
                set_cmd=None,
                get_cmd=None,
                docstring='The trigger threshold for channel {}'.format(n)
            )

            # For DAQconfig
            self.add_parameter(
                'points_per_cycle_{}'.format(n),
                label='Points per cycle for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The number of points per cycle for DAQ {}'.format(n)
            )

            self.add_parameter(
                'n_cycles_{}'.format(n),
                label='n cycles for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The number of cycles to collect on DAQ {}'.format(n)
            )

            self.add_parameter(
                'trigger_delay_{}'.format(n),
                label='Trigger delay for for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The trigger delay for DAQ {}'.format(n)
            )

            # For DAQtriggerExternalConfig
            self.add_parameter(
                'trigger_mode_{}'.format(n),
                label='Trigger mode for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The trigger mode for DAQ {}'.format(n)
            )

            self.add_parameter(
                'digital_trigger_mode_{}'.format(n),
                label='Digital trigger mode for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The digital trigger mode for DAQ {}'.format(n)
            )

            self.add_parameter(
                'digital_trigger_source_{}'.format(n),
                label='Digital trigger source for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The digital trigger source for DAQ {}'.format(n)
            )

            self.add_parameter(
                'analog_trigger_mask_{}'.format(n),
                label='Analog trigger mask for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The analog trigger mask for DAQ {}'.format(n)
            )

            # For DAQread
            self.add_parameter(
                'n_points_{}'.format(n),
                label='n points for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The number of points to be read using DAQread on DAQ {}'.format(n)
            )

            self.add_parameter(
                'timeout_{}'.format(n),
                label='timeout for channel {}'.format(n),
                initial_value=0,
                vals=Ints(),
                set_cmd=None,
                get_cmd=None,
                docstring='The read timeout for DAQ {}'.format(n)
            )


    def set_trigger_mode(channel, mode=None):
        """ Sets the current trigger mode from those defined in SD_TriggerMode

        Args:
            channel (int)       : the input channel you are modifying
            mode (int)          : the trigger mode drawn from the class SD_TriggerMode
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels)
        if (mode not in SD_trigger_modes):
            raise ValueError("The specified mode {mode} does not exist.".format(mode=mode))
        self.__trigger_mode[self, channel] = mode
        # TODO: Call the SD library to set the current mode


    def get_trigger_mode(channel):
        """ Returns the current trigger mode

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_mode[self, channel]


    def set_trigger_threshold(channel, threshold=0):
        """ Sets the current trigger threshold, in the range of -3V and 3V

        Args:
            channel (int)       : the input channel you are modifying
            threshold (float)   : the value in volts for the trigger threshold
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels)
        if (threshold > 3 or threshold < -3):
            raise ValueError("The specified threshold {thresh} V does not exist.".format(thresh=threshold))
        self.__trigger_threshold[self, channel] = threshold
        # TODO: Call the SD library to set the current threshold


    def get_trigger_threshold(channel):
        """ Returns the current trigger threshold

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_threshold[self, channel]

