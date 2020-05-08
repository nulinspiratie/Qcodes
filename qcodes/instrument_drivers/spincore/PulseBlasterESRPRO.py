import qcodes as qc
import ctypes
from time import sleep

from qcodes import Instrument
import qcodes.utils.validators as vals

from qcodes.instrument_drivers.spincore import spinapi as api

class PulseBlasterESRPRO(Instrument):
    """
    This is the qcodes driver for the PulseBlaster ESR-PRO.
    The driver communicates with the underlying python wrapper spinapi.py,
    which in turn communicates with spinapi.dll.
    Both can be obtained from the manufacturer's website.

    Args:
        name (str): name of the instrument
        api_path(str): Path of the spinapi.py file

    Note that only a single instance can communicate with the PulseBlaster.
    To avoid a locked instrument, always close connection with the Pulseblaster.
    """
    program_instructions_map = {
        'CONTINUE': 0,  #inst_data=Not used
        'STOP': 1,      #inst_data=Not used
        'LOOP': 2,      #inst_data=Number of desired loops
        'END_LOOP': 3,  #inst_data=Address of instruction originating loop
        'JSR': 4,       #inst_data=Address of first instruction in subroutine
        'RTS': 5,       #inst_data=Not Used
        'BRANCH': 6,    #inst_data=Address of instruction to branch to
        'LONG_DELAY': 7,#inst_data=Number of desired repetitions
        'WAIT': 8}      #inst_data=Not used

    def __init__(self, name, board_number=0, **kwargs):
        super().__init__(name, **kwargs)

        # It seems that the core_clock is not the same as the sampling rate.
        # At core_clock(500), the PulseBlaster uses 1 ns per wait duration.
        # The wait duration is inversely proportional to the core clock, in contrast to the sampling rate
        self.add_parameter('core_clock',
                           label='Core clock',
                           unit='MHz',
                           set_cmd=self.set_core_clock,
                           vals=vals.Numbers(0, 500))

        self.add_parameter('board_number',
                           set_cmd=None,
                           initial_value=board_number)

        self.add_function('initialize',
                          call_cmd=self.initialize)

        self.add_function('detect_boards',
                          call_cmd=self.detect_boards)

        self.add_function('select_board',
                          call_cmd=api.pb_select_board,
                          args=[vals.Enum(0, 1, 2, 3, 4)])

        self.add_function('start_programming',
                          call_cmd=self.start_programming)

        self.add_function('send_instruction',
                          call_cmd=self.send_instruction,
                          args=[vals.Ints(), vals.Strings(),
                                vals.Ints(), vals.Ints()])

        self.add_function('stop_programming',
                          call_cmd=self.stop_programming)

        self.add_function('start',
                          call_cmd=self.start)

        self.add_function('stop',
                          call_cmd=self.stop)

        self.add_function('close',
                          call_cmd=self.close)

        self.add_function('get_error',
                          call_cmd=api.pb_get_error)

        self.add_parameter('instruction_sequence',
                           set_cmd=None,
                           initial_value=[],
                           vals=vals.Anything(),
                           snapshot_value=False
                           )

        self.setup(initialize=True)

    def initialize(self):
        '''
        Initializes board. This needs to be performed before any communication with the board is possible
        Raises error if return message indicates an error.

        Returns:
            return_msg
        '''
        self.select_board(self.board_number())
        return_msg = api.pb_init()
        assert return_msg == 0, 'Error initializing board: {}'.format(api.pb_get_error())
        return return_msg

    def detect_boards(self):
        '''
        Detects the number of boards.
        Raises an error if the number of boards is zero, or an error has occurred.
        Returns:
            return_msg
        '''
        return_msg = api.pb_count_boards()
        assert return_msg > 0, 'No boards detected'
        return return_msg

    def setup(self, initialize=False):
        """
        Sets up the board, must be called before programming it
        Args:
            initialize: Whether to initialize (should only be done once at
            the start). False by default

        Returns:

        """
        self.detect_boards()
        if initialize:
            self.initialize()


    def set_core_clock(self, core_clock):
        self.select_board(self.board_number())
        # Does not return value
        api.pb_core_clock(core_clock)

    def start_programming(self):
        '''
        Indicate the start of device programming, after which instruction commands can be sent using PB.send_instruction
        Raises error if return message indicates an error.

        Returns:
            return_msg
        '''

        # Reset instruction sequence
        self.instruction_sequence([])

        # Needs constant PULSE_PROGRAM, which is set to equal 0 in the api)

        self.select_board(self.board_number())
        return_msg = api.pb_start_programming(api.PULSE_PROGRAM)
        assert return_msg == 0, 'Error starting programming: {}'.format(api.pb_get_error())
        return return_msg

    def send_instruction(self, flags, instruction, inst_args, length, log=True):
        '''
        Send programming instruction to Pulseblaster.
        Programming instructions can only be sent after the initial command pb.start_programming.
        Raises error if return message indicates an error.

        The different types of instructions are:
            ['CONTINUE', 'LOOP', 'END_LOOP', 'JSR', "RTS', 'BRANCH', 'LONG_DELAY', "WAIT']
        See manual for detailed description of each of the commands

        Args:
            flags: Bit representation of state of each output 0=low, 1=high
            instruction: Instruction to be sent, case-insensitive (see above for possible instructions)
            inst_args: Accompanying instruction argument, dependent on instruction type
            length: Number of clock cycles to perform instruction
            log: Whether to log to instruction_sequence (True by default)

        Returns:
            return_msg, which contains instruction address
        '''

        # Add instruction to log
        if log:
            self.instruction_sequence(self.instruction_sequence() +
                              [(flags, instruction, inst_args, length)])

        instruction_int = self.program_instructions_map[instruction.upper()]
        #Need to call underlying spinapi because function does not exist in wrapper
        self.select_board(self.board_number())
        return_msg = api.spinapi.pb_inst_pbonly(ctypes.c_uint64(flags),
                                                ctypes.c_int(instruction_int),
                                                ctypes.c_int(inst_args),
                                                ctypes.c_double(length))
        assert return_msg >= 0, \
            'Error sending instruction: {}'.format(api.pb_get_error())
        return return_msg

    def send_instructions(self, *instructions):
        for instruction in instructions:
            self.send_instruction(*instruction, log=False)

        # Add instructions to log
        self.instruction_sequence(list(self.instruction_sequence()) + list(instructions))

    def stop_programming(self):
        '''
        Stop programming. After this function, instructions may not be sent to PulseBlaster
        Raises error if return message indicates an error.

        Returns:
            return_msg
        '''
        self.select_board(self.board_number())
        return_msg = api.pb_stop_programming()
        assert return_msg == 0, 'Error stopping programming: {}'.format(api.pb_get_error())
        return return_msg

    def start(self):
        '''
        Start PulseBlaster sequence.
        Raises error if return message indicates an error.

        Returns:
            return_msg
        '''
        self.select_board(self.board_number())
        return_msg = api.pb_start()
        try:
            assert return_msg == 0, 'Error starting: {}'.format(api.pb_get_error())
        except AssertionError:
            api.pb_stop()
            api.pb_start()
            assert return_msg == 0, 'Error starting: {}'.format(
                api.pb_get_error())
        return return_msg

    def stop(self):
        '''
        Stop PulseBlaster sequence
        Raises error if return message indicates an error.

        Returns:
            return_msg
        '''
        self.select_board(self.board_number())
        return_msg = api.pb_stop()
        assert return_msg == 0, 'Error Stopping: {}'.format(api.pb_get_error())
        return return_msg

    def close(self):
        '''
        Terminate communication with PulseBlaster.
        After this command, communication is no longer possible with PulseBlaster

        Returns:
            None
        '''
        self.select_board(self.board_number())
        api.pb_close()
        super().close()
