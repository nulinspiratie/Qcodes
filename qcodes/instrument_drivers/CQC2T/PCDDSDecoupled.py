from typing import List, Union

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Bool, Ints

import numpy as np

try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')

model_channels = {'M3201A': 4,
                  'M3300A': 4}

"""
Instruction Code Syntax:

Each 32-bit instruction code packet is split into a 10-bit command, 9-bit 
pulse pointer and 13 bits of unused space. They are currently laid out as 
follows:
[(10 bit) Command, (13 bit) unused, (9 bit) pulse pointer]

Within the command, we have the following command breakdown structure

Bit 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 | Instruction
-------------------------------------------------------------------------------
    1 | x | x | x | x | x | x | x | x | x | Software Trigger
    x | 1 | x | x | x | x | x | x | x | s | Set pcdds_enable to s
    x | x | 1 | a | x | x | x | x | x | x | Set output_enable to s
    x | 0 | x | x | 1 | x | x | x | x | x | Write pulse
    x | 0 | x | x | 0 | 1 | a | b | c | d | Set load_delay to abcd
    x | 0 | x | x | 0 | 0 | x | x | 1 | x | Set next pulse

"""


class PCDDSChannel(InstrumentChannel):
    """
    Class that holds all the actions related to a specific PCDDS channel
    """
    max_pulse_idx = 2**11 - 1  # 2k
    max_instruction_idx = 2**15 - 1  # 32k

    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.fpga = self._parent.fpga
        self.id = id
        self.n_pulse_pointer_bits = 11
        self.n_instruction_pointer_bits = 17  # TODO change
        self.n_op_bits = 10
        self.n_phase_bits = 45
        self.n_accum_bits = 45
        self.n_amp_bits = 16
        self.n_timing_bits = 32  # number of bits for pulse to auto-advance
        self.clk = 100e6
        self.v_max = 1.5
        self.f_max = 200e6
        self.fpga_port = self.id

        self.pulse_timing_offset = 14  # Offset for auto-advancing of pulses

        self.add_parameter(
            'output_enable',
            label=f'ch{self.id} output_enable',
            set_cmd=self._set_output_enable,
            vals=Bool(),
            docstring='Whether the system has an enabled output'
        )

        self.add_parameter(
            'load_delay',
            label=f'ch{self.id} load_delay',
            set_cmd=self._set_load_delay,
            vals=Ints(0, 15),
            docstring='How long the delay should be during loading a new pulse '
                      'to calculate the new coefficients (delay in samples).'
        )

        self.add_parameter(
            'pcdds_enable',
            label=f'ch{self.id} pcdds_enable',
            set_cmd=self._set_pcdds_enable,
            vals=Bool(),
            docstring='Set the output of the device to use the PCDDS system '
                      'and not the inbuilt functionality'
        )

        self.add_parameter(
            'pulses',
            label=f'ch{self.id} pulses',
            initial_value=[],
            snapshot_value=False,
            docstring='List of pulses'
        )
        self.add_parameter(
            'instruction_sequence',
            label=f'ch{self.id} instruction sequence of pulses',
            initial_value=[],
            snapshot_value=False,
            docstring='The instruction sequence for all pulses sent to the DDS.'
                      'Gets cleared during clear_memory()'
        )

        self.debug_mode = Parameter(
            initial_value=False,
            set_cmd=None,
            docstring='Print debug messages'
        )

        # Initially set load delay to 10 samples
        self.load_delay(10)

        # Keep the zeroth pulse as an idle pulse
        self.write_zero_pulse(0)

    def _set_output_enable(self, output_enable):
        """
        Set the output enable state of the module
        Args:
            output_enable: (Bool) Is the output enabled
        """
        operation = int('0010000000', 2)
        if output_enable:
            operation += int('0001000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def _set_pcdds_enable(self, pcdds_enable: bool):
        """
        Select the output between the PCDDS system and the passthrough
        connection
        Args:
            pcdds_enable: If true, select the PCDDS system. Otherwise
                          use passthrough
        """
        operation = int('0100000000', 2)
        if pcdds_enable:
            operation += int('0000000001', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def _set_load_delay(self, delay: int):
        """
        Set the delay that the system will apply to calculate all the phase
        and frequency coefficients
        Args:
            delay: Delay in clock cycles
        """
        operation = int('0000010000', 2) + delay
        # Construct and send instruction
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def construct_instruction(self, operation: int, pointer: int) -> int:
        """
        Function to construct the int instruction packet from the operation and
        pointer
        Args:
            operation: Operation that we want to do
            pointer: ID of the pulse that this instruction refers to

        Returns: Instruction

        """
        # TODO should be separate for pulse and instruction memory
        # Check that the pointer is in the allowed range
        assert pointer < 2**self.n_pulse_pointer_bits, \
            f'Pointer with value {pointer} is outside of bounds ' \
            f'[0, {2 ** self.n_pulse_pointer_bits - 1}]'
        # Convert and return
        return (operation << 22) + pointer

    def write_zero_pulse(self, pulse_idx: int):
        """
        Function to write all zeros to a specific memory location
        Args:
            pulse_idx: The location where the zeros are to be written
        """
        self.write_pulse(pulse_idx=pulse_idx, phase=0, frequency=0,
                         frequency_accumulation=0, amplitude=0, offset=0,
                         pulse_timing=0)

    def clear_memory(self, max_pulse_idx=100, max_instruction_idx=200):
        """
        Function to clear all memory
        """
        if max_pulse_idx is None:
            max_pulse_idx = self.max_pulse_idx
        for i in np.arange(max_pulse_idx):
            self.write_zero_pulse(pulse_idx=i)

        if max_instruction_idx is None:
            max_instruction_idx = self.max_instruction_idx
        for i in np.arange(max_instruction_idx):
            self.write_instruction(
                instruction_idx=i,
                pulse_idx=0,
                next_instruction=0
            )

        self.instruction_sequence().clear()

    def write_instruction(self, instruction_idx: int,  pulse_idx: int, next_instruction: int):
        assert instruction_idx <= self.max_instruction_idx

        operation = int('0000110000', 2)
        instruction = (operation << 22) + instruction_idx

        # Construct the pulse parameter to be written to memory
        instruction_data = pulse_idx
        instruction_data += (next_instruction << self.n_pulse_pointer_bits)
        instruction_data = self.split_value(instruction_data)

        self.fpga.set_fpga_pc_port(self.fpga_port, [instruction], self.id, 0, 1)
        for data_row in instruction_data[::-1]:
            self.fpga.set_fpga_pc_port(self.fpga_port, [data_row], self.id, 0, 1)

        # Update self.instruction_sequence
        instruction_dict = {
            'instruction_idx': instruction_idx,
            'pulse_idx': pulse_idx,
            'next_instruction': next_instruction
        }
        for k, existing_instruction in enumerate(self.instruction_sequence()):
            if existing_instruction['instruction_idx'] == instruction_idx:
                self.instruction_sequence[k] = instruction_dict
                break
        else:
            self.instruction_sequence().append(instruction_dict)

        # Print results
        if self.debug_mode():
            print("\nInstruction properties:\n")
            for key, val in instruction_dict.items():
                print(f'\t{key}: {val}')
            print(f'\tPulse data: {instruction_data}')


    # TODO should not be instr, choose different name
    def write_instr(self, instr: dict):
        """
        Function to write an instruction to the pulse memory.

        It takes a dictionary with the following entries:
        - instr: The instruction type. Valid instruction types are 'dc',
                 'sine' and 'chirp'
        - pulse_idx: The pulse id that this instruction is to be stored in
        - phase: The phase of the relevant pulse (in degrees).
            Only applicable to 'sine' and 'chirp' pulses
        - freq: The frequency of this pulse. Only applicable to 'sine' and
                'chirp' pulses
        - accum: The frequency accumulation rate (in Hz/s).
            Only applicable to 'chirp' pulses
        - amp: The amplitude of the pulse
        - next_pulse: The next pulse that should be played after this one
        Args:
            instr: The instruction to write to memory
        """
        if instr['instr'] == 'dc':
            self.write_dc_pulse(
                pulse_idx=instr['pulse_idx'],
                voltage=instr['amp'],
                duration=instr.get('duration', None),
                label=instr.get('label', None)
            )
        elif instr['instr'] == 'sine':
            self.write_sine_pulse(
                pulse_idx=instr['pulse_idx'],
                phase=instr['phase'],
                frequency=instr['freq'],
                amplitude=instr['amp'],
                offset=instr['offset'],
                duration=instr.get('duration', None),
                label=instr.get('label', None)
            )
        elif instr['instr'] == 'chirp':
            self.write_chirp_pulse(
                pulse_idx=instr['pulse_idx'],
                phase=instr['phase'],
                frequency=instr['freq'],
                frequency_accumulation=instr['accum'],
                amplitude=instr['amp'],
                offset=instr['offset'],
                duration=instr.get('duration', None),
                label=instr.get('label', None)
            )
        else:
            raise ValueError(f'Unknown instruction type: {instr["instr"]}')

    def write_sine_pulse(
            self,
            pulse_idx: int,
            phase: float,
            frequency: float,
            amplitude: float,
            offset: float,
            duration: int = None,
            label: str = None
    ):
        """
        Write a normal sinusoidal pulse with the desired properties to the
        pulse memory.
        Args:
            pulse_idx: The location in pulse memory that this is to be written to
            phase: The phase of the signal in degrees
            frequency: The frequency of the signal in Hz
            amplitude: The desired output maximum amplitude in V
            offset: The desired pulse offset in V
        """
        if not isinstance(pulse_idx, int):
            raise TypeError('Incorrect type for function input pulse. It should '
                            'be an int')
        assert np.abs(offset) + np.abs(amplitude) <= self.v_max + 1e-10, \
            'The combined output will be larger than the maximum output voltage'

        # Convert all the pulse parameters to the correct register values
        self.write_pulse(
            pulse_idx=pulse_idx,
            phase=self.phase2val(phase),
            frequency=self.freq2val(frequency),
            frequency_accumulation=0,
            amplitude=self.amp2val(amplitude),
            offset=self.offset2val(offset),
            pulse_timing=self.duration2val(duration),
            label=label
        )

    def write_dc_pulse(
            self,
            pulse_idx: int,
            voltage: float,
            duration: float = None,
            label: str = None):
        """
        Write a DC pulse to memory. This sets up a pulse with a phase offset
        of 90 or 270 degrees, 0 frequency and a certain amplitude
        Args:
            pulse_idx: The location in pulse memory that this is to be
                   written to
            voltage: The desired DC voltage
        """
        if not isinstance(pulse_idx, int):
            raise TypeError('Incorrect type for function input pulse. It '
                            'should be an int')
        # Convert the voltage to the correct register value
        self.write_pulse(
            pulse_idx=pulse_idx,
            phase=0,
            frequency=0,
            frequency_accumulation=0,
            amplitude=0,
            offset=self.offset2val(voltage),
            pulse_timing=self.duration2val(duration),
            label=label
        )

    def write_chirp_pulse(self, pulse_idx: int, phase: float, frequency: float,
                          frequency_accumulation: float, amplitude: float,
                          offset: float, duration: float = None, label: str = None):
        """
        Write a pulse to pulse memory which contains a frequency sweep
        Args:
            pulse_idx: The location in pulse memory that this is to be written to
            phase: The phase value for this pulse in degrees
            frequency: The frequency value for this pulse in Hz
            frequency_accumulation: The frequency accumulation for this pulse
                                    in Hz/s
            amplitude: The amplitude for this pulse in V
            offset: The offset for this pulse in V
        """
        if not isinstance(pulse_idx, int):
            raise TypeError('Incorrect type for function input pulse. '
                            'It should be an int')
        assert np.abs(offset) + np.abs(amplitude) <= self.v_max + 1e-10, \
            'The combined output will be larger than the maximum output voltage'

        self.write_pulse(
            pulse_idx=pulse_idx,
            phase=self.phase2val(phase),
            frequency=self.freq2val(frequency),
            frequency_accumulation=self.accum2val(frequency_accumulation),
            amplitude=self.amp2val(amplitude),
            offset=self.offset2val(offset),
            pulse_timing=self.duration2val(duration),
            label=label
        )

    def write_pulse(self, pulse_idx: int, phase: int, frequency: int,
                    frequency_accumulation: int, amplitude: int, offset: int,
                    pulse_timing: int, label: str = None):
        """
        Function to write a pulse with given register values to a given
        location in pulse memory
        Args:
            pulse_idx: The location in pulse memory that this is to be written to
            phase: The phase register value for this pulse
            frequency: The frequency register value for this pulse
            frequency_accumulation: The frequency accumulation register for
                                    this pulse
            amplitude: The amplitude register for this pulse
            offset: The offset register value for this pulse
        """
        pulse_dict = {
            'pulse_idx': pulse_idx,
            'phase': phase,
            'frequency': frequency,
            'frequency_accumulation': frequency_accumulation,
            'amplitude': amplitude,
            'offset': offset,
            'pulse_timing': pulse_timing,
            'label': label
        }

        # TODO should this be n_pointer_bits?
        if pulse_idx < 0 or pulse_idx > min(2 ** self.n_pulse_pointer_bits - 1, self.max_pulse_idx):
            raise ValueError(
                f'The pulse index is outside of memory. '
                f'It should be between 0 and {2 ** self.n_pulse_pointer_bits - 1}')

        # Construct the initial instruction to write a new pulse to memory
        operation = int('0000100000', 2)
        instr = self.construct_instruction(operation, pulse_idx)

        # Construct the pulse parameter to be written to memory
        pulse_data = phase
        pulse_data += (frequency << self.n_phase_bits)
        pulse_data += (frequency_accumulation << 2 * self.n_phase_bits)
        pulse_data += (amplitude << (2 * self.n_phase_bits + self.n_accum_bits))
        pulse_data += (offset << (2 * self.n_phase_bits + self.n_accum_bits
                       + self.n_amp_bits))
        pulse_data += pulse_timing << (2 * self.n_phase_bits + self.n_accum_bits
                       + 2*self.n_amp_bits)
        pulse_data = self.split_value(pulse_data)

        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)
        for data_row in pulse_data[6::-1]:
            self.fpga.set_fpga_pc_port(self.fpga_port, [data_row], self.id, 0, 1)

        # Update self.pulses
        for k, pulse in enumerate(self.pulses()):
            if pulse['pulse_idx'] == pulse_idx:
                self.pulses()[k] = pulse_dict
                break
        else:
            self.pulses().append(pulse_dict)

        # Print results
        if self.debug_mode():
            print("\nPulse properties:\n")
            for key, val in pulse_dict.items():
                print(f'\t{key}: {val}')
            print(f'\tPulse data: {pulse_data}')

    def split_value(self, value: int) -> List[int]:
        """
        Splits a 24 byte message up into 6x 32 bit messages
        Args:
            value: The message that is to be split

        Returns: List of 32 bit length ints to be sent as messages
        """
        if not isinstance(value, int):
            raise TypeError('Incorrect type passed to split_value')

        bit_messages = [
            int(value & 0xFFFFFFFF),
            int((value >> 1*32) & 0xFFFFFFFF),
            int((value >> 2*32) & 0xFFFFFFFF),
            int((value >> 3*32) & 0xFFFFFFFF),
            int((value >> 4*32) & 0xFFFFFFFF),
            int((value >> 5*32) & 0xFFFFFFFF),
            int((value >> 6*32) & 0xFFFFFFFF)
        ]

        if self.debug_mode():
            print(tuple(hex(bit_message) for bit_message in bit_messages))

        return bit_messages

    def set_next_pulse(self, pulse: int, update: bool):
        """
        Function to set the next pulse to be played. It is also possible to
        update to this new pulse via this function
        Args:
            pulse: Next pulse to be played
            update: Should the system update right now
        """
        operation = int('0000000010', 2)
        if update:
            operation += int('1000000000', 2)
        instr = self.construct_instruction(operation, pulse)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def send_trigger(self):
        """
        Send a trigger signal to the FPGA
        """
        operation = int('1000000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def phase2val(self, phase: float) -> int:
        """
        Function to calculate the correct phase register values for a given
        phase
        Args:
            phase: The desired phase in degrees.

        Returns: The register value for the desired phase
        """
        phase = phase % 360.0
        return int(np.round((2 ** self.n_phase_bits / 360.0) * phase))

    def freq2val(self, freq: float) -> int:
        """Calculates correct frequency register values for a given frequency

        The frequency register is any int between 0 and 2**self.n_phase_bits-1,
        where 0 corresponds to 0 Hz, and 2**n_phase_bits corresponds to the
        maximum frequency (inverse of the maximum sampling rate including any
        interleaving), which is effectively 0 Hz.
        Negative frequencies are effectively:
        (max_frequency + frequency) % 2**n_phase_bits

        Args:
            freq: The desired frequency in Hz

        Returns: The register value for the desired frequency
        """
        assert -self.f_max < freq < self.f_max, \
            f'Frequency of {freq} is outside of allowed values: ' \
            f'[0, {self.f_max/1e6}MHz]'
        bit_value = int(np.round((2**self.n_phase_bits / (5*self.clk)) * freq))
        # If frequency is negative, bit value is also negative.
        # Ensure positive value by taking modulus
        bit_value %= 2**self.n_phase_bits
        return bit_value

    def accum2val(self, accum: float) -> int:
        """
        Function to calculate the correct accumulation register values for a
        given accumulation
        Args:
            accum: The desired accumulation in Hz/s

        Returns: The register value for the desired accumulation
        """
        if accum < 0 or accum > (5*self.clk ** 2):
            raise ValueError(f'Frequency Accumulation of {accum} is outside of '
                             f'allowed values [0,{(5*self.clk ** 2)}Hz/s]')
        return int(np.round(accum * 2 ** self.n_accum_bits / (5*self.clk) ** 2))

    def amp2val(self, amp: float) -> int:
        """
        Function to calculate the correct amplitude register values for a given
        amplitude
        Args:
            amp: The desired amplitude in V

        Returns: The register value for the desired amplitude
        """
        if amp < 0 - 1e-10 or amp > self.v_max + 1e-10:
            raise ValueError(f'Amplitude of {amp} is outside of allowed values '
                             f'[0, {self.v_max}V')
        value = int(np.round((2**self.n_amp_bits-1) * amp/self.v_max))
        # Limit offset
        value = min(value, 2 ** self.n_amp_bits - 1)
        value = max(value, 0)
        return value

    # Limit offset
    def offset2val(self, offset: float) -> int:
        """
        Function to calculate the correct offset register value for a given
        offset
        Args:
            offset: The desired offset in V

        Returns: The register value for the desired offset

        """
        assert -self.v_max -1e-10 <= offset <= self.v_max + 1e-10, \
            f'Offset value is not within allowed bounds of [-{self.v_max}, ' \
            f'{self.v_max}]'

        value = int(np.round(offset/self.v_max * 2**(self.n_amp_bits-1)))
        value = min(value, 2 ** (self.n_amp_bits - 1) - 1)
        value = max(value, -2 ** (self.n_amp_bits - 1))
        # Tricking python into not thinking it's a negative number
        value += 1 << 17
        value &= 0xFFFF
        return value

    def duration2val(self, duration: Union[float, None]):
        if duration is None:
            # Wait for trigger
            return 0
        else:
            cycles = int(round(self.clk * duration))
            cycles -= self.pulse_timing_offset
            assert cycles > 0, f"PCDDS pulse must be at least 140 ns long, not {duration}"
            return cycles


class PCDDS(Instrument):
    """
    This class is the driver for the Phase Coherent Pulse Generation Module
    implemented on the FPGA onboard a Keysight PXI AWG card
    """
    def __init__(self, name: str, FPGA, channels: int = None, **kwargs):
        """ Constructor for the pulse generation modules """
        super().__init__(name,  **kwargs)
        self.fpga = FPGA

        if channels is None:
            channels = model_channels[self.fpga.model]
        self.n_channels = channels

        channels = ChannelList(self,
                               name='channels',
                               chan_type=PCDDSChannel)
        for ch in range(self.n_channels):
            channel = PCDDSChannel(self, name=f'ch{ch}', id=ch)
            setattr(self, f'ch{ch}', channel)
            channels.append(channel)
        self.add_submodule('channels', channels)

    def reset(self):
        """ Sends the reset signal to the FPGA.

        Hasn't been tested if it actually does something.
        """
        self.fpga.reset(reset_mode=keysightSD1.SD_ResetMode.PULSE)

    def get_debug_messages(self, id):
        address_labels = ['pulse_addr_b',
                          'coeff_len',
                          'trigger',
                          'o_en',
                          'pxi_trigger_in',
                          'trigger_in',
                          ]
        return {
            label: self.fpga.get_fpga_pc_port(
                port=id,
                data_size=1,
                address=debug_port,
                address_mode=1,
                access_mode=0
            )[0]
            for debug_port, label in enumerate(address_labels, start=1)}
