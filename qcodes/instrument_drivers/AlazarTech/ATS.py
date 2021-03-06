import socket
import ctypes
from ctypes import wintypes as wt
import logging
import numpy as np
import os

from qcodes.instrument.base import Instrument
from qcodes.utils import validators
from qcodes.instrument.parameter import Parameter, MultiParameter

# TODO(damazter) (C) logging

# these items are important for generalizing this code to multiple alazar cards
# TODO(damazter) (W) remove 8 bits per sample requirement
# TODO(damazter) (W) some alazar cards have a different number of channels :(
# this driver only works with 2-channel cards

# TODO(damazter) (S) tests to do:
# acquisition that would overflow the board if measurement is not stopped
# quickly enough. can this be solved by not reposting the buffers?

logger = logging.getLogger(__name__)

class AlazarTech_ATS(Instrument):
    """
    This is the qcodes driver for Alazar data acquisition cards

    status: beta-version

    this driver is written with the ATS9870 in mind
    updates might/will be necessary for other versions of Alazar cards

    Args:

        name: name for this instrument, passed to the base instrument
        system_id: target system id for this instrument
        board_id: target board id within the system for this instrument
        dll_path: string contianing the path of the ATS driver dll

    """
    # override dll_path in your init script or in the board constructor
    # if you have it somewhere else
    dll_path = 'C:\\WINDOWS\\System32\\ATSApi'

    # override channels in a subclass if needed
    channels = 2

    _success = 512

    _error_codes = {
        513: 'ApiFailed',
        514: 'ApiAccessDenied',
        515: 'ApiDmaChannelUnavailable',
        516: 'ApiDmaChannelInvalid',
        517: 'ApiDmaChannelTypeError',
        518: 'ApiDmaInProgress',
        519: 'ApiDmaDone',
        520: 'ApiDmaPaused',
        521: 'ApiDmaNotPaused',
        522: 'ApiDmaCommandInvalid',
        523: 'ApiDmaManReady',
        524: 'ApiDmaManNotReady',
        525: 'ApiDmaInvalidChannelPriority',
        526: 'ApiDmaManCorrupted',
        527: 'ApiDmaInvalidElementIndex',
        528: 'ApiDmaNoMoreElements',
        529: 'ApiDmaSglInvalid',
        530: 'ApiDmaSglQueueFull',
        531: 'ApiNullParam',
        532: 'ApiInvalidBusIndex',
        533: 'ApiUnsupportedFunction',
        534: 'ApiInvalidPciSpace',
        535: 'ApiInvalidIopSpace',
        536: 'ApiInvalidSize',
        537: 'ApiInvalidAddress',
        538: 'ApiInvalidAccessType',
        539: 'ApiInvalidIndex',
        540: 'ApiMuNotReady',
        541: 'ApiMuFifoEmpty',
        542: 'ApiMuFifoFull',
        543: 'ApiInvalidRegister',
        544: 'ApiDoorbellClearFailed',
        545: 'ApiInvalidUserPin',
        546: 'ApiInvalidUserState',
        547: 'ApiEepromNotPresent',
        548: 'ApiEepromTypeNotSupported',
        549: 'ApiEepromBlank',
        550: 'ApiConfigAccessFailed',
        551: 'ApiInvalidDeviceInfo',
        552: 'ApiNoActiveDriver',
        553: 'ApiInsufficientResources',
        554: 'ApiObjectAlreadyAllocated',
        555: 'ApiAlreadyInitialized',
        556: 'ApiNotInitialized',
        557: 'ApiBadConfigRegEndianMode',
        558: 'ApiInvalidPowerState',
        559: 'ApiPowerDown',
        560: 'ApiFlybyNotSupported',
        561: 'ApiNotSupportThisChannel',
        562: 'ApiNoAction',
        563: 'ApiHSNotSupported',
        564: 'ApiVPDNotSupported',
        565: 'ApiVpdNotEnabled',
        566: 'ApiNoMoreCap',
        567: 'ApiInvalidOffset',
        568: 'ApiBadPinDirection',
        569: 'ApiPciTimeout',
        570: 'ApiDmaChannelClosed',
        571: 'ApiDmaChannelError',
        572: 'ApiInvalidHandle',
        573: 'ApiBufferNotReady',
        574: 'ApiInvalidData',
        575: 'ApiDoNothing',
        576: 'ApiDmaSglBuildFailed',
        577: 'ApiPMNotSupported',
        578: 'ApiInvalidDriverVersion',
        579: ('ApiWaitTimeout: operation did not finish during '
              'timeout interval. Check your trigger.'),
        580: 'ApiWaitCanceled',
        581: 'ApiBufferTooSmall',
        582: ('ApiBufferOverflow:rate of acquiring data > rate of '
              'transferring data to local memory. Try reducing sample rate, '
              'reducing number of enabled channels, increasing size of each '
              'DMA buffer or increase number of DMA buffers.'),
        583: 'ApiInvalidBuffer',
        584: 'ApiInvalidRecordsPerBuffer',
        585: ('ApiDmaPending:Async I/O operation was successfully started, '
              'it will be completed when sufficient trigger events are '
              'supplied to fill the buffer.'),
        586: ('ApiLockAndProbePagesFailed:Driver or operating system was '
              'unable to prepare the specified buffer for DMA transfer. '
              'Try reducing buffer size or total number of buffers.'),
        587: 'ApiWaitAbandoned',
        588: 'ApiWaitFailed',
        589: ('ApiTransferComplete:This buffer is last in the current '
              'acquisition.'),
        590: 'ApiPllNotLocked:hardware error, contact AlazarTech',
        591: ('ApiNotSupportedInDualChannelMode:Requested number of samples '
              'per channel is too large to fit in on-board memory. Try '
              'reducing number of samples per channel, or switch to '
              'single channel mode.')
    }

    _board_names = {
        1: 'ATS850',
        2: 'ATS310',
        3: 'ATS330',
        4: 'ATS855',
        5: 'ATS315',
        6: 'ATS335',
        7: 'ATS460',
        8: 'ATS860',
        9: 'ATS660',
        10: 'ATS665',
        11: 'ATS9462',
        12: 'ATS9434',
        13: 'ATS9870',
        14: 'ATS9350',
        15: 'ATS9325',
        16: 'ATS9440',
        17: 'ATS9410',
        18: 'ATS9351',
        19: 'ATS9310',
        20: 'ATS9461',
        21: 'ATS9850',
        22: 'ATS9625',
        23: 'ATG6500',
        24: 'ATS9626',
        25: 'ATS9360',
        26: 'AXI9870',
        27: 'ATS9370',
        28: 'ATU7825',
        29: 'ATS9373',
        30: 'ATS9416'
    }

    @classmethod
    def find_boards(cls, dll_path=None):
        """
        Find Alazar boards connected

        Args:
            dll_path: (string) path of the Alazar driver dll

        Returns:
            list: list of board info for each connected board
        """
        dll = ctypes.cdll.LoadLibrary(dll_path or cls.dll_path)

        system_count = dll.AlazarNumOfSystems()
        boards = []
        for system_id in range(1, system_count + 1):
            board_count = dll.AlazarBoardsInSystemBySystemID(system_id)
            for board_id in range(1, board_count + 1):
                boards.append(cls.get_board_info(dll, system_id, board_id))
        return boards

    @classmethod
    def get_board_info(cls, dll, system_id, board_id):
        """
        Get the information from a connected Alazar board

        Args:
            dll (string): path of the Alazar driver dll
            system_id: id of the Alazar system
            board_id: id of the board within the alazar system

        Return:

            Dictionary containing

                - system_id
                - board_id
                - board_kind (as string)
                - max_samples
                - bits_per_sample
        """

        # make a temporary instrument for this board, to make it easier
        # to get its info
        board = cls('temp', system_id=system_id, board_id=board_id,
                    server_name=None)
        handle = board._handle
        board_kind = cls._board_names[dll.AlazarGetBoardKind(handle)]

        max_s, bps = board._get_channel_info(handle)
        return {
            'system_id': system_id,
            'board_id': board_id,
            'board_kind': board_kind,
            'max_samples': max_s,
            'bits_per_sample': bps
        }

    def __init__(self, name, system_id=1, board_id=1, dll_path=None, **kwargs):
        super().__init__(name, **kwargs)
        self._ATS_dll = ctypes.cdll.LoadLibrary(dll_path or self.dll_path)

        self._handle = self._ATS_dll.AlazarGetBoardBySystemID(system_id,
                                                              board_id)
        if not self._handle:
            raise Exception('AlazarTech_ATS not found at '
                            'system {}, board {}'.format(system_id, board_id))

        self.buffer_list = []

        # Some ATS models do not support a bwlimit. This flag defines if the
        # ATS supports a bwlimit or not. True by default.
        self._bwlimit_support = True

        # get channel info
        max_s, bps = self._get_channel_info(self._handle)
        self.add_parameter(name='bits_per_sample',
                           set_cmd=None,
                           initial_value=bps)
        self.add_parameter(name='bytes_per_sample',
                           set_cmd=None,
                           initial_value=int((bps + 7)//8))
        self.add_parameter(name='maximum_samples',
                           set_cmd=None,
                           initial_value=max_s)

        # Buffers can be pre-allocated using ATS.preallocate_buffers.
        # See docstring for details
        self._preallocated_buffers = []

    def preallocate_buffers(self, num_buffers: int, samples_per_buffer: int):
        """Pre-allocate buffers for acquisition

        This method is especially useful when using 64-bit Python.
        In this case, the buffer memory address can exceed 32 bits, which
        causes an error because the ATS cannot handle such memory addresses.
        This issue appears more frequently for long acquisitions.

        If the buffers are pre-allocated at the start of a Python session,
        there is a much higher chance that a memory address is available below
        32 bits (the lowest available memory address is chosen).

        Args:
            num_buffers: Number of buffers to pre-allocate.
                An error will be raised in an acquisition if the number of
                required buffers does not match the number of pre-allocated buffers.
            samples_per_buffer: Samples per buffer for each channel.
                An error will be raised in an acquisition if the required
                samples_per_buffer exceeds the value given here.
                This value should therefore be chosen well above the expected
                maximum number of samples per buffer.

        Returns:
            Pre-allocated buffer list
        """
        assert all(not buffer._allocated for buffer in self._preallocated_buffers)

        self._preallocated_buffers = [
            Buffer(
                bits_per_sample=self.bits_per_sample(),
                samples_per_buffer=int(samples_per_buffer),
                number_of_channels=len(self.channels)
            )
            for _ in range(num_buffers)
        ]

    def get_idn(self):
        """
        This methods gets the most relevant information of this instrument

        Returns:

            Dictionary containing

                - 'firmware': None
                - 'model': as string
                - 'serial': board serial number
                - 'vendor': 'AlazarTech',
                - 'CPLD_version': version of the CPLD
                - 'driver_version': version of the driver dll
                - 'SDK_version': version of the SDK
                - 'latest_cal_date': date of the latest calibration (as string)
                - 'memory_size': size of the memory in samples,
                - 'asopc_type': type of asopc (as decimal number),
                - 'pcie_link_speed': the speed of a single pcie link (in GB/s),
                - 'pcie_link_width': number of pcie links
        """
        board_kind = self._board_names[
            self._ATS_dll.AlazarGetBoardKind(self._handle)]

        major = np.array([0], dtype=np.uint8)
        minor = np.array([0], dtype=np.uint8)
        revision = np.array([0], dtype=np.uint8)
        self._call_dll('AlazarGetCPLDVersion',
                       self._handle,
                       major.ctypes.data,
                       minor.ctypes.data)
        cpld_ver = str(major[0])+"."+str(minor[0])

        # Use error_check=False, because in some cases the driver version
        # cannot be obtained.
        self._call_dll('AlazarGetDriverVersion',
                       major.ctypes.data,
                       minor.ctypes.data,
                       revision.ctypes.data, error_check=False)
        driver_ver = str(major[0])+"."+str(minor[0])+"."+str(revision[0])

        self._call_dll('AlazarGetSDKVersion',
                       major.ctypes.data,
                       minor.ctypes.data,
                       revision.ctypes.data)
        sdk_ver = str(major[0])+"."+str(minor[0])+"."+str(revision[0])

        value = np.array([0], dtype=np.uint32)
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000024, 0, value.ctypes.data)
        serial = str(value[0])
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000026, 0, value.ctypes.data)
        latest_cal_date = (str(value[0])[0:2] + "-" +
                           str(value[0])[2:4] + "-" +
                           str(value[0])[4:6])

        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x1000002A, 0, value.ctypes.data)
        memory_size = str(value[0])
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x1000002C, 0, value.ctypes.data)
        asopc_type = str(value[0])

        # see the ATS-SDK programmer's guide
        # about the encoding of the link speed
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000030, 0, value.ctypes.data)
        pcie_link_speed = str(value[0]*2.5/10)+"GB/s"
        self._call_dll('AlazarQueryCapability',
                       self._handle, 0x10000031, 0, value.ctypes.data)
        pcie_link_width = str(value[0])


        return {'firmware': None,
                'model': board_kind,
                'serial': serial,
                'vendor': 'AlazarTech',
                'CPLD_version': cpld_ver,
                'driver_version': driver_ver,
                'SDK_version': sdk_ver,
                'latest_cal_date': latest_cal_date,
                'memory_size': memory_size,
                'asopc_type': asopc_type,
                'pcie_link_speed': pcie_link_speed,
                'pcie_link_width': pcie_link_width}

    def config(self, clock_source=None, sample_rate=None, clock_edge=None,
               decimation=None, coupling=None, channel_range=None,
               impedance=None, bwlimit=None, trigger_operation=None,
               trigger_engine1=None, trigger_source1=None,
               trigger_slope1=None, trigger_level1=None,
               trigger_engine2=None, trigger_source2=None,
               trigger_slope2=None, trigger_level2=None,
               external_trigger_coupling=None, external_trigger_range=None,
               trigger_delay=None, timeout_ticks=None):
        """
        configure the ATS board and set the corresponding parameters to the
        appropriate values.
        For documentation of the parameters, see ATS-SDK programmer's guide

        Args:
            clock_source:
            sample_rate:
            clock_edge:
            decimation:
            coupling:
            channel_range:
            impedance:
            bwlimit:
            trigger_operation:
            trigger_engine1:
            trigger_source1:
            trigger_slope1:
            trigger_level1:
            trigger_engine2:
            trigger_source2:
            trigger_slope2:
            trigger_level2:
            external_trigger_coupling:
            external_trigger_range:
            trigger_delay:
            timeout_ticks:

        Returns:
            None
        """
        # region set parameters from args

        self._set_if_present('clock_source', clock_source)
        self._set_if_present('sample_rate', sample_rate)
        self._set_if_present('clock_edge', clock_edge)
        self._set_if_present('decimation', decimation)

        self._set_list_if_present('coupling', coupling)
        self._set_list_if_present('channel_range', channel_range)
        self._set_list_if_present('impedance', impedance)
        self._set_list_if_present('bwlimit', bwlimit)

        self._set_if_present('trigger_operation', trigger_operation)
        self._set_if_present('trigger_engine1', trigger_engine1)
        self._set_if_present('trigger_source1', trigger_source1)
        self._set_if_present('trigger_slope1', trigger_slope1)
        self._set_if_present('trigger_level1', trigger_level1)

        self._set_if_present('trigger_engine2', trigger_engine2)
        self._set_if_present('trigger_source2', trigger_source2)
        self._set_if_present('trigger_slope2', trigger_slope2)
        self._set_if_present('trigger_level2', trigger_level2)

        self._set_if_present('external_trigger_coupling',
                             external_trigger_coupling)
        self._set_if_present('external_trigger_range',
                             external_trigger_range)
        self._set_if_present('trigger_delay', trigger_delay)
        self._set_if_present('timeout_ticks', timeout_ticks)
        # endregion

        self._call_dll('AlazarSetCaptureClock',
                       self._handle, self.clock_source, self.sample_rate,
                       self.clock_edge, self.decimation)

        for i, ch in enumerate(self.channels):
            self._call_dll('AlazarInputControl',
                           self._handle, 2**i, # Channel in binary format
                           self.parameters['coupling' + ch],
                           self.parameters['channel_range' + ch],
                           self.parameters['impedance' + ch])
            if self._bwlimit_support:
                self._call_dll('AlazarSetBWLimit',
                               self._handle, i - 1,
                               self.parameters['bwlimit' + ch])

        self._call_dll('AlazarSetTriggerOperation',
                       self._handle, self.trigger_operation,
                       self.trigger_engine1, self.trigger_source1,
                       self.trigger_slope1, self.trigger_level1,
                       self.trigger_engine2, self.trigger_source2,
                       self.trigger_slope2, self.trigger_level2)

        self._call_dll('AlazarSetExternalTrigger',
                       self._handle, self.external_trigger_coupling,
                       self.external_trigger_range)

        self._call_dll('AlazarSetTriggerDelay',
                       self._handle, self.trigger_delay)

        self._call_dll('AlazarSetTriggerTimeOut',
                       self._handle, self.timeout_ticks)

        # TODO(damazter) (W) config AUXIO

    def _get_channel_info(self, handle):
        bps = np.array([0], dtype=np.uint8)  # bps bits per sample
        max_s = np.array([0], dtype=np.uint32)  # max_s memory size in samples
        self._call_dll('AlazarGetChannelInfo',
                       handle, max_s.ctypes.data, bps.ctypes.data)
        return max_s[0], bps[0]

    def acquire(self, mode=None, samples_per_record=None,
                records_per_buffer=None, buffers_per_acquisition=None,
                channel_selection=None, transfer_offset=None,
                external_startcapture=None, enable_record_headers=None,
                alloc_buffers=None, fifo_only_streaming=None,
                interleave_samples=None, get_processed_data=None,
                allocated_buffers=None, buffer_timeout=None,
                acquisition_controller=None):
        """
        perform a single acquisition with the Alazar board, and set certain
        parameters to the appropriate values
        for the parameters, see the ATS-SDK programmer's guide

        Args:
            mode:
            samples_per_record:
            records_per_buffer:
            buffers_per_acquisition:
            channel_selection:
            transfer_offset:
            external_startcapture:
            enable_record_headers:
            alloc_buffers:
            fifo_only_streaming:
            interleave_samples:
            get_processed_data:
            allocated_buffers:
            buffer_timeout:
            acquisition_controller: An instance of an acquisition controller
                that handles the dataflow of an acquisition

        Returns:
            Whatever is given by acquisition_controller.post_acquire method
        """
        # region set parameters from args
        self._set_if_present('mode', mode)
        self._set_if_present('samples_per_record', samples_per_record)
        self._set_if_present('records_per_buffer', records_per_buffer)
        self._set_if_present('buffers_per_acquisition',
                             buffers_per_acquisition)
        self._set_if_present('channel_selection', channel_selection)
        self._set_if_present('transfer_offset', transfer_offset)
        self._set_if_present('external_startcapture', external_startcapture)
        self._set_if_present('enable_record_headers', enable_record_headers)
        self._set_if_present('alloc_buffers', alloc_buffers)
        self._set_if_present('fifo_only_streaming', fifo_only_streaming)
        self._set_if_present('interleave_samples', interleave_samples)
        self._set_if_present('get_processed_data', get_processed_data)
        self._set_if_present('allocated_buffers', allocated_buffers)
        self._set_if_present('buffer_timeout', buffer_timeout)

        # endregion
        self.mode._set_updated()
        mode = self.mode.get()
        if mode not in ('TS', 'NPT', 'CS'):
            raise Exception("Only the 'TS', 'CS', 'NPT' modes are implemented "
                            "at this point")

        # -----set final configurations-----

        # Abort any previous measurement
        self._call_dll('AlazarAbortAsyncRead', self._handle)

        # Set record size for NPT mode
        if mode in ['CS', 'NPT']:
            pretriggersize = 0  # pretriggersize is 0 for NPT and CS always
            post_trigger_size = self.samples_per_record._get_byte()
            self._call_dll('AlazarSetRecordSize',
                           self._handle, pretriggersize,
                           post_trigger_size)

        number_of_channels = len(self.channel_selection.get_latest())
        samples_per_buffer = 0
        buffers_per_acquisition = self.buffers_per_acquisition._get_byte()
        samples_per_record = self.samples_per_record._get_byte()
        acquire_flags = (self.mode._get_byte() |
                         self.external_startcapture._get_byte() |
                         self.enable_record_headers._get_byte() |
                         self.alloc_buffers._get_byte() |
                         self.fifo_only_streaming._get_byte() |
                         self.interleave_samples._get_byte() |
                         self.get_processed_data._get_byte())

        # set acquisition parameters here for NPT, TS, CS mode
        if mode == 'NPT':
            records_per_buffer = self.records_per_buffer._get_byte()
            records_per_acquisition = (
                records_per_buffer * buffers_per_acquisition)
            samples_per_buffer = samples_per_record * records_per_buffer

            self._call_dll('AlazarBeforeAsyncRead',
                           self._handle, self.channel_selection,
                           self.transfer_offset, samples_per_record,
                           records_per_buffer, records_per_acquisition,
                           acquire_flags)

        elif mode == 'TS':
            if (samples_per_record % buffers_per_acquisition != 0):
                logging.warning('buffers_per_acquisition is not a divisor of '
                                'samples per record which it should be in '
                                'TS mode, rounding down in samples per buffer '
                                'calculation')
            samples_per_buffer = int(samples_per_record /
                                     buffers_per_acquisition)
            if self.records_per_buffer._get_byte() != 1:
                logging.warning('records_per_buffer should be 1 in TS mode, '
                                'defauling to 1')
                self.records_per_buffer._set(1)
            records_per_buffer = self.records_per_buffer._get_byte()

            self._call_dll('AlazarBeforeAsyncRead',
                           self._handle, self.channel_selection,
                           self.transfer_offset, samples_per_buffer,
                           self.records_per_buffer, buffers_per_acquisition,
                           acquire_flags)

        elif mode == 'CS':
            if self.records_per_buffer._get_byte() != 1:
                logging.warning('records_per_buffer should be 1 in TS mode, '
                                'defauling to 1')
                self.records_per_buffer._set(1)

            samples_per_buffer = samples_per_record

            self._call_dll('AlazarBeforeAsyncRead',
                           self._handle, self.channel_selection,
                           self.transfer_offset, samples_per_buffer,
                           self.records_per_buffer, buffers_per_acquisition,
                           acquire_flags)

        self.samples_per_record._set_updated()
        self.records_per_buffer._set_updated()
        self.buffers_per_acquisition._set_updated()
        self.channel_selection._set_updated()
        self.transfer_offset._set_updated()
        self.external_startcapture._set_updated()
        self.enable_record_headers._set_updated()
        self.alloc_buffers._set_updated()
        self.fifo_only_streaming._set_updated()
        self.interleave_samples._set_updated()
        self.get_processed_data._set_updated()

        # create buffers for acquisition
        self.clear_buffers(free_memory=(not self._preallocated_buffers))

        # make sure that allocated_buffers <= buffers_per_acquisition and
        # buffer acquisition is not in acquire indefinite mode (0x7FFFFFFF)
        if (not self.buffers_per_acquisition._get_byte() == 0x7FFFFFFF) and \
                (self.allocated_buffers._get_byte() >
                     self.buffers_per_acquisition._get_byte()):
            logging.warning(
                "'allocated_buffers' should be smaller than or equal to"
                "'buffers_per_acquisition'. Defaulting 'allocated_buffers' to'"
                "" + str(self.buffers_per_acquisition._get_byte()))
            self.allocated_buffers._set(
                self.buffers_per_acquisition._get_byte())

        allocated_buffers = self.allocated_buffers._get_byte()

        try:
            if self._preallocated_buffers:
                # Buffers are already pre-allocated
                assert allocated_buffers <= len(self._preallocated_buffers)
                max_samples = self._preallocated_buffers[0].samples_per_buffer
                assert samples_per_buffer <= max_samples

                # format the numpy array to a subset of the allocated memory
                for buffer in self._preallocated_buffers[:allocated_buffers]:
                    buffer.create_array(samples_per_buffer=samples_per_buffer)
                    self.buffer_list.append(buffer)
            else:
                # Create new buffers
                for k in range(allocated_buffers):
                    buffer = Buffer(
                        self.bits_per_sample(),
                        samples_per_buffer,
                        number_of_channels
                    )
                    self.buffer_list.append(buffer)
        except:
            self.clear_buffers(free_memory=(not self._preallocated_buffers))
            raise

        # post buffers to Alazar
        for buffer in self.buffer_list:
            self._call_dll('AlazarPostAsyncBuffer',
                           self._handle, buffer.addr, buffer.size_bytes)
        self.allocated_buffers._set_updated()

        # -----start capture here-----
        acquisition_controller.pre_start_capture()
        # call the startcapture method
        self._call_dll('AlazarStartCapture', self._handle)

        acquisition_controller.pre_acquire()
        # buffer handling from acquisition
        buffers_completed = 0
        buffer_timeout = self.buffer_timeout._get_byte()
        self.buffer_timeout._set_updated()

        # Recycle buffers either if using continuous streaming mode or if
        # more buffers are needed than the number of allocated buffers
        buffer_recycling = \
            (self.buffers_per_acquisition._get_byte() == 0x7FFFFFFF) or \
            (self.buffers_per_acquisition._get_byte() >
             self.allocated_buffers._get_byte())

        while acquisition_controller.requires_buffer(buffers_completed):
            buffer = self.buffer_list[buffers_completed % allocated_buffers]

            self._call_dll('AlazarWaitAsyncBufferComplete',
                           self._handle, buffer.addr, buffer_timeout)

            # TODO(damazter) (C) last series of buffers must be handled
            # exceptionally
            # (and I want to test the difference) by changing buffer
            # recycling for the last series of buffers

            # if buffers must be recycled, extract data and repost them
            # otherwise continue to next buffer
            if buffer_recycling:
                acquisition_controller.handle_buffer(buffer.buffer)
                self._call_dll('AlazarPostAsyncBuffer',
                               self._handle, buffer.addr, buffer.size_bytes)
            buffers_completed += 1

        # stop measurement here
        self._call_dll('AlazarAbortAsyncRead', self._handle)

        # -----cleanup here-----
        # extract data if not yet done
        if not buffer_recycling:
            for buffer in self.buffer_list:
                acquisition_controller.handle_buffer(buffer.buffer)

        # free up memory if not using preallocated buffers
        self.clear_buffers(free_memory=(not self._preallocated_buffers))

        # check if all parameters are up to date
        for p in self.parameters.values():
            try:
                p.get()
            except (OSError, ctypes.ArgumentError):
                if p.name == 'IDN':
                    pass
                else:
                    raise

        # return result
        return acquisition_controller.post_acquire()

    def triggered(self):
        """
        Checks if the ATS has received at least one trigger.
        Returns:
            1 if there has been a trigger, 0 otherwise
        """
        return self._call_dll('AlazarTriggered', self._handle,
                              error_check=False)

    def get_status(self):
        """
        Returns t
        Returns:

        """
        return self._call_dll('AlazarGetStatus', self._handle,
                              error_check=False)

    def _set_if_present(self, param_name, value):
        if value is not None:
            self.parameters[param_name]._set(value)

    def _set_list_if_present(self, param_base, values):
        if values is not None:
            # Create list of identical values if a single value is given
            if not isinstance(values, list):
                values = [values] * len(self.channels)
            for val, ch in zip(values, self.channels):
                if param_base + ch in self.parameters.keys():
                    self.parameters[param_base + ch]._set(val)

    def _call_dll(self, func_name, *args, error_check=True):
        """
        Execute a dll function `func_name`, passing it the given arguments

        For each argument in the list
        - If an arg is a parameter of this instrument, the parameter
          value from `._get_bytes()` is used. If the call succeeds, these
          parameters will be marked as updated using their `._set_updated()`
          method
        - Otherwise the arg is used directly
        """
        # create the argument list
        args_out = []
        update_params = []
        for arg in args:
            if isinstance(arg,AlazarParameter):
                args_out.append(arg._get_byte())
                update_params.append(arg)
            else:
                args_out.append(arg)

        # run the function
        func = getattr(self._ATS_dll, func_name)
        return_code = func(*args_out)

        # check for errors
        if error_check and (return_code not in [self._success, 518]):
            # TODO(damazter) (C) log error

            argrepr = repr(args_out)
            if len(argrepr) > 100:
                argrepr = argrepr[:96] + '...]'

            if return_code not in self._error_codes:
                raise RuntimeError(
                    'unknown error {} from function {} with args: {}'.format(
                        return_code, func_name, argrepr))
            raise RuntimeError(
                'error {}: {} from function {} with args: {}'.format(
                    return_code, self._error_codes[return_code], func_name,
                    argrepr))
        elif not error_check:
            return return_code

        # mark parameters updated (only after we've checked for errors)
        for param in update_params:
            param._set_updated()

    def clear_buffers(self, free_memory=True):
        """
        This method uncommits all buffers that were committed by the driver.
        This method only has to be called when the acquistion crashes, otherwise
        the driver will uncommit the buffers itself
        :return: None
        """
        if free_memory:
            for b in self.buffer_list:
                b.free_mem()
        self.buffer_list = []

    def signal_to_volt(self, channel, signal):
        """
        convert a value from a buffer to an actual value in volts based on the
        ranges of the channel

        Args:
            channel: number of the channel where the signal value came from
            signal: the value that needs to be converted

        Returns:
             the corresponding value in volts
        """
        # TODO(damazter) (S) check this
        # TODO(damazter) (M) use byte value if range{channel}
        return (((signal - 127.5) / 127.5) *
                (self.parameters['channel_range' + str(channel)].get()))

    def get_sample_rate(self):
        """
        Obtain the effective sampling rate of the acquisition
        based on clock speed and decimation

        Returns:
            the number of samples (per channel) per second
        """
        if self.sample_rate.get() == 'EXTERNAL_CLOCK':
            raise Exception('External clock is used, alazar driver '
                            'could not determine sample speed.')

        rate = self.sample_rate.get()
        if rate == '1GHz_REFERENCE_CLOCK':
            rate = 1e9

        decimation = self.decimation.get()
        if decimation > 0:
            return rate / decimation
        else:
            return rate


class AlazarParameter(Parameter):
    """
    This class represents of many parameters that are relevant for the Alazar
    driver. This parameters only have a private set method, because the values
    are set by the Alazar driver. They do have a get function which return a
    human readable value. Internally the value is stored as an Alazar readable
    value.

    These parameters also keep track the up-to-dateness of the value of this
    parameter. If the private set_function is called incorrectly, this parameter
    raises an error when the get_function is called to warn the user that the
    value is out-of-date

    Args:
        name: see Parameter class
        label: see Parameter class
        unit: see Parameter class
        instrument: see Parameter class
        value: default value
        byte_to_value_dict: dictionary that maps byte values (readable to the
            alazar) to values that are readable to humans
        vals: see Parameter class, should not be set if byte_to_value_dict is
            provided
    """
    def __init__(self, name=None, label=None, unit=None, instrument=None,
                 value=None, byte_to_value_dict=None, vals=None,
                 **kwargs):
        if vals is None:
            if byte_to_value_dict is None:
                vals = validators.Anything()
            else:
                # TODO(damazter) (S) test this validator
                vals = validators.Enum(*byte_to_value_dict.values())

        super().__init__(name=name, label=label, unit=unit, vals=vals,
                         instrument=instrument, **kwargs)
        self.instrument = instrument
        self._byte = None
        self._uptodate_flag = False

        # TODO(damazter) (M) check this block
        if byte_to_value_dict is None:
            self._byte_to_value_dict = TrivialDictionary()
            self._value_to_byte_dict = TrivialDictionary()
        else:
            self._byte_to_value_dict = byte_to_value_dict
            self._value_to_byte_dict = {
                v: k for k, v in self._byte_to_value_dict.items()}

        self._set(value)

    def get_raw(self):
        """
        This method returns the name of the value set for this parameter
        :return: value
        """
        # TODO(damazter) (S) test this exception
        if self._uptodate_flag is False:
            raise Exception('The value of this parameter (' + self.name +
                            ') is not up to date with the actual value in '
                            'the instrument.\n'
                            'Most probable cause is illegal usage of ._set() '
                            'method of this parameter.\n'
                            'Don\'t use private methods if you do not know '
                            'what you are doing!')
        return self._byte_to_value_dict[self._byte]

    def _get_byte(self):
        """
        this method gets the byte representation of the value of the parameter
        :return: byte representation
        """
        return self._byte

    def _set(self, value):
        """
        This method sets the value of this parameter
        This method is private to ensure that all values in the instruments
        are up to date always
        :param value: the new value (e.g. 'NPT', 0.5, ...)
        :return: None
        """

        # TODO(damazter) (S) test this validation
        self.validate(value)
        self._byte = self._value_to_byte_dict[value]
        self._uptodate_flag = False
        self._save_val(value)
        return None

    def _set_updated(self):
        """
        This method is used to keep track of which parameters are updated in the
        instrument. If the end-user starts messing with this function, things
        can go wrong.

        Do not use this function if you do not know what you are doing
        :return: None
        """
        self._uptodate_flag = True


class Buffer:
    """
    This class represents a single buffer used for the data acquisition

    Args:
        bits_per_sample: the number of bits needed to store a sample
        samples_per_buffer: the number of samples needed per buffer(per channel)
        number_of_channels: the number of channels that will be stored in the
            buffer
    """
    logger = False

    def __init__(self, bits_per_sample, samples_per_buffer,
                 number_of_channels):

        if os.name != 'nt':
            raise Exception("Buffer: only Windows supported at this moment")

        self.samples_per_buffer = samples_per_buffer
        self.number_of_channels = number_of_channels

        self._allocated = True

        self.bytes_per_sample = int((bits_per_sample + 7)//8)
        self.np_sample_type = {1: np.uint8, 2: np.uint16}[self.bytes_per_sample]

        # try to allocate memory
        mem_commit = 0x1000
        page_readwrite = 0x4

        self.size_bytes = self.bytes_per_sample * samples_per_buffer * \
                          number_of_channels

        # for documentation please see:
        # https://msdn.microsoft.com/en-us/library/windows/desktop/aa366887(v=vs.85).aspx
        # https://stackoverflow.com/questions/61590363/enforce-virtualalloc-address-less-than-32-bits-on-64-bit-machine
        VirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
        VirtualAlloc.argtypes = [wt.LPVOID, ctypes.c_size_t, wt.DWORD, wt.DWORD]
        VirtualAlloc.restype = wt.LPVOID

        self.addr = VirtualAlloc(
            0,
            ctypes.c_size_t(self.size_bytes), 
            mem_commit, 
            page_readwrite
        )

        # Log buffer information
        if self.logger:
            # Divide address by 32 bits. If larger than 1, this results in a BSOD
            address_bits = None if self.addr is None else round(self.addr/2**32, 3)
            message = (
                f'Created buffer '
                f'addr: {self.addr}, '
                f'addr/2**32: {address_bits}, '
                f'allocated: {self._allocated}, '
                f'bytes_per_sample: {self.bytes_per_sample}, '
                f'sample_type: {self.np_sample_type}, '
                f'size_bytes: {self.size_bytes}, '
            )
            if isinstance(self.logger, socket.socket):
                # Send message to a socket
                self.logger.send((message + '\n').encode())
            else:
                # Write message to a file
                self.logger.write(message)
                self.logger.flush()

        if self.addr is None:
            self._allocated = False
            e = ctypes.windll.kernel32.GetLastError()
            raise Exception("Memory allocation error: " + str(e))
        elif self.addr >> 32:
            raise Exception(
                'Memory allcation address exceeds 32 bits. '
                'Raising error to avoid BSOD'
            )

        self.buffer = self.create_array()
        pointer, read_only_flag = self.buffer.__array_interface__['data']

    def create_array(self, samples_per_buffer: int = None):
        """Create a numpy array from (a subset of) the allocated memory

        Args:
            samples_per_buffer: Number of buffer samples.
                Must be less than or equal to the samples_per_buffer used to
                initialize the Buffer.
                If not set, will use the entire allocated memory

        Returns:
            Numpy buffer array
        """
        if samples_per_buffer is not None:
            assert samples_per_buffer <= self.samples_per_buffer

            size_bytes = self.bytes_per_sample * samples_per_buffer * \
                          self.number_of_channels
        else:
            size_bytes = self.size_bytes

        ctypes_array = (ctypes.c_uint8 * size_bytes).from_address(self.addr)
        self.buffer = np.frombuffer(ctypes_array, dtype=self.np_sample_type)

        return self.buffer

    def free_mem(self, addr=None):
        """
        uncommit memory allocated with this buffer object
        :return: None
        """
        mem_release = 0x8000

        if addr is None:
            addr = self.addr

        # for documentation please see:
        # https://msdn.microsoft.com/en-us/library/windows/desktop/aa366892(v=vs.85).aspx
        ctypes.windll.kernel32.VirtualFree.argtypes = [
            ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
        ctypes.windll.kernel32.VirtualFree.restype = ctypes.c_int
        ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(addr), 0, mem_release)
        self._allocated = False

    def __del__(self):
        """
        If python garbage collects this object, __del__ should be called and it
        is the last chance to uncommit the memory to prevent a memory leak.
        This method is not very reliable so users should not rely on this
        functionality
        :return:
        """
        if self._allocated:
            self.free_mem()
            logging.warning(
                'Buffer prevented memory leak; Memory released to Windows.\n'
                'Memory should have been released before buffer was deleted.')


class AcquisitionController(Instrument):
    """
    This class represents all choices that the end-user has to make regarding
    the data-acquisition. this class should be subclassed to program these
    choices.

    The basic structure of an acquisition is:

        - call to AlazarTech_ATS.acquire internal configuration
        - call to acquisitioncontroller.pre_start_capture
        - Call to the start capture of the Alazar board
        - call to acquisitioncontroller.pre_acquire
        - loop over all buffers that need to be acquired
          dump each buffer to acquisitioncontroller.handle_buffer
          (only if buffers need to be recycled to finish the acquisiton)
        - dump remaining buffers to acquisitioncontroller.handle_buffer
          alazar internals
        - return acquisitioncontroller.post_acquire

    Attributes:
        _alazar: a reference to the alazar instrument driver
    """
    def __init__(self, name, alazar_name, **kwargs):
        """
        :param alazar_name: The name of the alazar instrument on the server
        :return: nothing
        """
        super().__init__(name, **kwargs)
        self._alazar = self.find_instrument(alazar_name,
                                            instrument_class=AlazarTech_ATS)

        self._acquisition_settings = {}
        self._fixed_acquisition_settings = {}
        self.add_parameter(name="acquisition_settings",
                           get_cmd=lambda: self._acquisition_settings)

        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisition_settings. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           parameter_class=ATSAcquisitionParameter,
                           acquisition_controller=self)

        # Save bytes_per_sample received from ATS digitizer
        self._bytes_per_sample = self._alazar.bytes_per_sample() * 8

    def _get_alazar(self):
        """
        returns a reference to the alazar instrument. A call to self._alazar is
        quicker, so use that if in need for speed
        :return: reference to the Alazar instrument
        """
        return self._alazar

    def verify_acquisition_settings(self, **kwargs):
        """
        Ensure that none of the fixed acquisition settings are overwritten
        Args:
            **kwargs: List of acquisition settings

        Returns:
            acquisition settings wwith fixed settings
        """
        for key, val in self._fixed_acquisition_settings.items():
            if kwargs.get(key, val) != val:
                logging.warning('Cannot set {} to {}. Defaulting to {}'.format(
                    key, kwargs[key], val))
            kwargs[key] = val
        return kwargs

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
        """
        Updates acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten remain.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings.update(**kwargs)

    def set_acquisition_settings(self, **kwargs):
        """
        Sets acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten are removed.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings = kwargs

    def do_acquisition(self):
        """
        Performs an acquisition using the acquisition settings
        Returns:
            None
        """
        records = self._alazar.acquire(acquisition_controller=self,
                                       **self._acquisition_settings)
        return records

    def requires_buffer(self):
        """
        Check if enough buffers are acquired
        Returns:
            True if more buffers are needed, False otherwise
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def segment_buffer(self, buffer, scale_voltages=True):
        """
        Segments buffers into the distinct channels
        Args:
            buffer: 1D buffer array containing all channels
            scale_voltages: Whether or not to scale data to actual volts
        Returns:
            buffer_segments: Dictionary with items channel_idx: channel_buffer
        """

        buffer_segments = {}
        for ch, ch_idx in enumerate(self.channel_selection):
            buffer_slice = slice(ch * self.samples_per_record,
                            (ch + 1) * self.samples_per_record)
            # TODO int16 conversion necessary but should be done earlier
            buffer_segment = buffer[buffer_slice]

            if scale_voltages:
                # Convert data points from an uint16 to volts
                ch_range = self._alazar.parameters['channel_range'+ch_idx]()
                # Determine value corresponding to zero for unsigned int
                mid_val = 2.**(self._bytes_per_sample-1)
                buffer_segment = (buffer_segment - mid_val) / mid_val * ch_range

            buffer_segments[ch_idx] = buffer_segment
        return buffer_segments

    def pre_start_capture(self):
        """
        Use this method to prepare yourself for the data acquisition
        The Alazar instrument will call this method right before
        'AlazarStartCapture' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def pre_acquire(self):
        """
        This method is called immediately after 'AlazarStartCapture' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def handle_buffer(self, buffer):
        """
        This method should store or process the information that is contained
        in the buffers obtained during the acquisition.

        Args:
            buffer: np.array with the data from the Alazar card

        Returns:
            something, it is ignored in any case
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def post_acquire(self):
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Alazar driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get form the acquisition
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')


class ATSAcquisitionParameter(MultiParameter):
    def __init__(self, acquisition_controller=None, **kwargs):
        self.acquisition_controller = acquisition_controller
        super().__init__(snapshot_value=False,
                         names=[''], shapes=[()], **kwargs)

    @property
    def names(self):
        if self.acquisition_controller is None or \
                not hasattr(self.acquisition_controller, 'channel_selection')\
                or self.acquisition_controller.channel_selection is None:
            return ['']
        else:
            return tuple([f'ch{ch}_signal' for ch in
                          self.acquisition_controller.channel_selection])

    @names.setter
    def names(self, names):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def labels(self):
        return self.names

    @labels.setter
    def labels(self, labels):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def units(self):
        return ['V'] * len(self.names)

    @units.setter
    def units(self, units):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def shapes(self):
        if hasattr(self.acquisition_controller, 'average_mode'):
            average_mode = self.acquisition_controller.average_mode()

            if average_mode == 'point':
                shape = ()
            elif average_mode == 'trace':
                shape = (self.acquisition_controller.samples_per_record,)
            else:
                shape = (self.acquisition_controller.traces_per_acquisition(),
                         self.acquisition_controller.samples_per_record)
            return tuple([shape] * self.acquisition_controller.number_of_channels)
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get_raw(self):
        return self.acquisition_controller.do_acquisition()


class TrivialDictionary:
    """
    This class looks like a dictionary to the outside world
    every key maps to this key as a value (lambda x: x)
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        return item

    def __contains__(self, item):
        # this makes sure that this dictionary contains everything
        return True
