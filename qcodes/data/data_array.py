import numpy as np
import collections
import warnings
from typing import Iterable

from qcodes.utils.helpers import DelegateAttributes, full_class, warn_units, \
    smooth


class DataArray(DelegateAttributes):

    """
    A container for one parameter in a measurement loop.

    If this is a measured parameter, This object doesn't contain
    the data of the setpoints it was measured at, but it references
    the DataArray objects of these parameters. Those objects only have
    the dimensionality at which they were set - ie the inner loop setpoint
    the same dimensionality as the measured parameter, but the outer
    loop setpoint(s) have lower dimensionality

    When it's first created, a DataArray has no dimensionality, you must call
    .nest for each dimension.

    If preset_data is provided it is used to initialize the data, and the array
    can still be nested around it (making many copies of the data).
    Otherwise it is an error to nest an array that already has data.

    Once the array is initialized, a DataArray acts a lot like a numpy array,
    because we delegate attributes through to the numpy array

    Args:
        parameter (Optional[Parameter]): The parameter whose values will
            populate this array, if any. Will copy ``name``, ``full_name``,
            ``label``, ``unit``, and ``snapshot`` from here unless you
            provide them explicitly.

        name (Optional[str]): The short name of this array.
            TODO: use full_name as name, and get rid of short name

        full_name (Optional[str]): The complete name of this array. If the
            array is based on a parameter linked to an instrument, this is
            typically '<instrument_name>_<param_name>'

        label (Optional[str]): A description of the values in this array to
            use for axis and colorbar labels on plots.

        snapshot (Optional[dict]): Metadata snapshot to save with this array.

        array_id (Optional[str]): A name for this array that's unique within
            its ``DataSet``. Typically the full_name, but when the ``DataSet``
            is constructed we will append '_<i>' (``i`` is an integer starting
            from 1) if necessary to differentiate arrays with the same id.
            TODO: this only happens for arrays provided to the DataSet
            constructor, not those added with add_array. Fix this!
            Also, do we really need array_id *and* full_name (let alone name
            but I've already said we should remove this)?

        set_arrays (Optional[Tuple[DataArray]]): If this array is being
            created with shape already, you can provide one setpoint array
            per dimension. The first should have one dimension, the second
            two dimensions, etc.

        shape (Optional[Tuple[int]]): The shape (as in numpy) of the array.
            Will be prepended with new dimensions by any calls to ``nest``.

        action_indices (Optional[Tuple[int]]): If used within a ``Loop``,
            these are the indices at each level of nesting within the
            ``Loop`` of the loop action that's populating this array.
            TODO: this shouldn't be in DataArray at all, the loop should
            handle converting this to array_id internally (maybe it
            already does?)

        unit (Optional[str]): The unit of the values stored in this array.

        units (Optional[str]): DEPRECATED, redirects to ``unit``.

        is_setpoint (bool): True if this is a setpoint array, False if it
            is measured. Default False.

        preset_data (Optional[Union[ndarray, sequence]]): Contents of the
            array, if already known (for example if this is a setpoint
            array). ``shape`` will be inferred from this array instead of
            from the ``shape`` argument.
    """

    # attributes of self to include in the snapshot
    SNAP_ATTRS = (
        'array_id',
        'name',
        'shape',
        'unit',
        'label',
        'action_indices',
        'is_setpoint')

    # attributes of the parameter (or keys in the incoming snapshot)
    # to copy to DataArray attributes, if they aren't set some other way
    COPY_ATTRS_FROM_INPUT = (
        'name',
        'label',
        'unit')

    # keys in the parameter snapshot to omit from our snapshot
    SNAP_OMIT_KEYS = (
        'ts',
        'value',
        '__class__',
        'set_arrays',
        'shape',
        'array_id',
        'action_indices')

    def __init__(self, parameter=None, name=None, full_name=None, label=None,
                 snapshot=None, array_id=None, set_arrays=(), shape=None,
                 action_indices=(), unit=None, units=None, is_setpoint=False,
                 preset_data=None):
        self.name = name
        self.full_name = full_name or name
        self.label = label
        self.shape = shape
        if units is not None:
            warn_units('DataArray', self)
            if unit is None:
                unit = units
        self.unit = unit
        self.array_id = array_id
        self.is_setpoint = is_setpoint

        if isinstance(action_indices, Iterable):
            # Cast action indices into tuple if possible
            self.action_indices = tuple(action_indices)
        else:
            self.action_indices = action_indices
        self.set_arrays = set_arrays

        self._preset = False

        # store a reference up to the containing DataSet
        # this also lets us make sure a DataArray is only in one DataSet
        self._data_set = None

        self.last_saved_index = None
        self.modified_range = None

        self.ndarray = None
        if snapshot is None:
            snapshot = {}
        self._snapshot_input = {}

        if parameter is not None:
            param_full_name = getattr(parameter, 'full_name', None)
            if param_full_name and not full_name:
                self.full_name = parameter.full_name

            if hasattr(parameter, 'snapshot') and not snapshot:
                snapshot = parameter.snapshot()
            else:
                # TODO: why is this in an else clause?
                for attr in self.COPY_ATTRS_FROM_INPUT:
                    if (hasattr(parameter, attr) and
                            not getattr(self, attr, None)):
                        setattr(self, attr, getattr(parameter, attr))

        for key, value in snapshot.items():
            if key not in self.SNAP_OMIT_KEYS:
                self._snapshot_input[key] = value

                if (key in self.COPY_ATTRS_FROM_INPUT and
                        not getattr(self, key, None)):
                    setattr(self, key, value)

        if not self.label:
            self.label = self.name

        if preset_data is not None:
            self.init_data(preset_data)
        elif shape is None:
            self.shape = ()

    @property
    def data_set(self):
        """
        The DataSet this array belongs to.

        A DataArray can belong to at most one DataSet.
        TODO: make this a weakref
        """
        return self._data_set

    @data_set.setter
    def data_set(self, new_data_set):
        if (self._data_set is not None and
                new_data_set is not None and
                self._data_set != new_data_set):
            raise RuntimeError('A DataArray can only be part of one DataSet')
        self._data_set = new_data_set

    def nest(self, size, action_index=None, set_array=None):
        """
        Nest this array inside a new outer loop.

        You cannot call ``nest`` after ``init_data`` unless this is a
        setpoint array.
        TODO: is this restriction really useful? And should we maintain
        a distinction between _preset and is_setpoint, or can wejust use
        is_setpoint?

        Args:
            size (int): Length of the new loop.

            action_index (Optional[int]): Within the outer loop at this
                nesting level, which action does this array derive from?

            set_array (Optional[DataArray]): The setpoints of the new outer
                loop. If this DataArray *is* a setpoint array, you should
                omit both ``action_index`` and ``set_array``, and it will
                reference itself as the inner setpoint array.

        Returns:
            DataArray: self, in case you want to construct the array with
                chained method calls.
        """
        if self.ndarray is not None and not self._preset:
            raise RuntimeError('Only preset arrays can be nested after data '
                               'is initialized! {}'.format(self))

        if set_array is None:
            if self.set_arrays:
                raise TypeError('a setpoint array must be its own inner loop')
            set_array = self

        self.shape = (size, ) + self.shape

        if action_index is not None:
            self.action_indices = (action_index, ) + self.action_indices

        self.set_arrays = (set_array, ) + self.set_arrays

        if self._preset:
            inner_data = self.ndarray
            self.ndarray = np.ndarray(self.shape)
            # existing preset array copied to every index of the nested array.
            for i in range(size):
                self.ndarray[i] = inner_data

            # update modified_range so the entire array still looks modified
            self.modified_range = (0, self.ndarray.size - 1)

            self._set_index_bounds()

        return self

    def init_data(self, data=None):
        """
        Create the actual numpy array to hold data.

        The array will be sized based on either ``self.shape`` or
        data provided here.

        Idempotent: will do nothing if the array already exists.

        If data is provided, this array is marked as a preset
        meaning it can still be nested around this data.
        TODO: per above, perhaps remove this distinction entirely?

        Args:
            data (Optional[Union[ndarray, sequence]]): If provided,
                we fill the array with this data. Otherwise the new
                array will be filled with NaN.

        Raises:
            ValueError: if ``self.shape`` does not match ``data.shape``
            ValueError: if the array was already initialized with a
                different shape than we're about to create
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                if isinstance(data, collections.Iterator):
                    # faster than np.array(tuple(data)) (or via list)
                    # but requires us to assume float
                    data = np.fromiter(data, float)
                else:
                    data = np.array(data)

            if self.shape is None:
                self.shape = data.shape
            elif data.shape != self.shape:
                raise ValueError('preset data must be a sequence '
                                 'with shape matching the array shape',
                                 data.shape, self.shape)
            self.ndarray = data
            self._preset = True

            # mark the entire array as modified
            self.modified_range = (0, data.size - 1)

        elif self.ndarray is not None:
            if self.ndarray.shape != self.shape:
                raise ValueError('data has already been initialized, '
                                 'but its shape doesn\'t match self.shape')
            return
        else:
            self.ndarray = np.ndarray(self.shape)
            self.clear()
        self._set_index_bounds()

    def _set_index_bounds(self):
        self._min_indices = [0 for d in self.shape]
        self._max_indices = [d - 1 for d in self.shape]

    def clear(self):
        """Fill the (already existing) data array with nan."""
        # only floats can hold nan values. I guess we could
        # also raise an error in this case? But generally float is
        # what people want anyway.
        if self.ndarray.dtype != float:
            self.ndarray = self.ndarray.astype(float)
        self.ndarray.fill(float('nan'))

    def __setitem__(self, loop_indices, value):
        """
        Set data values.

        Follows numpy syntax, allowing indices of lower dimensionality than
        the array, if value makes up the extra dimension(s)

        Also update the record of modifications to the array. If you don't
        want this overhead, you can access ``self.ndarray`` directly.
        """
        if isinstance(loop_indices, collections.Iterable):
            min_indices = list(loop_indices)
            max_indices = list(loop_indices)
        else:
            min_indices = [loop_indices]
            max_indices = [loop_indices]

        for i, index in enumerate(min_indices):
            if isinstance(index, slice):
                start, stop, step = index.indices(self.shape[i])
                min_indices[i] = start
                max_indices[i] = start + (
                    ((stop - start - 1)//step) * step)

        min_li = self.flat_index(min_indices, self._min_indices)
        max_li = self.flat_index(max_indices, self._max_indices)
        self._update_modified_range(min_li, max_li)

        self.ndarray.__setitem__(loop_indices, value)

    def __getitem__(self, loop_indices):
        """
        Get a sub_array, which is also a DataArray.
        The behaviour is similar to np.array slicing, e.g.
            arr[4,:2] gets row 4, columns 0:2
        Set arrays, etc. are also included in the new DataArray
        Args:
            loop_indices: array slice

        Returns:
            Sliced DataArray
        """
        if not isinstance(loop_indices, tuple):
            # Return single element in first dimension
            loop_indices = loop_indices,

        sub_array = self.ndarray[loop_indices]

        if np.shape(loop_indices[0]) == self.shape and \
                isinstance(loop_indices[0], np.ndarray):
            return sub_array
        elif not sub_array.shape:
            # Single element
            return sub_array

        # Generate new set arrays
        sub_set_arrays = []
        for k, set_array in enumerate(self.set_arrays):
            if set_array is self:
                sub_set_array = self
            else:
                sub_set_array = set_array[loop_indices[:set_array.ndim]]
            # NOTE: This failed when trying to using advanced indexing
            # (e.g. using an integer or boolean array), so I've added this extra
            # option, although I don't see why we want to discard set arrays.
            if len(loop_indices) <= k or isinstance(loop_indices[k], slice) or \
                isinstance(loop_indices[k], np.ndarray):
                sub_set_arrays.append(sub_set_array)

        data_array = DataArray(name=self.name, label=self.label, unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=sub_array, set_arrays=sub_set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def array_slice(self, slice):
        """
        Get a numpy sub_array, as opposed to __getitem__ notation,
        which transforms the sub_array to a DataArray.
        This method may be preferred for speed optimization
        Args:
            slice: array slice

        Returns:
            numpy ndarray slice
        """
        return self.ndarray[slice]

    delegate_attr_objects = ['ndarray']

    def __len__(self):
        """
        Array length.

        Must be explicitly delegated, because len() will look for this
        attribute to already exist.
        """
        return len(self.ndarray)

    def mean(self, axis=None, dtype=None, out=None,
             min_filter=None, max_filter=None, **kwargs):
        if isinstance(self, np.ndarray): # Mean called with DataArray.mean
            arr = self
        else:
            arr = self.ndarray.copy()

        # Catch runtime warnings as we may compare NaN or take mean of all Nan's
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if min_filter is not None:
                arr[arr < min_filter] = np.nan
            if max_filter is not None:
                arr[arr > max_filter] = np.nan

            if isinstance(axis, int):
                sub_array = np.nanmean(arr, axis=axis, dtype=dtype, out=out, **kwargs)
                axis = (axis, )
            elif axis is not None:
                for k, ax in enumerate(axis):
                    arr = np.rollaxis(arr, ax, k)
                reshaped_arr = np.reshape(arr, (-1,) + arr.shape[len(axis):])
                sub_array = np.nanmean(reshaped_arr, axis=0, dtype=dtype, out=out, **kwargs)

        if axis is None:
            return np.nanmean(arr)
        elif not isinstance(sub_array, np.ndarray):
            return sub_array
        elif isinstance(self, np.ndarray):
            return sub_array
        else:
            sub_set_arrays = []
            for k, set_array in enumerate(self.set_arrays):
                if k in axis:
                    continue
                else:
                    loop_indices = [slice(None, None, None)] * set_array.ndim
                    for ax in axis:
                        if ax < set_array.ndim:
                            loop_indices[ax] = 0
                    sub_set_arrays.append(set_array[tuple(loop_indices)])
            return DataArray(name=self.name, label=self.label, unit=self.unit,
                             is_setpoint=self.is_setpoint,
                             preset_data=sub_array,
                             set_arrays=sub_set_arrays)

    def smooth(self, window_size, axis=-1, order=3, deriv=0, rate=1):
        if len(self.shape) == 1:
            smoothed_array = smooth(self.ndarray,
                                    window_size=window_size,
                                    order=order,
                                    deriv=deriv,
                                    rate=rate)


        elif len(self.shape) == 2:
            if axis == 0:
                arr = self.ndarray.transpose()
            else:
                arr = self.ndarray
            smoothed_array = np.array([smooth(row,
                                              window_size=window_size,
                                              order=order,
                                              deriv=deriv,
                                              rate=rate)
                                       for row in arr])
        else:
            raise RuntimeError('Smoothing works for up to 2 dimensional arrays,'
                               f'not {len(self.shape)}')

        return DataArray(name=self.name, label=self.label, unit=self.unit,
                         is_setpoint=self.is_setpoint,
                         preset_data=smoothed_array,
                         set_arrays=self.set_arrays)

    def unroll(self, *axes, flatten=False):
        """ Return  a list containing all elements along an axis.
        
        If multiple axes are passed, each element in the list is again unrolled
        by the second axis, etc.
        If flatten is True when multiple axes are passed, the unrolled elements are
        put into a 1D list rather than nesting lists.
        """
        axis = axes[0]
        elems = self.shape[axis]
        leading_indices = (slice(None),)*axis
        unrolled_arr = [self[leading_indices + (elem,)] for elem in range(elems)]

        if len(axes) > 1:
            unrolled_arr = [subarr.unroll(*axes[1:], flatten=flatten) for subarr in unrolled_arr]

            if flatten:
                # For multi-axis unrolls, reduces the list dimensionality to 1D
                # This method is likely the most efficient
                # https://stackoverflow.com/questions/29244286/how-to-flatten-a-2d-list-to-1d-without-using-numpy
                unrolled_arr = [j for i in unrolled_arr for j in i]

        return unrolled_arr

    def transpose(self, axes=None, conjugate=False):
        """ Returns a new data_array which has been transposed.
        The set arrays are also appropriately reconstructed,
        assuming that the elements are repeated.
        """
        if axes == None:
            N = np.ndim(self)
            if N >= 2:
                axes = tuple(k for k in range(N-2))
                axes += (N-1, N-2) # By default, swap final 2 dimensions
            else:
                axes = (0,)
        axes = tuple(axes)
        new_array = np.transpose(self.ndarray, axes)

        if conjugate:
            new_array = np.conj(new_array)

        # Generate new set arrays
        set_arrays = ()
        cum_shape = ()
        for k, axis in enumerate(axes):
            set_array = self.set_arrays[axis]


            N = np.ndim(set_array)

            # By default use the same set array
            new_set_array = set_array

            if k+1 != N:
                # If this axis has been shifted, rebuild the set array for new position.
                if N > 1:
                    # Slice over each tiling of the set values to get 1d array
                    set_array_1d = np.squeeze(set_array[(0,) * (N-1)])
                    # Re-tile the set_array to match the new array shapes
                    new_set_array = np.tile(set_array_1d, cum_shape + (1,))
                else:
                    new_set_array = np.tile(set_array, cum_shape + (1,))


            # Create a new set_array with the previous set_arrays as its 'parents'
            new_set_array = DataArray(name=set_array.name,
                                      label=set_array.label,
                                      unit=set_array.unit,
                                      is_setpoint=set_array.is_setpoint,
                                      preset_data=new_set_array,
                                      set_arrays=set_arrays
                        )
            set_arrays += new_set_array,
            cum_shape += set_array.shape[-1],

        data_array = DataArray(name=self.name, label=self.label, unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array, set_arrays=set_arrays)
        data_array.data_set = self.data_set
        return data_array


    def flat_index(self, indices, index_fill=None):
        """
        Generate the raveled index for the given indices.

        This is the index you would have if the array is reshaped to 1D,
        looping over the indices from inner to outer.

        Args:
            indices (sequence): indices of an element or slice of this array.

            index_fill (sequence, optional): extra indices to use if
                ``indices`` has less dimensions than the array, ie it points
                to a slice rather than a single element. Use zeros to get the
                beginning of this slice, and [d - 1 for d in shape] to get the
                end of the slice.

        Returns:
            int: the resulting flat index.
        """
        if len(indices) < len(self.shape):
            indices = indices + index_fill[len(indices):]
        return np.ravel_multi_index(tuple(zip(indices)), self.shape, mode='wrap')[0]

    def _update_modified_range(self, low, high):
        if self.modified_range:
            self.modified_range = (min(self.modified_range[0], low),
                                   max(self.modified_range[1], high))
        else:
            self.modified_range = (low, high)

    def mark_saved(self, last_saved_index):
        """
        Mark certain outstanding modifications as saved.

        Args:
            last_saved_index (int): The flat index of the last point
                saved. If ``modified_range`` extends beyond this, the
                data past ``last_saved_index`` will still be marked
                modified, otherwise ``modified_range`` is cleared
                entirely.
        """
        if self.modified_range:
            if last_saved_index >= self.modified_range[1]:
                self.modified_range = None
            else:
                self.modified_range = (max(self.modified_range[0],
                                           last_saved_index + 1),
                                       self.modified_range[1])
        self.last_saved_index = last_saved_index

    def clear_save(self):
        """
        Make previously saved parts of this array look unsaved (modified).

        This can be used to force overwrite or rewrite, like if we're
        moving or copying the ``DataSet``.
        """
        if self.last_saved_index is not None:
            self._update_modified_range(0, self.last_saved_index)

        self.last_saved_index = None

    def get_synced_index(self):
        """
        Get the last index which has been synced from the server.

        Will also initialize the array if this hasn't happened already.
        TODO: seems hacky to init_data here.

        Returns:
            int: the last flat index which has been synced from the server,
                or -1 if no data has been synced.
        """
        if not hasattr(self, 'synced_index'):
            self.init_data()
            self.synced_index = -1

        return self.synced_index

    def get_changes(self, synced_index):
        """
        Find changes since the last sync of this array.

        Args:
            synced_index (int): The last flat index which has already
                been synced.

        Returns:
            Union[dict, None]: None if there is no new data. If there is,
                returns a dict with keys:
                    start (int): the flat index of the first returned value.
                    stop (int): the flat index of the last returned value.
                    vals (List[float]): the new values
        """
        latest_index = self.last_saved_index
        if latest_index is None:
            latest_index = -1
        if self.modified_range:
            latest_index = max(latest_index, self.modified_range[1])

        vals = [
            self.ndarray[np.unravel_index(i, self.ndarray.shape)]
            for i in range(synced_index + 1, latest_index + 1)
        ]

        if vals:
            return {
                'start': synced_index + 1,
                'stop': latest_index,
                'vals': vals
            }

    def apply_changes(self, start, stop, vals):
        """
        Insert new synced values into the array.

        To be be called in a ``PULL_FROM_SERVER`` ``DataSet`` using results
        returned by ``get_changes`` from the ``DataServer``.

        TODO: check that vals has the right length?

        Args:
            start (int): the flat index of the first new value.
            stop (int): the flat index of the last new value.
            vals (List[float]): the new values
        """
        for i, val in enumerate(vals):
            index = np.unravel_index(i + start, self.ndarray.shape)
            self.ndarray[index] = val
        self.synced_index = stop

    def __repr__(self):
        array_id_or_none = ' {}'.format(self.array_id) if self.array_id else ''
        return '{}[{}]:{}\n{}'.format(self.__class__.__name__,
                                      ','.join(map(str, self.shape)),
                                      array_id_or_none, repr(self.ndarray))

    def snapshot(self, update=False):
        """JSON representation of this DataArray."""
        snap = {'__class__': full_class(self)}

        snap.update(self._snapshot_input)

        for attr in self.SNAP_ATTRS:
            snap[attr] = getattr(self, attr)

        return snap

    def fraction_complete(self):
        """
        Get the fraction of this array which has data in it.

        Or more specifically, the fraction of the latest point in the array
        where we have touched it.

        Returns:
            float: fraction of array which is complete, from 0.0 to 1.0
        """
        if self.ndarray is None:
            return 0.0

        last_index = -1
        if self.last_saved_index is not None:
            last_index = max(last_index, self.last_saved_index)
        if self.modified_range is not None:
            last_index = max(last_index, self.modified_range[1])
        if getattr(self, 'synced_index', None) is not None:
            last_index = max(last_index, self.synced_index)

        return (last_index + 1) / self.ndarray.size

    @property
    def units(self):
        warn_units('DataArray', self)
        return self.unit

    # Built-in numeric commands

    ## Math operators
    def __abs__(self):
        new_array = abs(self.ndarray)
        data_array = DataArray(name=self.name, label=self.label, unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array, set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __add__(self, other):
        new_array = self.ndarray + other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new_array = self.ndarray - other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rsub__(self, other):
        new_array = other - self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __mul__(self, other):
        new_array = self.ndarray*other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        new_array = self.ndarray // other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rfloordiv__(self, other):
        new_array = other // self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __truediv__(self, other):
        new_array = self.ndarray / other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rdiv__(self, other):
        new_array = other / self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __mod__(self, other):
        new_array = self.ndarray % other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rmod__(self, other):
        return self.__mod__(other)

    def __floor__(self):
        new_array = np.floor(self.ndarray)
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __ceil__(self):
        new_array = np.ceil(self.ndarray)
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __pow__(self, power, modulo=None):
        new_array = pow(self.ndarray, power, modulo)
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rpow__(self, other):
        new_array = pow(other, self.ndarray)
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    ## Binary operations
    def __and__(self, other):
        new_array = self.ndarray & other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        new_array = self.ndarray | other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        new_array = self.ndarray ^ other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rxor__(self, other):
        return self.__xor__(other)

    def __lt__(self, other):
        new_array = self.ndarray < other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __le__(self, other):
        new_array = self.ndarray <= other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    ## Currently __eq__ and __ne__ are left unchanged to preserve hashing
    # def __eq__(self, other):
    #     return np.all(self.ndarray == other)
    #
    # def __ne__(self, other):
    #     return np.all(self.ndarray != other)

    def __ge__(self, other):
        new_array = self.ndarray >= other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __gt__(self, other):
        new_array = self.ndarray > other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __lshift__(self, other):
        new_array = self.ndarray << other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rlshift__(self, other):
        new_array = other << self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rshift__(self, other):
        new_array = self.ndarray >> other
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __rrshift__(self, other):
        new_array = other >> self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    ## Unary operations
    def __pos__(self):
        return self

    def __neg__(self):
        new_array = -self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array

    def __invert__(self):
        new_array = ~self.ndarray
        data_array = DataArray(name=self.name, label=self.label,
                               unit=self.unit,
                               is_setpoint=self.is_setpoint,
                               preset_data=new_array,
                               set_arrays=self.set_arrays)
        data_array.data_set = self.data_set
        return data_array
