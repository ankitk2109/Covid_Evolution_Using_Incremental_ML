# encoding: utf-8
# module skmultiflow.metrics._classification_performance_evaluator
# from C:\Anaconda3\lib\site-packages\skmultiflow\metrics\_classification_performance_evaluator.cp37-win_amd64.pyd
# by generator 1.147
# no doc

# imports
import builtins as __builtins__  # <module 'builtins' (built-in)>
import numpy as np  # C:\Anaconda3\lib\site-packages\numpy\__init__.py
import scipy as sp  # C:\Anaconda3\lib\site-packages\scipy\__init__.py
import warnings as warnings  # C:\Anaconda3\lib\warnings.py
from skmultiflow.metrics._confusion_matrix import (ConfusionMatrix,
                                                   MultiLabelConfusionMatrix)


# functions

def __pyx_unpickle_ClassificationPerformanceEvaluator(*args, **kwargs):  # real signature unknown
    pass


def __pyx_unpickle_MultiLabelClassificationPerformanceEvaluator(*args, **kwargs):  # real signature unknown
    pass


def __pyx_unpickle_WindowClassificationPerformanceEvaluator(*args, **kwargs):  # real signature unknown
    pass


def __pyx_unpickle_WindowMultiLabelClassificationPerformanceEvaluator(*args, **kwargs):  # real signature unknown
    pass


# classes

class ClassificationPerformanceEvaluator(object):
    """
    Classification performance evaluator.

        Track a classifier's performance and provide, at any moment, updated
        performance metrics. This performance evaluator is designed for single-output
        (binary and multi-class) classification tasks.

        Parameters
        ----------
        n_classes: int, optional (default=2)
            The number of classes.

        Notes
        -----
        Although the number of classes can be defined (default=2 for the binary case),
        if more classes are observed, then the confusion matrix is reshaped to account
        for new (emerging) classes.
    """

    def accuracy_score(self, *args, **kwargs):  # real signature unknown
        """
        Accuracy score.

                The accuracy is the ratio of correctly classified samples to the total
                number of samples.

                Returns
                -------
                float
                    Accuracy.
        """
        pass

    def add_result(self, *args, **kwargs):  # real signature unknown
        """
        Update internal statistics with the results of a prediction.

                Parameters
                ----------
                y_true: int
                    The true (actual) value.

                y_pred: int
                    The predicted value.

                sample_weight: float
                    The weight of the sample.
        """
        pass

    def f1_score(self, *args, **kwargs):  # real signature unknown
        """
        F1 score.

                The F1 score can be interpreted as a weighted average of the precision and
                recall. The relative contribution of precision and recall to the F1 score
                are equal. The F1 score is defined as:

                .. math::
                    F1 = \frac{2 \times (precision \times recall)}{(precision + recall)}

                Parameters
                ----------
                class_value: int, optional (default=-1)
                    Class value to calculate this metric for. Not used by default.

                Returns
                -------
                float
                    F1-score.

                Notes
                -----
                If seen data corresponds to a multi-class problem then calculate the ``macro``
                average, that is, calculate metrics for each class, and find their unweighted mean.
        """
        pass

    def geometric_mean_score(self, *args, **kwargs):  # real signature unknown
        """
        Geometric mean score.

                The geometric mean is a good indicator of a classifier's performance
                in the presence of class imbalance because it is independent of the
                distribution of examples between classes [1]_. This implementation
                computes the geometric mean of class-wise sensitivity (recall)

                .. math::
                    gm = \sqrt[n]{s_1\cdot s_2\cdot s_3\cdot \ldots\cdot s_n}

                where :math:`s_i` is the sensitivity (recall) of class :math:`i` and : math: `n`
                is the number of classes.

                Returns
                -------
                float
                    Geometric mean score.

                References
                ----------
                .. [1] Barandela, R. et al. “Strategies for learning in class imbalance problems”,
                       Pattern Recognition, 36(3), (2003), pp 849-851.
        """
        pass

    def get_info(self, *args, **kwargs):  # real signature unknown
        """ Get (current) information about the performance evaluator. """
        pass

    def get_last(self, *args, **kwargs):  # real signature unknown
        """
        Last samples (y_true, y_pred) observed.

                Returns
                -------
                tuple
                    (last_true, last_pred) tuple
        """
        pass

    def kappa_m_score(self, *args, **kwargs):  # real signature unknown
        """
        Kappa-M score.

                The Kappa-M statistic [1]_ compares performance with the majority class classifier.
                It is defined as

                .. math::
                    \kappa_{m} = (p_o - p_e) / (1 - p_e)

                where :math:`p_o` is the empirical probability of agreement on the label
                assigned to any sample (prequential accuracy), and :math:`p_e` is
                the prequential accuracy of the ``majority classifier``.

                Returns
                -------
                float
                    Kappa-M.

                References
                ----------
                .. [1] A. Bifet et al. "Efficient online evaluation of big data stream classifiers."
                       In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery
                       and data mining, pp. 59-68. ACM, 2015.
        """
        pass

    def kappa_score(self, *args, **kwargs):  # real signature unknown
        """
        Kappa score.

                Cohen's Kappa [1]_ expresses the level of agreement between two annotators
                 on a classification problem. It is defined as

                .. math::
                    \kappa = (p_o - p_e) / (1 - p_e)

                where :math:`p_o` is the empirical probability of agreement on the label
                assigned to any sample (prequential accuracy), and :math:`p_e` is
                the expected agreement when both annotators assign labels randomly.

                Returns
                -------
                float
                    Cohen's Kappa.

                References
                ----------
                .. [1] J. Cohen (1960). "A coefficient of agreement for nominal scales".
                       Educational and Psychological Measurement 20(1):37-46.
                       doi:10.1177/001316446002000104.
        """
        pass

    def kappa_t_score(self, *args, **kwargs):  # real signature unknown
        """
        Kappa-T score.

                The Kappa Temp [1]_ measures the temporal correlation between samples.
                It is defined as

                .. math::
                    \kappa_{t} = (p_o - p_e) / (1 - p_e)

                where :math:`p_o` is the empirical probability of agreement on the label
                assigned to any sample (prequential accuracy), and :math:`p_e` is
                the prequential accuracy of the ``no-change classifier`` that predicts
                only using the last class seen by the classifier.

                Returns
                -------
                float
                    Kappa-T.

                References
                ----------
                .. [1] A. Bifet et al. (2013). "Pitfalls in benchmarking data stream classification
                       and how to avoid them." Proc. of the European Conference on Machine Learning
                       and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD'13),
                       Springer LNAI 8188, p. 465-479.
        """
        pass

    def majority_class(self, *args, **kwargs):  # real signature unknown
        """
        Compute the majority class.

                Returns
                -------
                int
                    The majority class.
        """
        pass

    def precision_score(self, *args, **kwargs):  # real signature unknown
        """
        Precision score.

                The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
                true positives and ``fp`` the number of false positives.

                Parameters
                ----------
                class_value: int, optional (default=-1)
                    Class value to calculate this metric for. Not used by default.

                Returns
                -------
                float
                    Precision.

                Notes
                -----
                If seen data corresponds to a multi-class problem then calculate the ``macro``
                average, that is, calculate metrics for each class, and find their unweighted mean.
        """
        pass

    def recall_score(self, *args, **kwargs):  # real signature unknown
        """
        Recall score.

                The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
                true positives and ``fn`` the number of false negatives.

                Parameters
                ----------
                class_value: int, optional (default=-1)
                    Class value to calculate this metric for. Not used by default.

                Returns
                -------
                float
                    Recall.

                Notes
                -----
                If seen data corresponds to a multi-class problem then calculate the ``macro``
                average, that is, calculate metrics for each class, and find their unweighted mean.
        """
        pass

    def reset(self, *args, **kwargs):  # real signature unknown
        """ Reset the evaluator to its initial state. """
        pass

    def _get_info_metrics(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    confusion_matrix = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    last_pred = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    last_true = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    n_samples = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    total_weight_observed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ The total weight observed.
        """

    weight_correct_no_change_classifier = property(lambda self: object(), lambda self, v: None,
                                                   lambda self: None)  # default

    weight_majority_classifier = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    _init_n_classes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    _total_weight_observed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002203AEF9540>'


class deque(object):
    """
    deque([iterable[, maxlen]]) --> deque object

    A list-like sequence optimized for data accesses near its endpoints.
    """

    def append(self, *args, **kwargs):  # real signature unknown
        """ Add an element to the right side of the deque. """
        pass

    def appendleft(self, *args, **kwargs):  # real signature unknown
        """ Add an element to the left side of the deque. """
        pass

    def clear(self, *args, **kwargs):  # real signature unknown
        """ Remove all elements from the deque. """
        pass

    def copy(self, *args, **kwargs):  # real signature unknown
        """ Return a shallow copy of a deque. """
        pass

    def count(self, value):  # real signature unknown; restored from __doc__
        """ D.count(value) -> integer -- return number of occurrences of value """
        return 0

    def extend(self, *args, **kwargs):  # real signature unknown
        """ Extend the right side of the deque with elements from the iterable """
        pass

    def extendleft(self, *args, **kwargs):  # real signature unknown
        """ Extend the left side of the deque with elements from the iterable """
        pass

    def index(self, value, start=None, stop=None):  # real signature unknown; restored from __doc__
        """
        D.index(value, [start, [stop]]) -> integer -- return first index of value.
        Raises ValueError if the value is not present.
        """
        return 0

    def insert(self, index, p_object):  # real signature unknown; restored from __doc__
        """ D.insert(index, object) -- insert object before index """
        pass

    def pop(self, *args, **kwargs):  # real signature unknown
        """ Remove and return the rightmost element. """
        pass

    def popleft(self, *args, **kwargs):  # real signature unknown
        """ Remove and return the leftmost element. """
        pass

    def remove(self, value):  # real signature unknown; restored from __doc__
        """ D.remove(value) -- remove first occurrence of value. """
        pass

    def reverse(self):  # real signature unknown; restored from __doc__
        """ D.reverse() -- reverse *IN PLACE* """
        pass

    def rotate(self, *args, **kwargs):  # real signature unknown
        """ Rotate the deque n steps to the right (default n=1).  If n is negative, rotates left. """
        pass

    def __add__(self, *args, **kwargs):  # real signature unknown
        """ Return self+value. """
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        pass

    def __contains__(self, *args, **kwargs):  # real signature unknown
        """ Return key in self. """
        pass

    def __copy__(self, *args, **kwargs):  # real signature unknown
        """ Return a shallow copy of a deque. """
        pass

    def __delitem__(self, *args, **kwargs):  # real signature unknown
        """ Delete self[key]. """
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        pass

    def __getattribute__(self, *args, **kwargs):  # real signature unknown
        """ Return getattr(self, name). """
        pass

    def __getitem__(self, *args, **kwargs):  # real signature unknown
        """ Return self[key]. """
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        pass

    def __iadd__(self, *args, **kwargs):  # real signature unknown
        """ Implement self+=value. """
        pass

    def __imul__(self, *args, **kwargs):  # real signature unknown
        """ Implement self*=value. """
        pass

    def __init__(self, iterable=None, maxlen=None):  # real signature unknown; restored from __doc__
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        """ Implement iter(self). """
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        """ Return len(self). """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mul__(self, *args, **kwargs):  # real signature unknown
        """ Return self*value. """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        """ Return state information for pickling. """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __reversed__(self):  # real signature unknown; restored from __doc__
        """ D.__reversed__() -- return a reverse iterator over the deque """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __setitem__(self, *args, **kwargs):  # real signature unknown
        """ Set self[key] to value. """
        pass

    def __sizeof__(self):  # real signature unknown; restored from __doc__
        """ D.__sizeof__() -- size of D in memory, in bytes """
        pass

    maxlen = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """maximum size of a deque or None if unbounded"""

    __hash__ = None


class DTYPE(object):
    """ Convert a string or number to a floating point number, if possible. """

    def as_integer_ratio(self):  # real signature unknown; restored from __doc__
        """
        Return integer ratio.

        Return a pair of integers, whose ratio is exactly equal to the original float
        and with a positive denominator.

        Raise OverflowError on infinities and a ValueError on NaNs.

        >>> (10.0).as_integer_ratio()
        (10, 1)
        >>> (0.0).as_integer_ratio()
        (0, 1)
        >>> (-.25).as_integer_ratio()
        (-1, 4)
        """
        pass

    def conjugate(self, *args, **kwargs):  # real signature unknown
        """ Return self, the complex conjugate of any float. """
        pass

    @classmethod
    def fromhex(cls, *args, **kwargs):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        Create a floating-point number from a hexadecimal string.

        >>> float.fromhex('0x1.ffffp10')
        2047.984375
        >>> float.fromhex('-0x1p-1074')
        -5e-324
        """
        pass

    def hex(self):  # real signature unknown; restored from __doc__
        """
        Return a hexadecimal representation of a floating-point number.

        >>> (-0.1).hex()
        '-0x1.999999999999ap-4'
        >>> 3.14159.hex()
        '0x1.921f9f01b866ep+1'
        """
        pass

    def is_integer(self, *args, **kwargs):  # real signature unknown
        """ Return True if the float is an integer. """
        pass

    def __abs__(self, *args, **kwargs):  # real signature unknown
        """ abs(self) """
        pass

    def __add__(self, *args, **kwargs):  # real signature unknown
        """ Return self+value. """
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        pass

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(self, value). """
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        pass

    def __float__(self, *args, **kwargs):  # real signature unknown
        """ float(self) """
        pass

    def __floordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return self//value. """
        pass

    def __format__(self, *args, **kwargs):  # real signature unknown
        """ Formats the float according to format_spec. """
        pass

    def __getattribute__(self, *args, **kwargs):  # real signature unknown
        """ Return getattr(self, name). """
        pass

    @classmethod
    def __getformat__(cls, *args, **kwargs):  # real signature unknown
        """
        You probably don't want to use this function.

          typestr
            Must be 'double' or 'float'.

        It exists mainly to be used in Python's test suite.

        This function returns whichever of 'unknown', 'IEEE, big-endian' or 'IEEE,
        little-endian' best describes the format of floating point numbers used by the
        C type named by typestr.
        """
        pass

    def __getnewargs__(self, *args, **kwargs):  # real signature unknown
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        pass

    def __hash__(self, *args, **kwargs):  # real signature unknown
        """ Return hash(self). """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        pass

    def __mul__(self, *args, **kwargs):  # real signature unknown
        """ Return self*value. """
        pass

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        pass

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        pass

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __round__(self, *args, **kwargs):  # real signature unknown
        """
        Return the Integral closest to x, rounding half toward even.

        When an argument is passed, work like built-in round(x, ndigits).
        """
        pass

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        pass

    @classmethod
    def __set_format__(cls, *args, **kwargs):  # real signature unknown
        """
        You probably don't want to use this function.

          typestr
            Must be 'double' or 'float'.
          fmt
            Must be one of 'unknown', 'IEEE, big-endian' or 'IEEE, little-endian',
            and in addition can only be one of the latter two if it appears to
            match the underlying C reality.

        It exists mainly to be used in Python's test suite.

        Override the automatic determination of C-level floating point type.
        This affects how floats are converted to and from binary strings.
        """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    def __sub__(self, *args, **kwargs):  # real signature unknown
        """ Return self-value. """
        pass

    def __truediv__(self, *args, **kwargs):  # real signature unknown
        """ Return self/value. """
        pass

    def __trunc__(self, *args, **kwargs):  # real signature unknown
        """ Return the Integral closest to x between 0 and x. """
        pass

    imag = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """the imaginary part of a complex number"""

    real = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """the real part of a complex number"""


class MultiLabelClassificationPerformanceEvaluator(object):
    """
    Multi-label classification performance evaluator.

        Track a classifier's performance and provide, at any moment, updated
        performance metrics. This performance evaluator is designed for multi-output
        (multi-label) classification tasks.

        Parameters
        ----------
        n_labels: int, optional (default=2)
            The number of labels.

        Notes
        -----
        Although the number of labels can be defined (default=2), if more labels are observed,
        then the confusion matrix is reshaped to account for new (emerging) labels.
    """

    def add_result(self, *args, **kwargs):  # real signature unknown
        """
        Update internal statistics with the results of a prediction.

                Parameters
                ----------
                y_true: np.ndarray of shape (n_labels,)
                    A 1D array with binary indicators for true (actual) values.

                y_pred: np.ndarray of shape (n_labels,)
                    A 1D array with binary indicators for predicted values.

                sample_weight: float
                    The weight of the sample.
        """
        pass

    def exact_match_score(self, *args, **kwargs):  # real signature unknown
        """
        Exact match score.

                This is the most strict multi-label metric, defined as the number of
                samples that have all their labels correctly classified, divided by the
                total number of samples.

                Returns
                -------
                float
                    Exact match score.
        """
        pass

    def get_info(self, *args, **kwargs):  # real signature unknown
        """ Get (current) information about the performance evaluator. """
        pass

    def get_last(self, *args, **kwargs):  # real signature unknown
        """
        Last samples (y_true, y_pred) observed.

                Returns
                -------
                tuple
                    (last_true, last_pred) tuple
        """
        pass

    def hamming_loss_score(self, *args, **kwargs):  # real signature unknown
        """
        Hamming loss score.

                The Hamming loss is the complement of the Hamming score.

                Returns
                -------
                float
                    Hamming loss score.
        """
        pass

    def hamming_score(self, *args, **kwargs):  # real signature unknown
        """
        Hamming score.

                The Hamming score is the fraction of labels that are correctly predicted.

                Returns
                -------
                float
                    Hamming score.
        """
        pass

    def jaccard_score(self, *args, **kwargs):  # real signature unknown
        """
        Jaccard similarity coefficient score.

                The Jaccard index, or Jaccard similarity coefficient, defined as
                the size of the intersection divided by the size of the union of two label
                sets, is used to compare the set of predicted labels for a sample with the
                corresponding set of labels in ``y_true``.

                Returns
                -------
                float
                    Jaccard score.

                Notes
                -----
                The Jaccard index may be a poor metric if there are no positives for some samples or labels.
                The Jaccard index is undefined if there are no true or predicted labels, this implementation
                will return a score of 0 if this is the case.
        """
        pass

    def reset(self, *args, **kwargs):  # real signature unknown
        """ Reset the evaluator to its initial state. """
        pass

    def _get_info_metrics(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    confusion_matrix = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    exact_match_cnt = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    jaccard_sum = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    last_pred = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    last_true = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    n_samples = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    _init_n_labels = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002203AEF9870>'


class WindowClassificationPerformanceEvaluator(ClassificationPerformanceEvaluator):
    """
    Window classification performance evaluator.

        Track a classifier's performance over a sliding window and provide, at any moment,
        updated performance metrics. This performance evaluator is designed for single-output
        (binary and multi-class) classification tasks.

        Parameters
        ----------
        n_classes: int, optional (default=2)
            The number of classes.

        window_size: int, optional (default=200)
            The size of the window.

        Notes
        -----
        Although the number of classes can be defined (default=2 for the binary case),
        if more classes are observed, then the confusion matrix is reshaped to account
        for new (emerging) classes.
    """

    def add_result(self, *args, **kwargs):  # real signature unknown
        """
        Update internal statistics with the results of a prediction.

                Parameters
                ----------
                y_true: int
                    The true (actual) value.

                y_pred: int
                    The predicted value.

                sample_weight: float
                    The weight of the sample.

                Notes
                -----
                Oldest samples are automatically removed when the window is full. Special care
                is taken to keep internal statistics consistent with the samples in the window.
        """
        pass

    def get_info(self, *args, **kwargs):  # real signature unknown
        """ Get (current) information about the performance evaluator. """
        pass

    def reset(self, *args, **kwargs):  # real signature unknown
        """ Reset the evaluator to its initial state. """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    _queue = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002203AEF94E0>'


class WindowMultiLabelClassificationPerformanceEvaluator(MultiLabelClassificationPerformanceEvaluator):
    """
    Window multi-label classification performance evaluator.

        Track a classifier's performance over a sliding window and provide, at any moment,
        updated performance metrics. This performance evaluator is designed for multi-output
        (multi-label) classification tasks.

        Parameters
        ----------
        n_labels: int, optional (default=2)
            The number of labels.

        window_size: int, optional (default=200)
            The size of the window.

        Notes
        -----
        Although the number of labels can be defined (default=2), if more labels are observed,
        then the confusion matrix is reshaped to account for new (emerging) labels.
    """

    def add_result(self, *args, **kwargs):  # real signature unknown
        """
        Update internal statistics with the results of a prediction.

                Parameters
                ----------
                y_true: np.ndarray of shape (n_labels,)
                    A 1D array with binary indicators for true (actual) values.

                y_pred: np.ndarray of shape (n_labels,)
                    A 1D array with binary indicators for predicted values.

                sample_weight: float
                    The weight of the sample.

                Notes
                -----
                Oldest samples are automatically removed when the window is full. Special care
                is taken to keep internal statistics consistent with the samples in the window.
        """
        pass

    def get_info(self, *args, **kwargs):  # real signature unknown
        """ Get (current) information about the performance evaluator. """
        pass

    def reset(self, *args, **kwargs):  # real signature unknown
        """ Reset the evaluator to its initial state. """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    _queue = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x000002203AEF9900>'


# variables with complex values

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x000002203AF08788>'

__pyx_capi__ = {
    '_check_multi_label_inputs': None,
    # (!) real value is '<capsule object "PyBoolObject *(PyArrayObject *, PyArrayObject *)" at 0x000002203AEF9420>'
}

__spec__ = None  # (!) real value is "ModuleSpec(name='skmultiflow.metrics._classification_performance_evaluator', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x000002203AF08788>, origin='C:\\\\Anaconda3\\\\lib\\\\site-packages\\\\skmultiflow\\\\metrics\\\\_classification_performance_evaluator.cp37-win_amd64.pyd')"

__test__ = {}

