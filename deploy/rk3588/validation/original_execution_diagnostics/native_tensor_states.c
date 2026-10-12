/* Private CPython metadata-loop prototype. No Torch/libtorch imports or ABI. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Read the original function's live globals, including builtin fallback. */
static PyObject *load_global(PyObject *globals, const char *name) {
    PyObject *value = PyDict_GetItemString(globals, name);
    if (value != NULL) return Py_NewRef(value);
    PyObject *builtins = PyDict_GetItemString(globals, "__builtins__");
    if (builtins != NULL && PyModule_Check(builtins)) builtins = PyModule_GetDict(builtins);
    if (builtins != NULL && PyDict_Check(builtins)) value = PyDict_GetItemString(builtins, name);
    if (value != NULL) return Py_NewRef(value);
    PyErr_Format(PyExc_NameError, "name '%s' is not defined", name);
    return NULL;
}

/* Use Python __ne__/truth semantics; identity shortcuts would change them. */
static int different(PyObject *left, PyObject *right) {
    PyObject *value = PyObject_RichCompare(left, right, Py_NE);
    if (value == NULL) return -1;
    int answer = PyObject_IsTrue(value);
    Py_DECREF(value);
    return answer;
}

static PyObject *call_zero(PyObject *object, const char *name) {
    PyObject *method = PyObject_GetAttrString(object, name);
    if (method == NULL) return NULL;
    PyObject *value = PyObject_CallNoArgs(method);
    Py_DECREF(method);
    return value;
}

/* LOAD_GLOBAL tuple precedes the argument's attribute/method evaluation. */
static PyObject *tuple_attribute(PyObject *globals, PyObject *tensor,
                                 const char *name, int call) {
    PyObject *function = load_global(globals, "tuple");
    if (function == NULL) return NULL;
    PyObject *value = call ? call_zero(tensor, name) : PyObject_GetAttrString(tensor, name);
    PyObject *result = value == NULL ? NULL : PyObject_CallOneArg(function, value);
    Py_XDECREF(value);
    Py_DECREF(function);
    return result;
}

/* Evaluate exception class before f-string formatting, as in the original. */
static PyObject *stale_exception(PyObject *globals, PyObject *name, const char *suffix) {
    PyObject *kind = load_global(globals, "PatchCacheStaleState");
    if (kind == NULL) return NULL;
    PyObject *formatted = PyObject_Format(name, NULL);
    PyObject *message = formatted == NULL ? NULL : PyUnicode_FromFormat("%U%s", formatted, suffix);
    PyObject *exception = message == NULL ? NULL : PyObject_CallOneArg(kind, message);
    Py_XDECREF(message);
    Py_XDECREF(formatted);
    Py_DECREF(kind);
    return exception;
}

static int raise_fixed_policy(PyObject *globals, PyObject *name) {
    PyObject *exception = stale_exception(globals, name,
        ": fixed model must retain CPU FP32 parameters/floating buffers");
    if (exception == NULL) return -1;
    if (!PyExceptionInstance_Check(exception)) {
        Py_DECREF(exception);
        PyErr_SetString(PyExc_TypeError, "exceptions must derive from BaseException");
        return -1;
    }
    PyErr_SetObject((PyObject *)Py_TYPE(exception), exception);
    Py_DECREF(exception);
    return -1;
}

/* Only RuntimeError from the _version read is wrapped; preserve explicit cause. */
static void wrap_version_error(PyObject *globals, PyObject *name) {
    PyObject *kind = NULL, *value = NULL, *traceback = NULL;
    PyErr_Fetch(&kind, &value, &traceback);
    PyErr_NormalizeException(&kind, &value, &traceback);
    PyObject *runtime_error = load_global(globals, "RuntimeError");
    if (runtime_error == NULL) {
        Py_XDECREF(kind); Py_XDECREF(value); Py_XDECREF(traceback);
        return;
    }
    int caught = PyErr_GivenExceptionMatches(kind, runtime_error);
    Py_DECREF(runtime_error);
    if (!caught) {
        PyErr_Restore(kind, value, traceback);
        return;
    }
    if (traceback != NULL && value != NULL) PyException_SetTraceback(value, traceback);
    PyObject *exception = stale_exception(globals, name, ": cannot verify tensor version");
    if (exception != NULL && PyExceptionInstance_Check(exception)) {
        PyException_SetContext(exception, Py_NewRef(value));
        PyException_SetCause(exception, Py_NewRef(value));
        PyErr_SetObject((PyObject *)Py_TYPE(exception), exception);
    } else if (exception != NULL) {
        PyErr_SetString(PyExc_TypeError, "exceptions must derive from BaseException");
    }
    Py_XDECREF(exception);
    Py_XDECREF(kind); Py_XDECREF(value); Py_XDECREF(traceback);
}

/* Attribute reads deliberately follow patch_cache._tensor_state exactly. */
static PyObject *one_state(PyObject *name, PyObject *tensor, int parameter,
                           PyObject *globals) {
    PyObject *device = PyObject_GetAttrString(tensor, "device");
    if (device == NULL) return NULL;
    PyObject *device_type = PyObject_GetAttrString(device, "type");
    Py_DECREF(device);
    if (device_type == NULL) return NULL;
    PyObject *cpu = PyUnicode_FromString("cpu");
    int bad_device = cpu == NULL ? -1 : different(device_type, cpu);
    Py_XDECREF(cpu); Py_DECREF(device_type);
    if (bad_device < 0) return NULL;
    if (bad_device) { raise_fixed_policy(globals, name); return NULL; }

    int floating = parameter;
    if (!parameter) {
        PyObject *value = call_zero(tensor, "is_floating_point");
        if (value == NULL) return NULL;
        floating = PyObject_IsTrue(value);
        Py_DECREF(value);
        if (floating < 0) return NULL;
    }
    if (floating) {
        PyObject *dtype = PyObject_GetAttrString(tensor, "dtype");
        if (dtype == NULL) return NULL;
        PyObject *torch = load_global(globals, "torch");
        PyObject *float32 = torch == NULL ? NULL : PyObject_GetAttrString(torch, "float32");
        Py_XDECREF(torch);
        int bad_dtype = float32 == NULL ? -1 : different(dtype, float32);
        Py_DECREF(dtype); Py_XDECREF(float32);
        if (bad_dtype < 0) return NULL;
        if (bad_dtype) { raise_fixed_policy(globals, name); return NULL; }
    }

    PyObject *version = PyObject_GetAttrString(tensor, "_version");
    if (version == NULL) { wrap_version_error(globals, name); return NULL; }
    PyObject *row = PyTuple_New(9);
    if (row == NULL) { Py_DECREF(version); return NULL; }
    PyTuple_SET_ITEM(row, 0, Py_NewRef(name));
    PyObject *function = load_global(globals, "id");
    PyObject *identity = function == NULL ? NULL : PyObject_CallOneArg(function, tensor);
    Py_XDECREF(function);
    if (identity == NULL) { Py_DECREF(version); Py_DECREF(row); return NULL; }
    PyTuple_SET_ITEM(row, 1, identity);
    PyTuple_SET_ITEM(row, 2, version);
    for (int i = 3; i < 9; ++i) {
        PyObject *value = NULL;
        if (i == 3) value = PyObject_GetAttrString(tensor, "dtype");
        else if (i == 4) value = PyObject_GetAttrString(tensor, "device");
        else if (i == 5) value = tuple_attribute(globals, tensor, "shape", 0);
        else if (i == 6) value = tuple_attribute(globals, tensor, "stride", 1);
        else if (i == 7) value = call_zero(tensor, "data_ptr");
        else value = PyObject_GetAttrString(tensor, "requires_grad");
        if (value == NULL) { Py_DECREF(row); return NULL; }
        PyTuple_SET_ITEM(row, i, value);
    }
    return row;
}

/* Python iterable unpack order, including generic pair iterators. */
static int unpack_pair(PyObject *item, PyObject **name, PyObject **tensor) {
    if (PyTuple_CheckExact(item) && PyTuple_GET_SIZE(item) == 2) {
        *name = Py_NewRef(PyTuple_GET_ITEM(item, 0));
        *tensor = Py_NewRef(PyTuple_GET_ITEM(item, 1));
        return 0;
    }
    if (PyList_CheckExact(item) && PyList_GET_SIZE(item) == 2) {
        *name = Py_NewRef(PyList_GET_ITEM(item, 0));
        *tensor = Py_NewRef(PyList_GET_ITEM(item, 1));
        return 0;
    }
    PyObject *iterator = PyObject_GetIter(item);
    if (iterator == NULL) return -1;
    PyObject *values[3] = {NULL, NULL, NULL};
    int count = 0;
    for (; count < 3; ++count) {
        values[count] = PyIter_Next(iterator);
        if (values[count] == NULL) break;
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) {
        for (int i = 0; i < 3; ++i) Py_XDECREF(values[i]);
        return -1;
    }
    if (count != 2) {
        for (int i = 0; i < 3; ++i) Py_XDECREF(values[i]);
        if (count == 3) PyErr_SetString(PyExc_ValueError, "too many values to unpack (expected 2)");
        else PyErr_Format(PyExc_ValueError, "not enough values to unpack (expected 2, got %d)", count);
        return -1;
    }
    *name = values[0]; *tensor = values[1];
    return 0;
}

static PyObject *tensor_states(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *iterable, *flag, *globals;
    if (!PyArg_ParseTuple(args, "OOO:tensor_states", &iterable, &flag, &globals)) return NULL;
    if (!PyBool_Check(flag) || !PyDict_CheckExact(globals)) {
        PyErr_SetString(PyExc_TypeError, "Exact bool parameter flag and original globals dict required");
        return NULL;
    }
    PyObject *iterator = PyObject_GetIter(iterable);
    if (iterator == NULL) return NULL;
    PyObject *rows = PyList_New(0), *item;
    if (rows == NULL) { Py_DECREF(iterator); return NULL; }
    while ((item = PyIter_Next(iterator)) != NULL) {
        PyObject *name = NULL, *tensor = NULL;
        int code = unpack_pair(item, &name, &tensor);
        Py_DECREF(item);
        if (code < 0) { Py_DECREF(rows); Py_DECREF(iterator); return NULL; }
        PyObject *row = one_state(name, tensor, flag == Py_True, globals);
        Py_DECREF(name); Py_DECREF(tensor);
        if (row == NULL || PyList_Append(rows, row) < 0) {
            Py_XDECREF(row); Py_DECREF(rows); Py_DECREF(iterator); return NULL;
        }
        Py_DECREF(row);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) { Py_DECREF(rows); return NULL; }
    PyObject *result = PyList_AsTuple(rows);
    Py_DECREF(rows);
    return result;
}

static PyMethodDef methods[] = {
    {"tensor_states", tensor_states, METH_VARARGS,
     "Build original cache tensor states from one fresh named iterator; GIL retained."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_v76_tensor_states", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__v76_tensor_states(void) { return PyModule_Create(&module); }
