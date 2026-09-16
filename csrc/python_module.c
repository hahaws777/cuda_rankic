#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include "rankic.h"

static PyObject *status_error(int status, const char *error) {
    PyErr_SetString(status == RANKIC_INVALID_ARGUMENT ||
                   status == RANKIC_WORKSPACE_TOO_SMALL ? PyExc_ValueError : PyExc_RuntimeError,
                   error[0] ? error : "RankIC backend failed");
    return NULL;
}
static PyObject *workspace_size(PyObject *self, PyObject *args) {
    long long rows, cols;
    int strategy, status;
    size_t bytes = 0;
    char error[512] = {0};
    if (!PyArg_ParseTuple(args, "LLi", &rows, &cols, &strategy)) return NULL;
    Py_BEGIN_ALLOW_THREADS
    status = rankic_workspace_size(rows, cols, strategy, &bytes, error, sizeof(error));
    Py_END_ALLOW_THREADS
    if (status) return status_error(status, error);
    return PyLong_FromSize_t(bytes);
}
static PyObject *run(PyObject *self, PyObject *args) {
    unsigned long long x, y, out, workspace, bytes, stream;
    long long rows, cols;
    int strategy, status;
    char error[512] = {0};
    if (!PyArg_ParseTuple(args, "KKKLLKKiK", &x, &y, &out, &rows, &cols,
                         &workspace, &bytes, &strategy, &stream)) return NULL;
    Py_BEGIN_ALLOW_THREADS
    status = rankic_cuda_f32((const float *)(uintptr_t)x, (const float *)(uintptr_t)y,
        (float *)(uintptr_t)out, rows, cols, (void *)(uintptr_t)workspace,
        (size_t)bytes, strategy, (void *)(uintptr_t)stream, error, sizeof(error));
    Py_END_ALLOW_THREADS
    if (status) return status_error(status, error);
    Py_RETURN_NONE;
}
static PyObject *build_info(PyObject *self, PyObject *args) {
    return Py_BuildValue("{s:s,s:s,s:i}", "version", "0.1.0",
        "backend", "C ABI / CUDA CUB", "abi_version", 1);
}
static PyMethodDef methods[] = {
    {"workspace_size", workspace_size, METH_VARARGS, "Query caller-owned GPU scratch bytes."},
    {"run", run, METH_VARARGS, "Launch asynchronously using raw CUDA pointers (internal API)."},
    {"build_info", build_info, METH_NOARGS, "Native backend information."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
