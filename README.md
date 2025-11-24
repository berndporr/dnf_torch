# Deep Neuronal Filter (DNF) -- libtorch version

## Prerequisites Libraries and packages

1) Make sure you have `cmake` and a c++ compiler installed.

2) Libtorch

You can get libtorch from the [PyTorch homepage](https://pytorch.org/get-started/locally/).

Add `CMAKE_PREFIX_PATH=/path/to/libtorch` pointing to the libtorch directory 
as an environment variable.

## How to compile

Type:

```
cmake .
```
to create the makefile and then

```
make
```
to compile the library and the demos.

## Installation

```
sudo make install
```

## Example

[Simple instructional example which removes 50Hz from an ECG](ecg_filt_demo).

## Documentation

[Doxygen generated documentation](https://berndporr.github.io/dnf_torch/)

## Credits

 - Bernd Porr
 - Sama Daryanavard
