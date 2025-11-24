# Deep Neuronal Filter (DNF) -- libtorch version

## Prerequisites Libraries and packages

1) Install the IIR and FIR filter libraries

Installation instructions are in these repositories:

 - IIR: https://github.com/berndporr/iir1
 - FIR: https://github.com/berndporr/fir1

2) Make sure you have `cmake` installed.

3) Libtorch

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
