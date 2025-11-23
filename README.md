# Deep Neuronal Filter (DNF) -- libtorch version

This is work in progress.

## Prerequisites Libraries and packages

1) Install the IIR and FIR filter libraries

Installation instructions are in these repositories:

 - IIR: https://github.com/berndporr/iir1
 - FIR: https://github.com/berndporr/fir1

2) Make sure you have `cmake` installed.

3) Libtorch

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
