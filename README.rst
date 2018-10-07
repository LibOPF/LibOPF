THIS FORK OF LIBOPF IS UNMAINTAINED. Go to https://github.com/jppbsi/LibOPF for the latest version.

LibOPF is a library of functions and programs for free usage in the
design of optimum-path forest classifiers. This second version 
contains some additional resources related to the supervised
OPF classifier reported in reference [PapaIJIST09], and also
contains the unsupervised version of OPF reported in reference
[RochaIJIST09].

## How to install

Download and uncompress the code.

Change to the LibOPF directory:
`cd LibOPF`

Create the directories 'lib' and 'obj':
`mkdir lib`
`mkdir obj`

Compile:
`make`

If you want the python bindings.
For default python version:
`make bindings`
`source env.sh`

For e.g. python3:
`make  PYTHON=python3  PYTHON_CONFIG=python3-config  bindings`
`source env.sh`



A short explanation about the method can be found in
http://www.ic.unicamp.br/~afalcao/LibOPF. Please read the COPYRIGHT
file before using LibOPF.

For more information (building instructions and examples), see:

http://adessowiki.fee.unicamp.br/adesso/wiki/toolboxOPF/MainPage/view/

For any questions and comments, please send your email to
victormatheus@gmail, papa.joaopaulo@gmail.com or afalcao@ic.unicamp.br.
