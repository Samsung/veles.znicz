
There are two ways to get Znicz:

1. Clone Veles and run ./init script, which updates all Veles submodules (including Znicz)::

    sudo apt-get install git
    git clone https://github.com/Samsung/veles.git
    mv veles Veles
    cd Veles
    ./init

2. Clone Znicz inside of veles folder into Veles project::

    sudo apt-get install git
    git clone https://github.com/Samsung/veles.git
    mv veles Veles
    cd Veles
    cd veles
    git clone https://github.com/Samsung/veles.znicz.git

