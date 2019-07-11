# Installing Coral Edge TPU libraries on a Raspberry PI 4

1. Install the opencv libraries (not needed for coral edge, just for running examples in this repo!)
`sudo apt install python3-opencv`

2. Follow the installation steps of the coral edge TPU

Before running the installation script, go to step 3

3. Modify the installation script to enable the recognition of this model

Add the following code to the line #60 in install.sh script **Before the end of the IF (fi)**
~~~~
  elif [[ "${MODEL}" == "Raspberry Pi 4 Model B Rev 1.1"* ]]; then
    warn "Recognized as Raspberry Pi 4 B Rev 1.1."
    LIBEDGETPU_SUFFIX=arm32
    HOST_GNU_TYPE=arm-linux-gnueabihf 
~~~~

4. Run the install script. It may need sudo permission

5. Link the library.(I don't know whether this was due to point 3 or is it just a bug in the installation)
~~~~
cd /usr/local/lib/python3.7/dist-packages/edgetpu/swig/
sudo ln -s _edgetpu_cpp_wrapper.cpython-35m-arm-linux-gnueabihf.so _edgetpu_cpp_wrapper.so
~~~~
