Goals:
    -come up with formula to compute g from image directly
    -given 2 kernels (1 for image to feature map f, 1 for f to g) 
        -something about matrix of dot products of the matrices
    -answer: why is relu necessary? Can a useful CNN be made up of only convolutions (no rectification)?
        -relu increases the nonlinearity, is a lot faster, computationally inexpensive
    -setup python venv w/ numpy, imageio; design kernels to detect something interesting, convolve some 
    -images with those kernels (maybe numpy function for inner loop, dot & sum function), normalize brightness,
    -try adding bias term, then using rectification, either save images or display using imshow in matplotlib

 What was Acheived:
    -Got venv working, though have to use anaconda prompt (can't figure out otherwise)
    -Got packages installed
    -Created random kernel
    -Used imageio image, np arrays, and matplotlib imshow function
    -Copied over apply function, wrote rectify function
    -Answered questions (basic)
    -Able to display image

Needs to be done:
    -Learn how to get kernels to detect something, convolve them
    -Modify apply function by adding numpy functions where needed
    -Add bias, use rectification, then display

Questions:
    -How do I get kernels to detect something?
    -Where do I use each numpy function in the apply function?
    -How do I add bias?
