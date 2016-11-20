# CNNEncrypt

CNNEncrypt was made as part of a final project for 6.867 - Machine Learning (MIT)
It contains an example implementation of the network design described in the recently published paper:

"Learning to Protect Communications with Adversarial Neural Cryptography" (2016) by  Martín Abadi, David G. Andersen (Google Brain).

## Usage 

The required packages can be found in the requirements.txt file and can be installed using:  

`pip install -r requirements.txt`

Note that using the GPU implementation of Tensorflow increases performance dramatically.
We recommend using a GPU for training since this particular 
experiment requires thousands of training iterations over large batches.

To run the training, modify the parameters at the top of the train.py script:   

* N - size of the keys and plaintext  
* batch_size - the number of training examples to use in a single iteration
* epochs - the number of training iterations

and run:  

`python train.py`

## Liscence

Provided under the MIT License (MIT)
Copyright (c) 2016 Jeremy Wohlwend, Luis Sanmiguel

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
