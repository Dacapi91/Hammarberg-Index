# Hammarberg-Index

This repository contains the implementation of the Hammarberg Index [1]. This index value is applied in speech signals and is related to the vocal effort reflected in the
spectral information. It is a spectral slope measurements and can be used as a feature in emotionally expressive speech.
The Hammarberg Index is defined as the intensity difference [in dB] between the maximum intensity in a
lower frequency band [0–2000 Hz] versus a higher frequency band [2000–5000 Hz] [2]

![Hammarbeg-Index](https://user-images.githubusercontent.com/55755680/135444548-2c458fa8-5e0a-4900-b33e-dbd1daa1c6a8.png)

**Implementation details**

    The signal must be more than 5 seconds of duration. This is a temporal suggestion. Based on the
    data that the experiments conducted in [1] used speech signals of duration around 40
    seconds. 

    The speech signal should not contain relevant noise components. It is assumed a clean
    speech signal or a preprocessing stage that attenuates the noise before the computation
    of the Hammarberg Index. Otherwise, high power noise frequency components will mask
    the relevant frequencies involved in the computation of the index. This might be a
    limitation in a high noise environment. 

    The speech signal must be sampled at least 12KHz. This is implied due to the fact that
    there is a need to compute properly a maximum of 5KHz component

    The signal must be .wav format according to the code implemented.


**References**

    [1] HAMMARBERG, Britta, et al. Perceptual and acoustic correlates of abnormal voice
        qualities. Acta oto-laryngologica, 1980, vol. 90, no 1-6, p. 441-451.

    [2] SCHMIDT, Juliane; JANSE, Esther; SCHARENBORG, Odette. Perception of emotion in
        conversational speech by younger and older listeners. Frontiers in psychology, 2016, vol. 7,
        p. 781.
    

**MIT License**

Copyright (c) 2021 David Castro Piñol

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
