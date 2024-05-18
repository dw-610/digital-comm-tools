# Python Digital Communication Tools

This modules in this repository contain tools for baseband simulation of a digital communication system. The physical-layer functionalities are performed according to the IEEE 802.11 standard [[1](https://ieeexplore.ieee.org/document/9442429), [2](https://ieeexplore.ieee.org/document/9363693)].

See class/functions definitions within the modules for documentation and usage details.

## Modules

### `wifi_phy.py`

Defines a single class `PHY` which performs physical-layer operations according to the IEEE 802.11 standard. These operations include:

- Channel coding/decoding
  - Convolutional channel coding with 1/2, 2/3, and 3/4 rate options
  - Decoding using the Viterbi algorithm, with hard or soft decoding options
- Modulation/demodulation
  - BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM, 1024-QAM
- Orthogonal frequency-division modulation (OFDM) and demodulation
  - Options for the number of subcarriers and cyclic prefix length

### `channels.py`

Defines three classes for simulating different channel effects. For the fading channels, the symbol period is assumed to be approximately equal to the channel coherence time, so that the fading coefficients are IID and differ from one symbols to the next.

- `AWGN`: simulates a basic additive white Gaussian noise channel with no fading effects.
- `Rayleigh`: simulates a channel with Rayleigh fading effects and AWGN.
- `Rician`: simulates a channel with Rician fading effects and AWGN.

### `ieee754.py`

Defines a pair of functions `ieee754_b2f` and `ieee754_f2b` for strings of 32 bits to the corresponding floating-point value defined by the IEEE 754 standard, and vice-versa.

### `quantizer.py`

Defines a single class `Quantizer` which implements a uniform quantization scheme to convert a given number of bits to a floating point value, and vice-versa. The range of the quantizer and the number of quantization bits are specified as parameters.