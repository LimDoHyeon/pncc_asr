# pncc_asr
An implementation of Power-Normalized Cepstral Coefficients(PNCC) (supports GPU computation)

This code is based on **"Power-Normalized Cepstral Coefﬁcients (PNCC) for Robust Speech Recognition(Chanwoo Kim, 2016)"**.
< paper link: https://ieeexplore.ieee.org/document/7439789 >


## Usage
- This code is based on torchaudio; the audio data have to be loaded by torchaudio.
- All parameters follow the origianl paper.
- 
First, Clone this repository and install requirements.
```bash
cd ~
git clone https://github.com/LimDoHyeon/pncc_asr.git
cd pncc_asr
pip install -r requirements.txt
```

And use this function:
```python
def pncc(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
```

