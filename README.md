## Current Progress
- Fixed the previous mistake on cosine offsets by including $\phi x_i|_{t_a}^{t_b}$ into the $\bm{b}$-terms of the least squares problem; the Fourier-based weak SINDy is now doing consistently better over all noise levels
- Numerical experiments are carried out for the Lorenz system, and varying noise levels and numbers of test functions are tested
  
## Next Steps
- Try a different attractor
- Try FFT-based frequency selections based on clean trajectory data (maybe try noisy ones as well now that the mistake is fixed)
- Add error bars