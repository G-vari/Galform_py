from scipy.interpolate import interp1d

def Linear_Interpolation(x1_in, y1_in, x2_in):
    
    f = interp1d(x1_in, y1_in, kind="linear",axis=0)

    y2_out = f(x2_in)

    return y2_out
