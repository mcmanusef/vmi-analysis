import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


def get_calibration(t_peaks=(756600,758600,759325,760600),mq_peaks=(18,28,32,40),return_params=False):

    t_peaks=np.asarray(t_peaks)
    mq_peaks=np.asarray(mq_peaks)
    def inv_calibration(mq,a,b):
        return a*np.sqrt(mq)+b

    popt, *_ = scipy.optimize.curve_fit(inv_calibration,mq_peaks,t_peaks)

    def calibration(t,a=popt[0],b=popt[1]):
        return ((t-b)/a)**2
    if return_params:
        return calibration, popt
    return calibration

if __name__=="__main__":
    t_peaks = [756600,758600,759325,760600]
    mq_peaks = [18,28,32,40]

    cal,params=get_calibration(t_peaks,mq_peaks,return_params=True)
    print(params)

    plt.figure(1)
    plt.scatter(t_peaks,mq_peaks)
    plt.plot(np.linspace(750000,765000),cal(np.linspace(750000,765000)))