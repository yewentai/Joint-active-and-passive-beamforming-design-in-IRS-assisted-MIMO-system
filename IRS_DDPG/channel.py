import numpy as np
import cmath
from abc import ABC, abstractmethod
from random import gauss


def dB_to_linear(dB: float) -> float:
    """Convert decibel values to linear scale.

    Args:
        dB: The value in decibels.

    Returns:
        The value in a linear scale.
    """
    return pow(10, dB / 10)


def PL_dB(
    distance_m=50,
    path_loss_exponent=2.0,
    sigma_dB=3,
    PL_d0_dB=30,
) -> float:
    """
    distance_m: the distance between the tx and the rx
    path_loss_exponent: the path loss exponent
    sigma_dB: # shadowing standard deviation (dB), typically 2.7 to 3.5
    PL_d0_dB: # path loss (dB) at reference distance
    """
    return -(
        PL_d0_dB
        + 10 * path_loss_exponent * np.log10(distance_m)
        + gauss(0, sigma_dB)
    )


def a(N: int, theta: float) -> float:
    temp = []
    for i in range(N):
        temp.append(cmath.exp(-1j * 2 * np.pi * i * theta / 2))
    return np.reshape(np.matrix(temp), (1, N))


class H(ABC):
    def __init__(self, carrier_freq_GHz: float = 2.1e9) -> None:
        # super.__init__(self)
        self.frequece = carrier_freq_GHz
        self.c = 3e8
        self.lambda_ = self.c / self.frequece
        self.antenna_spacing = (
            0.5 * self.lambda_
        )  # the default antenna spacing is half of wavelength

    @abstractmethod
    def __call__(self):
        pass


class Hd(H):
    def __init__(
        self,
        distance: float = 51,
        num_antenna_tx: int = 10,
        num_antenna_rx: int = 1,
        path_loss_exponent=2,
    ) -> None:
        super().__init__()
        self.d = distance
        self.M = num_antenna_tx
        self.N = num_antenna_rx
        self.ple = path_loss_exponent

    def __call__(self):
        pl = PL_dB(self.d, self.ple)

        rayleigh = []
        for _ in range(500):
            temp = np.random.normal(loc=0, scale=np.sqrt(
                2) / 2, size=(self.M, 2)).view(np.complex128)
            rayleigh.append(temp.T)

        return np.sqrt(dB_to_linear(pl)) * np.mean(rayleigh, 0)


class G(H):
    def __init__(
        self,
        distance: float = 45,
        num_antenna_tx: int = 10,
        num_antenna_rx_x: int = 10,
        num_antenna_rx_y: int = 5,
        path_loss_exponent: float = 2.0,
        Rician_factor: float = 1.0,
        theta_AOAh: float = np.pi / 3,
        theta_AOAv: float = np.pi / 3,
        theta_AODb: float = np.pi / 3,
    ) -> None:
        super().__init__()
        self.d = distance
        self.M = num_antenna_tx
        self.Nx = num_antenna_rx_x
        self.Ny = num_antenna_rx_y
        self.ple = path_loss_exponent
        self.K = Rician_factor
        self.d = distance
        self.AOAh = theta_AOAh
        self.AOAv = theta_AOAv
        self.AODb = theta_AODb

    def __call__(self):
        pl = PL_dB(self.d, self.ple)
        f1 = np.sqrt(1 / (1 + (1 / self.K)))
        f2 = np.sqrt(1 / (self.K + 1))

        aNx = a(self.Nx, np.sin(self.AOAh)).getH()
        aNy = a(self.Ny, np.sin(self.AOAv)).getH()
        aM = a(self.M, np.sin(self.AODb))
        LoS = np.matmul(np.kron(aNx, aNy), aM)

        rayleigh = []
        for i in range(500):
            temp = []
            for i in range(self.Nx * self.Ny):
                temp.append(
                    np.random.normal(
                        loc=0, scale=np.sqrt(2) / 2, size=(self.M, 2)
                    ).view(np.complex128)
                )
            rayleigh.append(np.array(temp).squeeze(2))
        NLoS = np.mean(rayleigh, 0)

        return np.sqrt(dB_to_linear(pl)) * (f1 * LoS + f2 * NLoS)


class Hr(H):
    def __init__(
        self,
        distance: float = 10,
        num_antenna_tx_x: int = 10,
        num_antenna_tx_y: int = 5,
        path_loss_exponent: float = 2,
        Rician_factor: float = 1.0,
        theta_AODv: float = np.pi / 3,
        theta_AODh: float = np.pi / 3,
    ) -> None:
        super().__init__()
        self.d = distance
        self.num_antenna_rx = 1
        self.Nx = num_antenna_tx_x
        self.Ny = num_antenna_tx_y
        self.ple = path_loss_exponent
        self.K = Rician_factor
        self.d = distance
        self.AODv = theta_AODv
        self.AODh = theta_AODh

    def __call__(self):
        pl = PL_dB(self.d, self.ple)
        f1 = np.sqrt(1 / (1 + (1 / self.K)))
        f2 = np.sqrt(1 / (self.K + 1))

        aNy = a(self.Ny, np.sin(self.AODv)).getH()
        aNx = a(self.Nx, np.cos(self.AODv) * np.sin(self.AODh)).getH()
        LoS = np.kron(aNx, aNy)

        rayleigh = []
        for _ in range(500):
            rayleigh.append(
                np.random.normal(
                    loc=0, scale=np.sqrt(2) / 2, size=(self.Nx * self.Ny, 2)
                ).view(np.complex128)
            )
        NLoS = np.mean(rayleigh, 0)

        return np.sqrt(dB_to_linear(pl)) * (f1 * LoS + f2 * NLoS)


def Phi(angles):
    angles_ = []
    for i in range(len(angles)):
        angles_.append(cmath.exp(1j * angles[i]))
    shifts = np.diag(angles_)
    return shifts


def H_sum(hr, shifts, g, hd):
    reflect = np.matmul(np.matmul(hr.getH(), shifts), g)
    direct = hd
    sum = reflect + direct
    return sum


def SNR(hr, shifts, g, hd, Pmax, sigma):
    reflect = np.matmul(np.matmul(hr.getH(), shifts), g)
    direct = hd
    sum = reflect + direct
    w = np.sqrt(Pmax) * sum.getH() / np.sqrt(np.matmul(sum, sum.getH()).item())
    snr = (1 / sigma) * np.abs(np.matmul(sum, w)) ** 2
    return snr.item()
