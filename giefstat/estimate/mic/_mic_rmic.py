from minepy import MINE
import numpy as np

ALPHA = 0.6
C = 5
EPS = 1e-6


class MaximalInfoCoeff(object):
    """MIC成对检验"""

    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        self.x, self.y = x.flatten(), y.flatten()
        if mic_params is None:
            mic_params = {"alpha": ALPHA, "c": C}
        self.mic_params = mic_params

    def cal_assoc(self):
        mine = MINE(**self.mic_params)
        mine.compute_score(self.x, self.y)
        return mine.mic()


class RefinedMaximalInfoCoeff(object):
    """RMIC成对检验"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, mic_params: dict = None):
        self.x, self.y = x.flatten(), y.flatten()
        self.N = len(self.x)
        if mic_params is None:
            mic_params = {"alpha": ALPHA, "c": C}
        self.mic_params = mic_params
    
    def cal_assoc(self, rebase: bool = True):
        mine = MINE(**self.mic_params)
        mine.compute_score(self.x + EPS * np.random.random(self.N), self.y + EPS * np.random.random(self.N))
        mic = mine.mic()
        
        if not rebase:
            return mic
        base_mic = self._cal_base_mic(len(self.x))
        return (mic - base_mic) / (1.0 - base_mic)
        # return (mic - base_mic) / (1.0 - base_mic) if mic > base_mic else 0.0
    
    @staticmethod
    def _cal_base_mic(n):
        """经过数值实验测试所得"""
        return 0.9893 * np.power(n, -0.292)
        
        # B = ALPHA
        # return (0.0342 - 0.1382 * B) * np.log(n) + 1.6065 * B - 0.4814
    
        # return -0.057 * np.log(n) + 0.5134
        # return model.predict(np.array([[n]]))[0]