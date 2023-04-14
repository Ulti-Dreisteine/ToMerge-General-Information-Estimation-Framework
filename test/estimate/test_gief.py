import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.gief import MargEntropy, CondEntropy, MutualInfoGIEF, CondMutualInfoGIEF


# ---- 边际熵 --------------------------------------------------------------------------------------

def test_c():
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(0, 1, 10000)
    
    self = MargEntropy(x, "c")
    Hx = self()
    
    self = MargEntropy(y, "c")
    Hy = self()
    
    self = MargEntropy(np.c_[x, y], "c")
    Hxy = self()
    
    print(f"Hx: {Hx} \tHy: {Hy} \tHxy: {Hxy} \tMI: {Hx + Hy - Hxy}") # NOTE 此处MI计算不准确
    
    
def test_d():
    x = np.random.randint(0, 5, 10000)
    y = np.random.randint(10, 20, 10000)
    
    self = MargEntropy(x, "d")
    Hx = self()
    
    self = MargEntropy(y, "d")
    Hy = self()
    
    self = MargEntropy(np.c_[x, y], "d")
    Hxy = self()
    
    print(f"Hx: {Hx} \tHy: {Hy} \tHxy: {Hxy} \tMI: {Hx + Hy - Hxy}")  


print("测试边际熵")
print("test c")    
test_c()

print("test d")    
test_d()


# ---- 条件熵 --------------------------------------------------------------------------------------

def test_cc():
    x = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    
    self = CondEntropy(x, "c", z, "c")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_dd():
    x = np.random.randint(0, 5, 1000)
    z = np.random.randint(0, 5, 1000)
    
    self = CondEntropy(x, "d", z, "d")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_cd():
    x = np.random.normal(0, 1, 1000)
    z = np.random.randint(0, 5, 1000)
    
    self = CondEntropy(x, "c", z, "d")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_dc():
    x = np.random.randint(0, 5, 1000)
    z = np.random.normal(0, 1, 1000)
    
    self = CondEntropy(x, "d", z, "c")
    ce = self()
    
    print(f"ce: {ce}")


print("\n测试条件熵")
print("test cc")
test_cc()

print("test dd")
test_dd()

print("test cd")
test_cd()

print("test dc")
test_dc()


# ---- 互信息 --------------------------------------------------------------------------------------

def test_cc():
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(0, 1, 10000)
    
    self = MargEntropy(x, "c")
    Hx = self()
    
    self = MargEntropy(y, "c")
    Hy = self()
    
    self = MargEntropy(np.c_[x, y], "c")
    Hxy = self()
    
    self = MutualInfoGIEF(x, "c", y, "c")
    MI = self()
    
    print(f"Hx + Hy - Hxy: {Hx + Hy - Hxy} \tMI: {MI}")
    
    
def test_dd():
    x = np.random.randint(0, 5, 10000)
    y = np.random.randint(0, 5, 10000)
    
    self = MargEntropy(x, "d")
    Hx = self()
    
    self = MargEntropy(y, "d")
    Hy = self()
    
    self = MargEntropy(np.c_[x, y], "d")
    Hxy = self()
    
    self = MutualInfoGIEF(x, "d", y, "d")
    MI = self()
    
    print(f"Hx + Hy - Hxy: {Hx + Hy - Hxy} \tMI: {MI}")
    
    
def test_cd():
    x = np.random.normal(0, 1, 10000)
    y = np.random.randint(0, 5, 10000)
    
    self = MutualInfoGIEF(x, "c", y, "d")
    MI = self()
    
    print(f"MI: {MI}")
    
    
def test_dc():
    x = np.random.randint(0, 5, 10000)
    y = np.random.normal(0, 1, 10000)
    
    self = MutualInfoGIEF(x, "d", y, "c")
    MI = self()
    
    print(f"MI: {MI}")


print("\n测试互信息")
print("test cc")
test_cc()

print("test dd")
test_dd()

print("test cd")
test_cd()

print("test dc")
test_dc()


# ---- 测试条件互信息 -------------------------------------------------------------------------------

def test_ccc():
        x = np.random.normal(0, 1, 20000)
        y = np.random.normal(0, 1, 20000)
        z = np.random.normal(0, 1, 20000)
        
        self = CondMutualInfoGIEF(x, "c", y, "c", z, "c")
        cmi = self()
        
        print(f"cmi: {cmi}")
      
        
def test_cdc():
    x = np.random.normal(0, 1, 10000)
    y = np.random.randint(0, 5, 10000)
    z = np.random.normal(0, 1, 10000)
    
    self = CondMutualInfoGIEF(x, "c", y, "d", z, "c")
    cmi = self()
    
    print(f"cmi: {cmi}")
    
    
def test_cdd():
    x = np.random.normal(0, 1, 20000)
    y = np.random.randint(0, 5, 20000)
    z = np.random.randint(0, 3, 20000)
    
    self = CondMutualInfoGIEF(x, "c", y, "d", z, "d")
    cmi = self()
    
    print(f"cmi: {cmi}")


def test_ddc():
    x = np.random.randint(0, 5, 20000)
    y = np.random.randint(0, 3, 20000)
    z = np.random.normal(0, 1, 20000)
    
    self = CondMutualInfoGIEF(x, "d", y, "d", z, "c")
    cmi = self()
    
    print(f"cmi: {cmi}")


def test_ccd():
    x = np.random.normal(0, 1, 20000)
    y = np.random.normal(0, 1, 20000)
    z = np.random.randint(0, 3, 20000)
    
    self = CondMutualInfoGIEF(x, "c", y, "c", z, "d")
    cmi = self()
    
    print(f"cmi: {cmi}")
    
    
print("\n测试条件互信息")
print("test ccc")
test_ccc()

print("test cdc")
test_cdc()

print("test cdd")
test_cdd()

print("test ddc")
test_ddc()

print("test ccd")
test_ccd()