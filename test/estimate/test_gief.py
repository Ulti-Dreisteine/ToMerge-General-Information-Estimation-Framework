import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.gief import MargEntropy, CondEntropy, MutualInfoGIEF, CondMutualInfoGIEF


# ---- 边际熵 --------------------------------------------------------------------------------------

def test_marg_c():
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(0, 1, 10000)
    
    self = MargEntropy(x, "c")
    Hx = self()
    
    self = MargEntropy(y, "c")
    Hy = self()
    
    self = MargEntropy(np.c_[x, y], "c")
    Hxy = self()
    
    print(f"Hx: {Hx} \tHy: {Hy} \tHxy: {Hxy} \tMI: {Hx + Hy - Hxy}") # NOTE 此处MI计算不准确
    
    
def test_marg_d():
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
print("test marginal c")
test_marg_c()

print("test marginal d")
test_marg_d()


# ---- 条件熵 --------------------------------------------------------------------------------------

def test_cond_cc():
    x = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    
    self = CondEntropy(x, "c", z, "c")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_cond_dd():
    x = np.random.randint(0, 5, 1000)
    z = np.random.randint(0, 5, 1000)
    
    self = CondEntropy(x, "d", z, "d")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_cond_cd():
    x = np.random.normal(0, 1, 1000)
    z = np.random.randint(0, 5, 1000)
    
    self = CondEntropy(x, "c", z, "d")
    ce = self()
    
    print(f"ce: {ce}")
    
    
def test_cond_dc():
    x = np.random.randint(0, 5, 1000)
    z = np.random.normal(0, 1, 1000)
    
    self = CondEntropy(x, "d", z, "c")
    ce = self()
    
    print(f"ce: {ce}")


print("\n测试条件熵")
print("test cond cc")
test_cond_cc()

print("test cond dd")
test_cond_dd()

print("test cond cd")
test_cond_cd()

print("test cond dc")
test_cond_dc()


# ---- 互信息 --------------------------------------------------------------------------------------

def test_mi_cc():
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
    
    
def test_mi_dd():
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
    
    
def test_mi_cd():
    x = np.random.normal(0, 1, 10000)
    y = np.random.randint(0, 5, 10000)
    
    self = MutualInfoGIEF(x, "c", y, "d")
    MI = self()
    
    print(f"MI: {MI}")
    
    
def test_mi_dc():
    x = np.random.randint(0, 5, 10000)
    y = np.random.normal(0, 1, 10000)
    
    self = MutualInfoGIEF(x, "d", y, "c")
    MI = self()
    
    print(f"MI: {MI}")


print("\n测试互信息")
print("test cond cc")
test_mi_cc()

print("test cond dd")
test_mi_dd()

print("test cond cd")
test_mi_cd()

print("test cond dc")
test_mi_dc()


# ---- 测试条件互信息 -------------------------------------------------------------------------------

def test_cmi_ccc():
    x = np.random.normal(0, 1, 20000)
    y = np.random.normal(0, 1, 20000)
    z = np.random.normal(0, 1, 20000)
    
    self = CondMutualInfoGIEF(x, "c", y, "c", z, "c")
    cmi = self()
    
    print(f"cmi: {cmi}")
      
        
def test_cmi_cdc():
    x = np.random.normal(0, 1, 10000)
    y = np.random.randint(0, 5, 10000)
    z = np.random.normal(0, 1, 10000)
    
    self = CondMutualInfoGIEF(x, "c", y, "d", z, "c")
    cmi = self()
    
    print(f"cmi: {cmi}")
    
    
def test_cmi_cdd():
    x = np.random.normal(0, 1, 20000)
    y = np.random.randint(0, 5, 20000)
    z = np.random.randint(0, 3, 20000)
    
    self = CondMutualInfoGIEF(x, "c", y, "d", z, "d")
    cmi = self()
    
    print(f"cmi: {cmi}")


def test_cmi_ddc():
    x = np.random.randint(0, 5, 20000)
    y = np.random.randint(0, 3, 20000)
    z = np.random.normal(0, 1, 20000)
    
    self = CondMutualInfoGIEF(x, "d", y, "d", z, "c")
    cmi = self()
    
    print(f"cmi: {cmi}")


def test_cmi_ccd():
    x = np.random.normal(0, 1, 20000)
    y = np.random.normal(0, 1, 20000)
    z = np.random.randint(0, 3, 20000)
    
    self = CondMutualInfoGIEF(x, "c", y, "c", z, "d")
    cmi = self()
    
    print(f"cmi: {cmi}")
    
    
print("\n测试条件互信息")
print("test cmi ccc")
test_cmi_ccc()

print("test cmi cdc")
test_cmi_cdc()

print("test cmi cdd")
test_cmi_cdd()

print("test cmi ddc")
test_cmi_ddc()

print("test cmi ccd")
test_cmi_ccd()