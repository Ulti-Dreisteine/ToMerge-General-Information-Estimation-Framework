from ..estimate import cal_assoc, cal_cond_assoc

__all__ = ["cal_general_assoc"]


def cal_general_assoc(x, y, z, method, xtype, ytype, ztype, **kwargs):
    """关联和条件关联的通用计算"""    
    return cal_assoc(x, y, method, xtype, ytype, **kwargs) if z is None \
        else cal_cond_assoc(x, y, z, method, xtype, ytype, ztype, **kwargs)