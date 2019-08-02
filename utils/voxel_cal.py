


def cal_coords(x, y, conf_dict):
    xx = cfg.H - int((y - cfg.yrange[0]) / cfg.vh)
    yy = cfg.W - int((x - cfg.xrange[0]) / cfg.vw)
    return xx, yy